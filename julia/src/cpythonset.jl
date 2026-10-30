# CPython `set` emulation (CPython 3.12.16, setobject.c)
# ======================================================
# RivGraph's walker uses Python sets as its workhorse container for pixel
# indices, and link/node ids end up assigned according to set iteration
# order (hash-table slot order), pop order (finger-based circular scan),
# and the table layouts produced by resize/merge/difference. Reproducing
# the walker bug-for-bug therefore requires reproducing CPython's set
# internals, not just set *semantics*.
#
# This module ports the exact table layout and algorithms from
# CPython 3.12.16 (probing with 9-slot linear probes + perturb jumps,
# resize policy `used>50000 ? used*2 : used*4`, dummy entries, the
# finger-based pop, and the merge/copy/difference fast paths). Key hashing
# follows CPython for the two key types the walker uses:
#
#   * flat pixel indices  -> Int64   (hash(n) == n, except hash(-1) == -2)
#   * (from, to) id pairs -> Tuple{Int64,Int64} (tuplehash, xxHash-based)
#
# The port is validated by replaying recorded Python traces (see
# test/cpythonset_replay.jl); the trace dumper records every observable
# (iteration order, pop sequence, table size, finger) from real Python.

# ---------------------------------------------------------------------------
# Key hashing (CPython longobject/pyhash + tupleobject.c tuplehash)
# ---------------------------------------------------------------------------

const XXPRIME_1 = UInt64(11400714785074694791)
const XXPRIME_2 = UInt64(14029467366897019727)
const XXPRIME_5 = UInt64(2870177450012600261)

@inline rotl31(x::UInt64) = (x << 31) | (x >> 33)

"CPython integer hash: identity, except hash(-1) == -2 (sentinel collision)."
@inline py_hash(x::Int64) = ifelse(x == -1, Int64(-2), x)
@inline py_hash(x::Integer) = py_hash(Int64(x))

"""
    py_hash((a, b)) -> Int64

CPython `tuplehash` for 2-tuples of ints (tupleobject.c): xxHash-style
accumulation of lane hashes over `_PyHASH_XXPRIME_5`, rotated by 31 bits and
multiplied per lane; finalized with `len ^ (XXPRIME_5 ^ 3527539)`; the
all-bits-one result is remapped to 1546275796.
"""
function py_hash(t::Tuple{Int64,Int64})
    acc = XXPRIME_5
    for lane in (UInt64(py_hash(t[1])), UInt64(py_hash(t[2])))
        acc += lane * XXPRIME_2
        acc = rotl31(acc)
        acc = acc * XXPRIME_1
    end
    acc += UInt64(2) ⊻ (XXPRIME_5 ⊻ UInt64(3527539))
    acc == typemax(UInt64) && return Int64(1546275796)
    return reinterpret(Int64, acc)
end
# (Julia's `Int` IS `Int64`; any tuple-literal of machine ints matches the
# single canonical method above — no second definition, which precompilation
# forbids.)

# ---------------------------------------------------------------------------
# Table entry model
# ---------------------------------------------------------------------------
# slots[i] === nothing && hashes[i] == 0  -> virgin slot
# slots[i] === nothing && hashes[i] == -1 -> dummy slot
# otherwise                               -> active entry (key, hash)
#
# (No real key can collide with these sentinels: integer keys hash to
# themselves with -1 remapped to -2, and tuple hashes remap the sentinel.)

const _MINSIZE = 8
const _LINEAR_PROBES = 9
const _PERTURB_SHIFT = 5

@inline _keyeq(a::K, b::K) where {K<:Union{Int64,Tuple{Int64,Int64}}} = a == b

mutable struct CPythonSet{K}
    slots::Vector{Union{Nothing,K}}
    hashes::Vector{Int64}
    fill::Int
    used::Int
    mask::Int
    finger::Int
end

"Fresh CPython set: PySet_MINSIZE(8) slots, all virgin, finger 0."
CPythonSet{K}() where {K} = CPythonSet{K}(Vector{Union{Nothing,K}}(nothing, _MINSIZE), zeros(Int64, _MINSIZE), 0, 0, _MINSIZE - 1, 0)

"Python `set(iterable)`: add each element in iteration order."
function CPythonSet(keys::AbstractVector{K}) where {K}
    s = CPythonSet{K}()
    for k in keys
        py_add!(s, k)
    end
    s
end

Base.length(s::CPythonSet) = s.used
Base.isempty(s::CPythonSet) = s.used == 0
Base.eltype(::Type{CPythonSet{K}}) where {K} = K

"Python `set.__iter__`: table slot order (virgin and dummy slots skipped)."
function Base.iterate(s::CPythonSet{K}, state::Int = 1) where {K}
    i = state
    while i <= length(s.slots)
        s.slots[i] === nothing && (i += 1; continue)
        return (s.slots[i]::K, i + 1)
    end
    return nothing
end

"Slot-ordered snapshot (matches `list(pyset)` and iteration order)."
py_elements(s::CPythonSet{K}) where {K} = collect(K, s)

# ---------------------------------------------------------------------------
# set_lookkey (setobject.c): probe for an active entry or the first
# acceptable insertion slot. Returns (slot_1based, found_active).
# ---------------------------------------------------------------------------

function _lookkey(s::CPythonSet{K}, key::K, hash::Int64) where {K}
    mask = UInt64(s.mask)
    perturb = reinterpret(UInt64, hash)
    i = reinterpret(UInt64, hash) & mask
    while true
        probes = (i + _LINEAR_PROBES <= mask) ? _LINEAR_PROBES : 0
        j = i
        k = 0
        while true
            slot = j + 1
            sh = s.hashes[slot]
            if sh == 0 && s.slots[slot] === nothing
                return (slot, false)                 # virgin: end of chain
            elseif sh == hash
                key0 = s.slots[slot]::K
                _keyeq(key0, key) && return (slot, true)
            end
            # dummy slots and non-matching actives fall through
            if k >= probes
                break
            end
            j += 1
            k += 1
        end
        perturb >>= _PERTURB_SHIFT
        i = (i * 5 + 1 + perturb) & mask
    end
end

# ---------------------------------------------------------------------------
# set_table_resize (setobject.c)
# ---------------------------------------------------------------------------

function _resize!(s::CPythonSet{K}, minused::Int) where {K}
    newsize = _MINSIZE
    while newsize <= minused
        newsize <<= 1
    end
    newmask = newsize - 1
    newslots = Vector{Union{Nothing,K}}(nothing, newsize)
    newhashes = zeros(Int64, newsize)
    had_dummies = (s.fill != s.used)
    for i in 1:length(s.slots)
        k = s.slots[i]
        k === nothing && continue  # virgin and dummy slots both store nothing
        _insert_clean!(newslots, newhashes, UInt64(newmask), k, s.hashes[i])
    end
    s.slots = newslots
    s.hashes = newhashes
    s.mask = newmask
    had_dummies && (s.fill = s.used)
    return s
end

"set_insert_clean: insert a known-absent key; virgin slots only; exact probe order."
@inline function _insert_clean!(slots::Vector{Union{Nothing,K}}, hashes::Vector{Int64}, mask::UInt64,
                                key::K, hash::Int64) where {K}
    perturb = reinterpret(UInt64, hash)
    i = reinterpret(UInt64, hash) & mask
    while true
        if slots[i + 1] === nothing
            slots[i + 1] = key
            hashes[i + 1] = hash
            return
        end
        if i + _LINEAR_PROBES <= mask
            # C advances only the entry pointer through the window; the base i
            # stays put, so the perturb jump below uses the ORIGINAL slot.
            j = i
            for _ in 1:_LINEAR_PROBES
                j += 1
                if slots[j + 1] === nothing
                    slots[j + 1] = key
                    hashes[j + 1] = hash
                    return
                end
            end
        end
        perturb >>= _PERTURB_SHIFT
        i = (i * 5 + 1 + perturb) & mask
    end
end

# ---------------------------------------------------------------------------
# set_add_entry (setobject.c)
# ---------------------------------------------------------------------------

"Python `set.add` — exact CPython insertion with resize policy."
function py_add!(s::CPythonSet{K}, key::K) where {K}
    hash = py_hash(key)
    mask = UInt64(s.mask)
    perturb = reinterpret(UInt64, hash)
    i = reinterpret(UInt64, hash) & mask
    freeslot = 0  # 1-based slot of first dummy seen; 0 = none
    while true
        probes = (i + _LINEAR_PROBES <= mask) ? _LINEAR_PROBES : 0
        j = i
        k = 0
        while true
            slot = j + 1
            sh = s.hashes[slot]
            if sh == 0 && s.slots[slot] === nothing
                # found_unused_or_dummy
                if freeslot != 0
                    s.used += 1
                    s.slots[freeslot] = key
                    s.hashes[freeslot] = hash
                    return s
                end
                # found_unused
                s.fill += 1
                s.used += 1
                s.slots[slot] = key
                s.hashes[slot] = hash
                if !(UInt64(s.fill) * 5 < mask * 3)
                    _resize!(s, s.used > 50000 ? s.used * 2 : s.used * 4)
                end
                return s
            elseif sh == hash
                key0 = s.slots[slot]::K
                _keyeq(key0, key) && return s
            elseif sh == -1
                freeslot = slot  # C reassigns per dummy: LAST dummy in the probe wins
            end
            if k >= probes
                break
            end
            j += 1
            k += 1
        end
        perturb >>= _PERTURB_SHIFT
        i = (i * 5 + 1 + perturb) & mask
    end
end

# ---------------------------------------------------------------------------
# membership / discard / remove / pop / clear
# ---------------------------------------------------------------------------

"Python `key in set`."
function Base.in(key::K, s::CPythonSet{K}) where {K}
    _, active = _lookkey(s, key, py_hash(key))
    return active
end

"Python `set.discard` (no error when absent). Returns true when removed."
function py_discard!(s::CPythonSet{K}, key::K) where {K}
    slot, active = _lookkey(s, key, py_hash(key))
    active || return false
    s.slots[slot] = nothing
    s.hashes[slot] = -1
    s.used -= 1
    return true
end

"Python `set.remove` (KeyError when absent)."
function py_remove!(s::CPythonSet{K}, key::K) where {K}
    py_discard!(s, key) && return s
    throw(KeyError(key))
end

"""
    py_pop!(s) -> key

Python `set.pop`: circular scan starting at `finger & mask`, first live slot;
removes it (marks the slot as a dummy), advances the finger one past it.
"""
function py_pop!(s::CPythonSet{K}) where {K}
    s.used == 0 && throw(KeyError("pop from an empty set"))
    n = length(s.slots)
    i = (s.finger & s.mask) + 1  # 1-based starting slot
    while s.slots[i] === nothing
        i += 1
        i > n && (i = 1)
    end
    key = s.slots[i]::K
    s.slots[i] = nothing
    s.hashes[i] = -1
    s.used -= 1
    s.finger = i  # CPython stores (entry - table) + 1 == the 1-based slot
    return key
end

"Python `set.clear` (set_clear_internal)."
function py_clear!(s::CPythonSet{K}) where {K}
    if UInt64(s.fill) * 5 < UInt64(s.mask) * 3
        fill!(s.slots, nothing)
        fill!(s.hashes, 0)
        s.fill = 0
        s.used = 0
    else
        s.slots = Vector{Union{Nothing,K}}(nothing, _MINSIZE)
        s.hashes = zeros(Int64, _MINSIZE)
        s.fill = 0
        s.used = 0
        s.mask = _MINSIZE - 1
        s.finger = 0
    end
    return s
end

# ---------------------------------------------------------------------------
# copy / merge / union / difference / intersection
# ---------------------------------------------------------------------------

"Python `set.copy`: make_new_set(type, so) -> update_internal -> set_merge."
function py_copy(s::CPythonSet{K}) where {K}
    out = CPythonSet{K}()
    _merge!(out, s)
    return out
end

"Python `set.update(other_set)` / merge fast paths (set_merge)."
function _merge!(so::CPythonSet{K}, other::CPythonSet{K}) where {K}
    (so === other || other.used == 0) && return so
    if (so.fill + other.used) * 5 >= so.mask * 3
        _resize!(so, (so.used + other.used) * 2)
    end
    if so.fill == 0 && so.mask == other.mask && other.fill == other.used
        # slot-for-slot copy (preserves the other table's layout exactly)
        for i in 1:length(other.slots)
            k = other.slots[i]
            if k !== nothing
                so.slots[i] = k
                so.hashes[i] = other.hashes[i]
            end
        end
        so.fill = other.fill
        so.used = other.used
        return so
    end
    if so.fill == 0
        so.fill = other.used
        so.used = other.used
        mask = UInt64(so.mask)
        for i in 1:length(other.slots)
            k = other.slots[i]
            if k !== nothing
                _insert_clean!(so.slots, so.hashes, mask, k, other.hashes[i])
            end
        end
        return so
    end
    for i in 1:length(other.slots)
        k = other.slots[i]
        if k !== nothing
            py_add!(so, k)
        end
    end
    return so
end

"Python `a | b` (set_or): copy(a) then update with b."
function py_or(a::CPythonSet{K}, b::CPythonSet{K}) where {K}
    out = py_copy(a)
    _merge!(out, b)
    return out
end

"Python `a & b` (set_intersection): iterate the smaller set, probe the larger."
function py_and(a::CPythonSet{K}, b::CPythonSet{K}) where {K}
    a === b && return py_copy(a)
    so, other = a, b
    if length(other) > length(so)
        so, other = other, so
    end
    result = CPythonSet{K}()
    for k in other
        in(k, so) && py_add!(result, k)
    end
    return result
end

"Python `a - b` (set_difference)."
function py_difference(a::CPythonSet{K}, b::CPythonSet{K}) where {K}
    if (length(a) >> 2) > length(b)
        out = py_copy(a)
        _difference_update!(out, b)
        return out
    end
    result = CPythonSet{K}()
    for k in a
        in(k, b) || py_add!(result, k)
    end
    return result
end

"Python `a -= b` (set_difference_update_internal)."
function _difference_update!(so::CPythonSet{K}, other::CPythonSet{K}) where {K}
    so === other && return py_clear!(so)
    other0 = other
    if (length(other0) >> 3) > length(so)
        other0 = py_and(so, other0)  # intersection replaces the other set
    end
    for k in other0
        py_discard!(so, k)
    end
    # If more than 1/4 are dummies, resize them away (integer comparison).
    if (so.fill - so.used) > (so.mask >> 2)
        _resize!(so, so.used > 50000 ? so.used * 2 : so.used * 4)
    end
    return so
end

"Python `set(iterable)` for non-set iterables: add each in iteration order."
function py_set_from(keys::AbstractVector{K}) where {K}
    s = CPythonSet{K}()
    for k in keys
        py_add!(s, k)
    end
    return s
end

"Python `set(existing_set)`: structure-preserving copy (Python set(x) accepts sets too)."
function py_set_from(s::CPythonSet{K}) where {K}
    return py_copy(s)
end

"Python set equality: same size and every element of a present in b."
function py_seteq(a::CPythonSet{K}, b::CPythonSet{K}) where {K}
    a.used == b.used || return false
    for k in a
        in(k, b) || return false
    end
    return true
end
