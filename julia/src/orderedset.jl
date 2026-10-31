# OrderedSet port (rivgraph/ordered_set.py)
# =========================================
# A faithful port of RivGraph's OrderedSet: insertion-ordered set backed by an
# items list plus a value->index map. Only the operations the walker and
# cleanup paths use are ported (add, index, discard, remove, pop, iteration,
# len, membership, equality). Python raises KeyError from `index`/`remove`/
# `pop`-on-empty; the port matches.

mutable struct OrderedSet{K}
    items::Vector{K}
    map::Dict{K,Int}
end

OrderedSet{K}() where {K} = OrderedSet{K}(Vector{K}(), Dict{K,Int}())
OrderedSet() = OrderedSet{Any}()

"OrderedSet(iterable): unique-append in iteration order."
function OrderedSet(items)
    s = OrderedSet{eltype(items)}()
    for it in items
        oset_add!(s, it)
    end
    return s
end

"OrderedSet{K}(iterable): typed construction, unique-append in iteration order
(Python OrderedSet(list) semantics — later duplicates are dropped, order kept)."
function OrderedSet{K}(items) where {K}
    s = OrderedSet{K}()
    for it in items
        oset_add!(s, convert(K, it))
    end
    return s
end

Base.length(s::OrderedSet) = length(s.items)
Base.isempty(s::OrderedSet) = isempty(s.items)
Base.eltype(::Type{OrderedSet{K}}) where {K} = K

"oset_copy(s): order-preserving copy of the set structure (Python OrderedSet(s) semantics)."
function oset_copy(s::OrderedSet{K}) where {K}
    OrderedSet{K}(copy(s.items), copy(s.map))
end

function Base.iterate(s::OrderedSet{K}, state::Int = 1) where {K}
    state > length(s.items) && return nothing
    return (s.items[state], state + 1)
end

"OrderedSet.add: returns the (1-based) index of the element, new or existing."
function oset_add!(s::OrderedSet{K}, key::K) where {K}
    haskey(s.map, key) && return s.map[key]
    push!(s.items, key)
    s.map[key] = length(s.items)
    return length(s.items)
end

"OrderedSet.index — KeyError when absent."
function oset_index(s::OrderedSet{K}, key::K) where {K}
    haskey(s.map, key) || throw(KeyError(key))
    return s.map[key]
end

"OrderedSet.discard — silent when absent."
function oset_discard!(s::OrderedSet{K}, key::K) where {K}
    haskey(s.map, key) || return false
    i = s.map[key]
    deleteat!(s.items, i)
    delete!(s.map, key)
    for (k, v) in s.map
        v >= i && (s.map[k] = v - 1)
    end
    return true
end

"OrderedSet.remove — KeyError when absent."
function oset_remove!(s::OrderedSet{K}, key::K) where {K}
    oset_discard!(s, key) && return s
    throw(KeyError(key))
end

"OrderedSet.pop — removes and returns the last element; KeyError when empty."
function oset_pop!(s::OrderedSet{K}) where {K}
    isempty(s.items) && throw(KeyError("pop from an empty OrderedSet"))
    key = s.items[end]
    delete!(s.map, key)
    pop!(s.items)
    return key
end

function Base.in(key::K, s::OrderedSet{K}) where {K}
    return haskey(s.map, key)
end

"Order-sensitive equality (matches the Python class and the parity tests)."
function Base.:(==)(a::OrderedSet, b::OrderedSet)
    length(a.items) == length(b.items) || return false
    for (x, y) in zip(a.items, b.items)
        x == y || return false
    end
    return true
end
