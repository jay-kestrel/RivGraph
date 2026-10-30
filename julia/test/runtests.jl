# Parity suite for the Julia port of RivGraph's core numeric functionality.
#
# Oracles are Python-reference fixtures dumped with PYDELTARCM_NATIVE=0
# (the Rust native walker was force-disabled during dumping so every
# expected graph traces to `_skel_to_graph_python`). Fixtures live in
# test/fixtures/{walker,widths,delta} plus CPython set traces in
# test/fixtures/settraces; test/fixtures/manifest.json carries shapes,
# dtypes, sha256s, and per-case parameters.

using Test
using RivGraphJulia
using RivGraphJulia: CPythonSet, OrderedSet,
                     py_add!, py_discard!, py_remove!, py_pop!,
                     py_or, py_and, py_difference, _merge!, _difference_update!,
                     read_packed_bool_matrix, read_f64_matrix,
                     skel_to_graph, link_widths_and_lengths!,
                     solve_adjacency_steady_state, delta_subN_F, nl_entropy_rate
using JSON

FIX = joinpath(@__DIR__, "fixtures")
MANIFEST_JSON = joinpath(FIX, "manifest.json")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"JSON value -> Int key (int traces) or (Int,Int) key (tuple traces)."
to_key(e::Int, kind::String) = Int(e)
to_key(e::Vector, kind::String) = (Int(e[1]), Int(e[2]))

"Vector{Any} of ints -> Vector{Int}."
intvec(xs) = Int[x for x in xs]

"Vector{Any} of Vector{Any} -> Vector{Vector{Int}}."
intvecvec(xs) = Vector{Int}[intvec(row) for row in xs]

"Build a CPythonSet by sequential adds, matching CPython's set(iterable) generic path."
function py_set_from_list(keys::Vector{K}) where {K}
    s = CPythonSet{K}()
    for k in keys
        py_add!(s, k)
    end
    return s
end

"Collect a set in its internal table/insertion order (the parity-relevant order)."
elements(s) = collect(s)

# ---------------------------------------------------------------------------
# 1. CPythonSet trace replay (int keys, tuple keys, big set)
# ---------------------------------------------------------------------------

"Build `other` with the replay set's element type K (empty lists must not widen to Any)."
function other_set(op::AbstractDict{String,Any}, kind::String, ::Type{K}) where {K}
    o = CPythonSet{K}()
    for e in op["other"]
        py_add!(o, to_key(e, kind))
    end
    return o
end

function replay_op!(s::CPythonSet{K}, op::AbstractDict{String,Any}, kind::String) where {K}
    opname = op["op"]
    if opname == "add"
        py_add!(s, to_key(op["arg"], kind))
    elseif opname == "discard"
        py_discard!(s, to_key(op["arg"], kind))
    elseif opname == "remove"
        if haskey(op, "result")
            # empty-set remove: must raise KeyError ("ok" is unreachable by construction)
            @test op["result"] == "KeyError"
            @test_throws KeyError py_remove!(s, to_key(op["arg"], kind))
        else
            py_remove!(s, to_key(op["arg"], kind))
        end
    elseif opname == "pop"
        res = op["result"]
        if res isa String
            @test res == "KeyError"  # "ok(empty)" is unreachable by construction
            @test_throws KeyError py_pop!(s)
        else
            popped = py_pop!(s)
            @test popped == to_key(res, kind)
        end
    elseif opname == "iter"
        @test elements(s) == [to_key(e, kind) for e in op["result"]]
    elseif opname == "contains"
        @test (to_key(op["arg"], kind) in s) == op["result"]
    elseif opname in ("or", "sub", "and")
        other = other_set(op, kind, K)
        res = opname == "or" ? py_or(s, other) :
              opname == "sub" ? py_difference(s, other) : py_and(s, other)
        @test elements(res) == [to_key(e, kind) for e in op["result"]]
        return res
    elseif opname in ("ior", "isub")
        other = other_set(op, kind, K)
        opname == "ior" ? _merge!(s, other) : _difference_update!(s, other)
        @test elements(s) == [to_key(e, kind) for e in op["result"]]
    elseif opname == "pop_drain"
        drain = K[]
        while length(s) > 0
            push!(drain, py_pop!(s))
        end
        @test drain == [to_key(e, kind) for e in op["result"]]
    else
        error("unknown trace op: $opname")
    end
    return s
end

function replay_trace(tr::AbstractDict{String,Any})
    kind = tr["kind"]::String
    K = kind == "tuple" ? Tuple{Int64,Int64} : Int64
    # Typed comprehension: an empty init would otherwise widen to Vector{Any}
    # and build a CPythonSet{Any}.
    s = py_set_from_list(K[to_key(e, kind) for e in tr["init"]])::CPythonSet{K}
    for op in tr["ops"]
        s = replay_op!(s, op, kind)::CPythonSet{K}
    end
end

@testset "CPythonSet trace replay" begin
    for f in ("set_traces_int.json", "set_traces_tuple.json", "set_traces_big.json")
        traces = JSON.parsefile(joinpath(FIX, "settraces", f))
        @testset "$(f)" begin
            for tr in traces
                @testset "$(tr["trace_id"])" begin
                    replay_trace(tr)
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# 2. Walker parity (ordered exact, OrderedSet-order-sensitive)
# ---------------------------------------------------------------------------

function compare_ser_graph(actual, exp_values::AbstractDict{String,Any}, kind::String)
    # id: OrderedSet of ints (or nothing)
    if haskey(exp_values, "id") && !(exp_values["id"] === nothing)
        @test collect(actual.id) == intvec(exp_values["id"])
    end
    if haskey(exp_values, "idx") && !(exp_values["idx"] === nothing)
        if actual.idx isa OrderedSet
            @test collect(actual.idx) == intvec(exp_values["idx"])
        else
            @test actual.idx == intvecvec(exp_values["idx"])
        end
    end
    if haskey(exp_values, "conn") && !(exp_values["conn"] === nothing)
        @test actual.conn == intvecvec(exp_values["conn"])
    end
end

@testset "walker parity (Python reference oracle)" begin
    manifest = JSON.parsefile(MANIFEST_JSON)
    walker_cases = sort(collect(keys(manifest["inputs"])))
    @test length(walker_cases) >= 40
    for name in walker_cases
        startswith(name, "walker/") || continue
        meta = manifest["inputs"][name]
        d = joinpath(FIX, name)
        shape = meta["Iskel"]["shape"]
        Iskel = read_packed_bool_matrix(joinpath(d, "Iskel.npy"), shape[1], shape[2])
        @testset "$name" begin
            links, nodes = skel_to_graph(Iskel)
            lexp = JSON.parsefile(joinpath(d, "links.json"))["values"]
            nexp = JSON.parsefile(joinpath(d, "nodes.json"))["values"]
            compare_ser_graph(links, lexp, "links")
            compare_ser_graph(nodes, nexp, "nodes")
        end
    end
end

# ---------------------------------------------------------------------------
# 3. link_widths_and_lengths parity
# ---------------------------------------------------------------------------

"Rebuild a Links from a ser_graph payload (walker-stage fields only)."
function links_from_ser(v::AbstractDict{String,Any})
    vals = v["values"]
    links = RivGraphJulia.Links()
    links.idx = intvecvec(vals["idx"])
    links.conn = intvecvec(vals["conn"])
    links.id = OrderedSet{Int64}(intvec(vals["id"]))
    return links
end

const FLOAT_KEYS = ("len", "wid", "wid_adj", "wid_med", "len_adj", "sinuosity")

"NaN-aware relative comparison; floor to 1e-12 relative (or 1e-12 absolute near zero)."
function float_close(a::Float64, b::Float64)
    isnan(a) && isnan(b) && return true
    isnan(a) != isnan(b) && return false
    return abs(a - b) <= 1e-12 * max(1.0, abs(b))
end

@testset "link widths and lengths parity" begin
    manifest = JSON.parsefile(MANIFEST_JSON)
    for name in sort(filter(k -> startswith(k, "widths/"), collect(keys(manifest["inputs"]))))
        meta = manifest["inputs"][name]
        d = joinpath(FIX, name)
        links = links_from_ser(JSON.parsefile(joinpath(d, "links_in.json")))
        Idt = read_f64_matrix(joinpath(d, "Idt.npy.gz"))
        pixlen = Float64(meta["pixlen"])
        expected = JSON.parsefile(joinpath(d, "links_out.json"))["values"]
        @testset "$name (pixlen=$pixlen)" begin
            link_widths_and_lengths!(links, Idt, pixlen)
            for (i, key) in enumerate(FLOAT_KEYS)
                got = getproperty(links, Symbol(key))
                exp = Float64[Float64(x) for x in expected[key]]
                @test length(got) == length(exp)
                @test all(float_close(got[i2], exp[i2]) for i2 in 1:length(exp))
            end
            # per-pixel width profiles: exact (same integer grid traversal)
            got_wp = links.wid_pix
            exp_wp = [[Float64(x) for x in row] for row in expected["wid_pix"]]
            @test length(got_wp) == length(exp_wp)
            @test all(got_wp[i] == exp_wp[i] for i in 1:length(exp_wp))
        end
    end
end

# ---------------------------------------------------------------------------
# 4. delta steady state + entropy rate parity
# ---------------------------------------------------------------------------

@testset "delta steady-state and entropy-rate parity" begin
    manifest = JSON.parsefile(MANIFEST_JSON)
    for name in sort(filter(k -> startswith(k, "delta/"), collect(keys(manifest["inputs"]))))
        meta = manifest["inputs"][name]
        d = joinpath(FIX, name)
        A = read_f64_matrix(joinpath(d, "A.npy.gz"))
        expected = JSON.parsefile(joinpath(d, "expected.json"))
        epsilon = Float64(meta["epsilon"])
        @testset "$name (n=$(meta["n"]))" begin
            res = solve_adjacency_steady_state(A; atol=epsilon)
            # node flux: bit-exact against the little-endian double dump
            # (struct.pack("<d", x).hex() — bytes are already the little-endian
            # memory layout, which matches the native float bit pattern)
            for (i, hexbits) in enumerate(expected["node_flux_sha_bits"])
                b = UInt8[parse(UInt8, hexbits[j:j+1], base=16) for j in 1:2:16]
                exp_bits = reinterpret(UInt64, b)[1]
                got_bits = reinterpret(UInt64, res.node_flux[i])
                @test got_bits == exp_bits
            end
            SubNexp = Float64[expected["SubN"][i][j]
                              for i in 1:length(expected["SubN"]),
                                  j in 1:length(expected["SubN"][1])]
            Pexp = Float64[expected["P"][i][j]
                           for i in 1:length(expected["P"]),
                               j in 1:length(expected["P"][1])]
            @test res.subnetwork_membership == SubNexp
            @test res.transition_matrix ≈ Pexp atol = 1e-12
            @test res.apex == Int(meta["apex"])
            @test res.outlets == intvec(meta["outlets"])
            nER = nl_entropy_rate(A)
            @test float_close(nER, Float64(meta["nER"]))
            # delta_subN_F composite: same F and SubN (returns (F, SubN) per
            # the Python contract delta_metrics.delta_subN_F)
            F, SubN = delta_subN_F(A; epsilon=epsilon)
            @test all(float_close(F[i], res.node_flux[i]) for i in 1:length(F))
            @test SubN == res.subnetwork_membership
        end
    end
end
