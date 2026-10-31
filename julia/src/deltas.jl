# Delta DAG steady-state solver + entropy rate (bug-for-bug)
# ==========================================================
# Ports of:
#   - _delta_metrics_core.solve_adjacency_steady_state (:379)
#   - _delta_metrics_core._column_normalize_adjacency (:366)
#   - delta_metrics.delta_subN_F (:760) and nl_entropy_rate (:778)
# Convention: A[v, u] is the edge score from upstream node u to downstream
# node v (historical delta-metrics orientation). All sums are naive
# left-to-right accumulations in the source's iteration order; NumPy pairwise
# summation differs only at ~1e-16 relative, within the 1e-10 mass-balance
# and 1e-12 flux tolerances.

using LinearAlgebra

"""
    _np_sum_contiguous(x) -> Float64

Bit-faithful port of NumPy's pairwise_sum for a contiguous float64 vector
(numpy/core/src/umath/loops.c.src): naive below 8 elements; 8 partial
accumulators combined as ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)) up to the 128
block size; recursive bisection (halves re-aligned to multiples of 8) beyond.
Julia's `sum` reassociates differently, which shifts low bits and breaks the
bit-exact flux gates.
"""
function _np_sum_contiguous(x::AbstractVector{Float64})
    n = length(x)
    n == 0 && return 0.0
    if n < 8
        res = 0.0
        for i in 1:n
            res += x[i]
        end
        return res
    elseif n <= 128
        r0 = x[1]; r1 = x[2]; r2 = x[3]; r3 = x[4]
        r4 = x[5]; r5 = x[6]; r6 = x[7]; r7 = x[8]
        i = 9
        stop = n - (n % 8)
        while i <= stop
            r0 += x[i]; r1 += x[i+1]; r2 += x[i+2]; r3 += x[i+3]
            r4 += x[i+4]; r5 += x[i+5]; r6 += x[i+6]; r7 += x[i+7]
            i += 8
        end
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i <= n
            res += x[i]
            i += 1
        end
        return res
    else
        n2 = n ÷ 2
        n2 -= n2 % 8
        return _np_sum_contiguous(view(x, 1:n2)) +
               _np_sum_contiguous(view(x, n2+1:n))
    end
end

"""Result of solve_adjacency_steady_state (mirrors the Python dataclass)."""
struct AdjacencySteadyStateResult
    node_flux::Vector{Float64}
    subnetwork_membership::Matrix{Float64}
    apex::Int
    outlets::Vector{Int}
    transition_matrix::Matrix{Float64}
end

"""
    _column_normalize_adjacency(A) -> Matrix{Float64}

Port of _column_normalize_adjacency: divide each nonzero column by its sum;
zero columns (outlets) are left zero.
"""
function _column_normalize_adjacency(A::Matrix{Float64})
    n, m = size(A)
    (n == m) || throw(ErrorException("Adjacency matrix must be square."))
    # np.sum(A, axis=0): sequential row accumulation, elementwise per column
    colsum = zeros(Float64, m)
    for i in 1:n
        for j in 1:m
            colsum[j] += A[i, j]
        end
    end
    P = zeros(Float64, n, m)
    for j in 1:m
        if colsum[j] > 0
            for i in 1:n
                P[i, j] = A[i, j] / colsum[j]
            end
        end
    end
    return P
end

"""
    solve_adjacency_steady_state(A; atol=1e-12) -> AdjacencySteadyStateResult

Port of _delta_metrics_core.solve_adjacency_steady_state. Validates the
adjacency, zeroes values <= atol, finds the unique apex (no incoming) and
outlets (no outgoing), runs FIFO Kahn topological sort, propagates fluxes
forward from the apex, computes backward subnetwork membership, and enforces
the outlet mass-balance check (1e-10). Ported bug-for-bug, including the
`F[u] == 0` skip and the reversed-order membership propagation.
"""
function solve_adjacency_steady_state(A::Matrix{Float64}; atol::Float64 = 1e-12)
    n, m = size(A)
    (n == m) || throw(ErrorException("Adjacency matrix must be square."))
    for v in A
        isnan(v) && throw(ErrorException("Adjacency matrix contains non-finite values."))
        isinf(v) && throw(ErrorException("Adjacency matrix contains non-finite values."))
    end
    for i in 1:n, j in 1:n
        A[i, j] < -atol &&
            throw(ErrorException("Adjacency matrix contains negative values."))
    end

    # zero-out tiny numerical noise
    A = copy(A)
    for i in 1:n, j in 1:n
        abs(A[i, j]) <= atol && (A[i, j] = 0.0)
    end

    # rows with no incoming define the unique apex; columns with no outgoing
    # define outlets
    row_sum = Float64[_np_sum_contiguous(A[i, :]) for i in 1:n]   # np.sum(A, axis=1)
    col_sum = zeros(Float64, n)                                    # np.sum(A, axis=0)
    for i in 1:n, j in 1:n
        col_sum[j] += A[i, j]
    end
    apexes = [i for i in 1:n if abs(row_sum[i]) <= atol]
    length(apexes) != 1 && throw(ErrorException("The graph contains more than one apex."))
    apex = apexes[1]
    outlets = [j for j in 1:n if abs(col_sum[j]) <= atol]
    isempty(outlets) && throw(ErrorException("Adjacency matrix has no outlet nodes."))

    # DAG structure from the nonzero pattern of A[v, u]
    out_neighbors = Vector{Vector{Int}}(undef, n)
    for u in 1:n
        out_neighbors[u] = Int[v for v in 1:n if abs(A[v, u]) > atol]  # raster order
    end
    indeg = Int[count(abs(A[u, v]) > atol for v in 1:n) for u in 1:n]

    order = Int[]
    queue = Int[u for u in 1:n if indeg[u] == 0]   # dict order == ascending
    head = 1
    while head <= length(queue)                    # deque popleft (FIFO)
        u = queue[head]
        head += 1
        push!(order, u)
        for v in out_neighbors[u]
            indeg[v] -= 1
            if indeg[v] == 0
                push!(queue, v)
            end
        end
    end
    length(order) != n && throw(ErrorException("Adjacency matrix is not acyclic."))

    P = _column_normalize_adjacency(A)

    F = zeros(Float64, n)
    F[apex] = 1.0
    for u in order
        F[u] == 0 && continue
        for v in out_neighbors[u]
            F[v] += F[u] * P[v, u]
        end
    end

    subn = zeros(Float64, n, length(outlets))
    for (j, outlet) in enumerate(outlets)
        subn[outlet, j] = 1.0
    end
    outlet_set = Set{Int}(outlets)
    for u in reverse(order)
        u in outlet_set && continue
        for v in out_neighbors[u]
            for j in 1:length(outlets)
                subn[u, j] += P[v, u] * subn[v, j]
            end
        end
    end

    # QA only; outlet flux should sum to one.
    outlet_flux_sum = 0.0
    for o in outlets
        outlet_flux_sum += F[o]
    end
    mbe = abs(outlet_flux_sum - 1.0)
    mbe > max(atol, 1e-10) &&
        throw(ErrorException("Adjacency steady-state mass balance failed; abs diff=$mbe."))

    # Python contract: apex/outlets are 0-based indices; internal indexing
    # stays 1-based, so convert only at the boundary.
    return AdjacencySteadyStateResult(F, subn, apex - 1, outlets .- 1, P)
end

"""
    delta_subN_F(A; epsilon=1e-10) -> (Vector{Float64}, Matrix{Float64})

Port of delta_metrics.delta_subN_F: steady-state flux and subnetwork
membership via the topological solver with the epsilon tolerance.
"""
function delta_subN_F(A::Matrix{Float64}; epsilon::Float64 = 10.0^-10)
    result = solve_adjacency_steady_state(A; atol=epsilon)
    return result.node_flux, result.subnetwork_membership
end

"""
    nl_entropy_rate(A) -> Float64

Port of delta_metrics.nl_entropy_rate: normalize flux, per-node normalized
entropy of subnetwork membership, flux-weighted sum.
"""
function nl_entropy_rate(A::Matrix{Float64})
    F, SubN = delta_subN_F(A)

    total = _np_sum_contiguous(F)   # np.sum(F): pairwise contiguous reduction
    F = F ./ total

    # Entropy per node
    n = length(F)
    Entropy = Float64[]
    for i in 1:n
        I = [j for j in 1:size(SubN, 2) if SubN[i, j] > 0]   # np.where order
        # ent = -np.sum(SubN[i, I] * np.log2(SubN[i, I])): pairwise over the
        # selected products
        terms = Float64[-SubN[i, j] * log2(SubN[i, j]) for j in I]
        ent = _np_sum_contiguous(terms)
        if length(I) > 1
            ent = ent / log2(length(I))
        end
        push!(Entropy, ent)
    end

    # nER = np.sum(F * np.array(Entropy))
    nER = _np_sum_contiguous(F .* Entropy)
    return nER
end
