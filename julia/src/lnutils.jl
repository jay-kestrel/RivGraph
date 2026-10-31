# Link/node network primitives (rivgraph/ln_utils.py subset)
# ==========================================================
# The subset the skeleton→graph walker and its cleanup require, ported
# bug-for-bug from the Python reference. Flat pixel indices and node/link
# ids stay in the Python 0-based domain; only container access is Julia-side.

"""
    Links

Parallel-array link store. `idx[k]` is the pixel-index list of the k-th link
(1-based k here; Python's `links['id'].index` position is k-1). `id` is an
OrderedSet of link ids. The width/length attributes are `nothing` until
`link_widths_and_lengths!` runs.
"""
Base.@kwdef mutable struct Links
    idx::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    conn::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    id::OrderedSet{Int64} = OrderedSet{Int64}()
    n_networks::Union{Nothing,Int} = nothing
    ids_finalized::Union{Nothing,Bool} = nothing
    ids_finalization_method::Union{Nothing,String} = nothing
    len::Union{Nothing,Vector{Float64}} = nothing
    wid_pix::Union{Nothing,Vector{Vector{Float64}}} = nothing
    wid::Union{Nothing,Vector{Float64}} = nothing
    wid_adj::Union{Nothing,Vector{Float64}} = nothing
    wid_med::Union{Nothing,Vector{Float64}} = nothing
    len_adj::Union{Nothing,Vector{Float64}} = nothing
    sinuosity::Union{Nothing,Vector{Float64}} = nothing
end

"""
    Nodes

Parallel-array node store; `idx` is an OrderedSet (order-sensitive parity),
`conn[k]` the link ids connected to the k-th node.
"""
Base.@kwdef mutable struct Nodes
    idx::OrderedSet{Int64} = OrderedSet{Int64}()
    conn::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    id::Union{Nothing,OrderedSet{Int64}} = nothing
    ids_finalized::Union{Nothing,Bool} = nothing
    ids_finalization_method::Union{Nothing,String} = nothing
end

"Python `list.index` on an OrderedSet: 0-based position (KeyError when absent)."
function oset_index0(s::OrderedSet{K}, key::K) where {K}
    haskey(s.map, key) || throw(KeyError(key))
    return s.map[key] - 1
end

"Port of ln_utils.mark_network_ids_provisional(reason='topology not finalized')."
function mark_network_ids_provisional!(links::Links, nodes::Nodes,
                                       reason::String = "topology not finalized")
    links.ids_finalized = false
    links.ids_finalization_method = reason
    nodes.ids_finalized = false
    nodes.ids_finalization_method = reason
    return links, nodes
end

"Port of ln_utils.node_updater(nodes, idx, conn)."
function node_updater!(nodes::Nodes, idx::Int64, conn::Int64)
    if !in(idx, nodes.idx)
        oset_add!(nodes.idx, idx)
    end
    if length(nodes.conn) < length(nodes.idx)
        push!(nodes.conn, Int64[])
    end
    nodeid = oset_index0(nodes.idx, idx)
    nodes.conn[nodeid+1] = vcat(nodes.conn[nodeid+1], [conn])
    return nodes
end

"Port of ln_utils.link_updater(links, linkid, idx=-1, conn=-1): scalar idx wrapped to a list; -1 skips the field (Python sentinel semantics)."
function link_updater!(links::Links, linkid::Int64, idx::Union{Int64,Vector{Int64}} = Int64(-1),
                       conn::Int64 = Int64(-1))
    in(linkid, links.id) || oset_add!(links.id, linkid)
    linkidx = oset_index0(links.id, linkid) + 1  # 1-based vector position
    if idx != -1
        idxv = isa(idx, Vector{Int64}) ? idx : [idx]
        if length(links.idx) < length(links.id.items)
            push!(links.idx, Int64[])
            push!(links.conn, Int64[])
        end
        links.idx[linkidx] = vcat(links.idx[linkidx], idxv)
    end
    if conn != -1
        if length(links.conn) < length(links.id.items)
            push!(links.conn, Int64[])
            push!(links.idx, Int64[])
        end
        links.conn[linkidx] = vcat(links.conn[linkidx], [conn])
    end
    return links
end

"""
    adjust_for_padding!(links, nodes, npad, dims, initial_dims)

Port of ln_utils.adjust_for_padding: shift every pixel index back by the
walker's internal pad (per side) and rebuild the node OrderedSet in the
original node order. The Python code performs no bounds filtering —
np.ravel_multi_index would raise on pad-region pixels — so neither does the
port.
"""
function adjust_for_padding!(links::Links, nodes::Nodes, npad::Int, dims::Tuple{Int,Int},
                             initial_dims::Tuple{Int,Int})
    ncols = dims[2]
    newncols = initial_dims[2]
    for li in eachindex(links.idx)
        adjusted = Int64[]
        for idx in links.idx[li]
            r, c = idx ÷ ncols, idx % ncols                       # 0-based unravel
            push!(adjusted, (r - npad) * newncols + (c - npad))   # 0-based flat
        end
        links.idx[li] = adjusted
    end
    newnodeidx = Int64[]
    for idx in nodes.idx.items
        r, c = idx ÷ ncols, idx % ncols                           # 0-based unravel
        push!(newnodeidx, (r - npad) * newncols + (c - npad))     # 0-based flat
    end
    nodes.idx = OrderedSet{Int64}()
    for v in newnodeidx
        oset_add!(nodes.idx, v)
    end
    return links, nodes
end

"""
    remove_duplicate_links!(links, nodes)

Port of ln_utils.remove_duplicate_links. Duplicate links share the same pixel
index SET (Python `set(idx)` equality — list order may differ). Candidate
pairs come from every node's conn list; they are combined into an undirected
graph exactly as networkx does (edges from `combinations(set(pair), 2)`, node
insertion order preserved), and per connected component everything but the
first member — in CPython set iteration order — is deleted.
"""
function remove_duplicate_links!(links::Links, nodes::Nodes)
    pairs = Vector{Vector{Int64}}()
    for nconn in nodes.conn
        for l1 in nconn
            for l2 in nconn
                l1 == l2 && continue
                if py_set_from(links.idx[oset_index0(links.id, l1)+1]) ==
                   py_set_from(links.idx[oset_index0(links.id, l2)+1])
                    push!(pairs, [l1, l2])
                end
            end
        end
    end
    isempty(pairs) && return links, nodes

    # nx.Graph(edges): undirected, adjacency lists in insertion order
    node_order = Int64[]          # first-insertion order of nodes
    node_seen = CPythonSet{Int64}()
    adj = Dict{Int64,Vector{Int64}}()
    function add_edge!(x::Int64, y::Int64)
        if !in(x, node_seen)
            py_add!(node_seen, x)
            push!(node_order, x)
            adj[x] = Int64[]
        end
        if !in(y, node_seen)
            py_add!(node_seen, y)
            push!(node_order, y)
            adj[y] = Int64[]
        end
        if !in(y, adj[x])
            push!(adj[x], y)
        end
        if !in(x, adj[y])
            push!(adj[y], x)
        end
    end
    for n in pairs
        # combinations(set(n), 2): the 2-element set's iteration order fixes
        # the edge tuple order, hence node insertion order
        s = CPythonSet{Int64}()
        py_add!(s, n[1])
        py_add!(s, n[2])
        elems = py_elements(s)
        add_edge!(elems[1], elems[2])
    end

    # nx.connected_components: iterate nodes in insertion order; unseen nodes
    # start a BFS whose members accumulate in visit order; list(set) yields
    # CPython table order of the accumulated component set.
    seen = CPythonSet{Int64}()
    for v in node_order
        in(v, seen) && continue
        comp_adds = Int64[]
        comp_seen = CPythonSet{Int64}()
        nextlevel = [v]
        while !isempty(nextlevel)
            thislevel = nextlevel
            nextlevel = Int64[]
            for u in thislevel
                in(u, comp_seen) && continue
                py_add!(comp_seen, u)
                push!(comp_adds, u)
                for w in adj[u]
                    push!(nextlevel, w)
                end
            end
        end
        for w in comp_adds
            py_add!(seen, w)
        end
        # list(component set) in CPython table order; drop the first, delete rest
        sames = py_elements(comp_seen)
        for lid in sames[2:end]
            delete_link_full!(links, nodes, lid)
        end
    end
    return links, nodes
end

"""
    delete_link_full!(links, nodes, linkid)

Port of ln_utils.delete_link: removes idx/conn entries, the OrderedSet id, and
repairs node connectivity, deleting any node left with no links.
"""
function delete_link_full!(links::Links, nodes::Nodes, linkid::Int64)
    lidx = oset_index0(links.id, linkid) + 1
    connected_node_ids = copy(links.conn[lidx])
    deleteat!(links.idx, lidx)
    deleteat!(links.conn, lidx)
    oset_remove!(links.id, linkid)
    for cni in connected_node_ids
        cnodeidx = oset_index0(nodes.id::OrderedSet{Int64}, cni) + 1
        lst = nodes.conn[cnodeidx]
        pos = findfirst(==(linkid), lst)
        # Python list.remove raises ValueError when absent — preserve that
        pos === nothing && throw(ErrorException("link $linkid missing from node $cni conn"))
        deleteat!(lst, pos)
        if isempty(nodes.conn[cnodeidx])
            delete_node_full!(nodes, cni)
        end
    end
    mark_network_ids_provisional!(links, nodes)
    return links, nodes
end

"""
    delete_node_full!(nodes, nodeid)

Port of ln_utils.delete_node: removes the node id (OrderedSet), its idx
(OrderedSet), and its conn list position.
"""
function delete_node_full!(nodes::Nodes, nodeid::Int64)
    nodeidx = oset_index0(nodes.id::OrderedSet{Int64}, nodeid) + 1
    oset_remove!(nodes.id::OrderedSet{Int64}, nodeid)
    oset_remove!(nodes.idx, nodes.idx.items[nodeidx])
    deleteat!(nodes.conn, nodeidx)
    return nodes
end
