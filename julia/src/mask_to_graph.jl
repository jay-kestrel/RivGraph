# Skeleton → graph driver (rivgraph/mask_to_graph.py, bug-for-bug)
# ================================================================
# Port of the skel_to_graph wrapper (isolated-pixel pre-classification and
# dispatch) and the _skel_to_graph_python reference driver. The Julia port
# targets the Python reference path (PYDELTARCM_NATIVE=0), which is the
# parity oracle for the existing Rust backend.

# ---------------------------------------------------------------------------
# Wrapper pre-classification (mask_to_graph.py:74-103)
# ---------------------------------------------------------------------------

"""
    classify_isolated_pixels(Iskel) -> (Matrix{Bool}, Int, Bool)

Port of the wrapper's isolated-pixel block: 8-connected components of the
nonzero skeleton; singleton components (size 1) are flagged. Returns
(mask, nsingletons, ncomp). The Python code raises when ALL components are
singletons, drops them with a warning otherwise; `raise_only_isolated`
performs the raise when nsingletons == ncomp > 0.
"""
function classify_isolated_pixels(Iskel::AbstractMatrix{Bool})
    skel_bool = Iskel .!= 0
    ncomp = 0
    nsingletons = 0
    singleton_mask = fill(false, size(Iskel))
    if size(Iskel, 1) >= 3 && size(Iskel, 2) >= 3 && !isempty(skel_bool)
        lab, ncomp = label_components(skel_bool; connectivity=2)  # 3x3 ones structure
        if ncomp > 0
            counts = zeros(Int, ncomp)
            for v in lab
                v != 0 && (counts[v] += 1)
            end
            singleton_labels = [l for l in 1:ncomp if counts[l] == 1]
            nsingletons = length(singleton_labels)
            if nsingletons > 0
                for r in 1:size(lab, 1), c in 1:size(lab, 2)
                    lab[r, c] in singleton_labels && (singleton_mask[r, c] = true)
                end
                skel_bool = skel_bool .& .!singleton_mask
            end
        end
    end
    return skel_bool, nsingletons, ncomp
end

# ---------------------------------------------------------------------------
# Nested helpers of _skel_to_graph_python (mask_to_graph.py:120-186)
# ---------------------------------------------------------------------------

"check_startpoint: True when the start pixel's (single) first neighbor is not a bp."
function check_startpoint(spidx::Int64, Iskel::Matrix{Bool})
    neighs = walkable_neighbors(Int64[spidx], Iskel)
    isempty(neighs) && throw(KeyError(Int64(-1)))  # Python set.pop() on empty
    isbp = is_bp(py_pop!(neighs), Iskel)
    return isbp == 0
end

"find_starting_pixels: one valid endpoint per connected component."
function find_starting_pixels(Iskel::Matrix{Bool})
    eps = py_set_from(skel_endpoints(Iskel))   # Python ravels np.where output; same set
    startpoints = Int64[]
    for comp in regionprops_coords(Iskel; connectivity=2)   # label order (raster first occurrence)
        idcs = py_set_from(comp)
        poss_id = py_and(idcs, eps)          # Python set.intersection -> a & b
        if !isempty(poss_id)
            for pid in collect(poss_id)      # SET iteration order snapshot
                if check_startpoint(pid, Iskel)
                    push!(startpoints, pid)
                    break
                end
            end
        end
    end
    return startpoints
end

"np.pad(Iskel, 20, mode='constant') on a Bool matrix."
function _pad_skel(Iskel::Matrix{Bool}, npad::Int)
    nr, nc = size(Iskel)
    out = fill(false, nr + 2npad, nc + 2npad)
    out[npad+1:npad+nr, npad+1:npad+nc] .= Iskel
    return out
end

# ---------------------------------------------------------------------------
# _skel_to_graph_python (mask_to_graph.py:120-382)
# ---------------------------------------------------------------------------

"The Python reference walker; the parity oracle for this port."
function _skel_to_graph_python(Iskel0::Matrix{Bool})
    initial_dims = size(Iskel0)
    npad = 20
    Iskel = _pad_skel(Iskel0, npad)
    dims = size(Iskel)

    startpoints = find_starting_pixels(Iskel)

    nodes = Nodes()
    # links['idx'] = [[]]; links['conn'] = [[]] — a single dummy seed list that
    # the first link_updater call consumes (bug-for-bug).
    links = Links(idx=[Int64[]], conn=[Int64[]])

    if isempty(startpoints)
        for comp in regionprops_coords(Iskel; connectivity=2)
            seed = nothing
            for pid in comp                     # numpy array order (raster)
                if check_startpoint(pid, Iskel)
                    seed = pid
                    break
                end
            end
            if seed === nothing && !isempty(comp)
                seed = comp[1]
            end
            seed !== nothing && push!(startpoints, seed)
        end
    end

    isempty(startpoints) &&
        throw(ErrorException("Skeleton contains no walkable pixels; cannot resolve graph."))

    for (i, sp) in enumerate(startpoints)
        # Python's enumerate gives 0-based i; Julia's is 1-based. Node/link ids
        # and conn values are the Python-domain (0-based) i; the link slot is
        # links.idx[i] (1-based slot of Python's links['idx'][i]).
        links = link_updater!(links, Int64(length(links.id.items)), Int64(sp), Int64(i - 1))
        nodes = node_updater!(nodes, Int64(sp), Int64(i - 1))
        first_step = walkable_neighbors(links.idx[i], Iskel)
        if isempty(first_step)
            # Degenerate one-pixel component: register the endpoint again
            # (Python appends conn i twice to the node).
            nodes = node_updater!(nodes, Int64(sp), Int64(i - 1))
        else
            links = link_updater!(links, Int64(i - 1), Int64(py_pop!(first_step)))
        end
    end

    links.n_networks = length(startpoints)

    links2do = oset_copy(links.id)

    while !isempty(links2do)
        linkid = first(links2do)
        linkidx = oset_index0(links.id, linkid)

        walking = true
        cantwalk = cant_walk(links, linkidx, nodes, Iskel)

        while walking
            poss_steps_set = walkable_neighbors(links.idx[linkidx+1], Iskel)

            if isempty(poss_steps_set) # endpoint reached
                node_updater!(nodes, links.idx[linkidx+1][end], linkid)
                link_updater!(links, linkid, Int64(-1),
                              Int64(oset_index0(nodes.idx, links.idx[linkidx+1][end])))
                oset_remove!(links2do, linkid)
                break
            end

            if length(links.idx[linkidx+1]) < 4
                poss_steps = collect(py_difference(poss_steps_set, cantwalk))  # table order
            else
                poss_steps = collect(poss_steps_set)                            # table order
            end

            if isempty(poss_steps) # reached an emanating link; delete current
                walker_delete_link!(linkid, links, nodes)
                oset_remove!(links2do, linkid)
                walking = false
            elseif length(poss_steps) == 1
                link_updater!(links, linkid, poss_steps)
                if is_bp(poss_steps[1], Iskel) == 1
                    links, nodes, links2do =
                        handle_bp(linkid, Int64(poss_steps[1]), nodes, links, links2do, Iskel)
                    links, nodes, links2do =
                        check_dup_links(linkid, links, nodes, links2do)
                    walking = false
                end
            else
                isbp = Int[is_bp(Int64(n), Iskel) for n in poss_steps]

                if sum(isbp) == 0
                    # 4-connected neighbors of the last pixel
                    isfourconn = Int[]
                    for ps in poss_steps
                        checkfour = links.idx[linkidx+1][end] - ps
                        if checkfour in [-1, 1, -dims[2], dims[2]]
                            push!(isfourconn, 1)
                        else
                            push!(isfourconn, 0)
                        end
                    end

                    noturn = idcs_no_turnaround(links.idx[linkidx+1][max(end-1, 1):end], Iskel)
                    noturnidx = Int64[n for n in noturn if n in poss_steps]  # list membership

                    if sum(isfourconn) == 1
                        link_updater!(links, linkid, Int64(poss_steps[findfirst(==(1), isfourconn)]))
                    elseif length(noturnidx) == 1
                        link_updater!(links, linkid, noturnidx)
                    else
                        throw(ErrorException(
                            "Ambiguous which step to take next :( Please raise issue at " *
                            "https://github.com/jonschwenk/RivGraph/issues."))
                    end

                elseif sum(isbp) == 1
                    link_updater!(links, linkid, Int64(poss_steps[findfirst(==(1), isbp)]))
                    links, nodes, links2do =
                        handle_bp(linkid, Int64(poss_steps[findfirst(==(1), isbp)]),
                                  nodes, links, links2do, Iskel)
                    links, nodes, links2do =
                        check_dup_links(linkid, links, nodes, links2do)
                    walking = false

                elseif sum(isbp) > 1
                    isfourconn = Int[]
                    for ps in poss_steps
                        checkfour = links.idx[linkidx+1][end] - ps
                        if checkfour in [-1, 1, -dims[2], dims[2]]
                            push!(isfourconn, 1)
                        else
                            push!(isfourconn, 0)
                        end
                    end

                    # poss_step(s) both 4-connected and a branchpoint
                    isbp_and_fourconn_idx = [i for i in 1:length(isbp)
                                             if isbp[i] == 1 && isfourconn[i] == 1]

                    if length(isbp_and_fourconn_idx) != 1
                        throw(ErrorException("There is not a unique branchpoint to step to."))
                    else
                        link_updater!(links, linkid, Int64(poss_steps[isbp_and_fourconn_idx[1]]))
                        links, nodes, links2do =
                            handle_bp(linkid, Int64(poss_steps[isbp_and_fourconn_idx[1]]),
                                      nodes, links, links2do, Iskel)
                        links, nodes, links2do =
                            check_dup_links(linkid, links, nodes, links2do)
                        walking = false
                    end
                end
            end
        end
    end

    # Unpad; assign node ids; dedupe
    links, nodes = adjust_for_padding!(links, nodes, npad, dims, initial_dims)
    nodes.id = OrderedSet{Int64}()
    for v in Int64(0):(length(nodes.idx.items) - 1)
        oset_add!(nodes.id, v)
    end
    links, nodes = remove_duplicate_links!(links, nodes)

    return links, nodes
end

# ---------------------------------------------------------------------------
# skel_to_graph wrapper (mask_to_graph.py:19-103)
# ---------------------------------------------------------------------------

"""
    skel_to_graph(Iskel) -> (Links, Nodes)

Port of mask_to_graph.skel_to_graph (Python reference path). Consumes a
Python-produced skeleton (bool matrix); returns links/nodes with pixel
indices in the ORIGINAL (unpadded) frame. Isolated pixels are dropped with a
warning when a real network remains, and raise when they are the only
content — mirroring the Python wrapper.
"""
function skel_to_graph(Iskel::Matrix{Bool})
    mask, nsingletons, ncomp = classify_isolated_pixels(Iskel)
    if nsingletons > 0
        if nsingletons == ncomp
            throw(ErrorException(
                "Skeleton contains only isolated pixels (no component with more " *
                "than one pixel); no network can be walked."))
        end
        @warn "Removing $nsingletons isolated skeleton pixels (no neighbors); " *
              "they cannot form links."
        Iskel = Matrix{Bool}(mask)
    end
    return _skel_to_graph_python(Iskel)
end
