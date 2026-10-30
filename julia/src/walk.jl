# Skeleton walker internals (rivgraph/walk.py, bug-for-bug)
# =========================================================
# Port of the full walk.py module. All flat pixel indices in the data domain
# are 0-BASED (identical values to the Python port); Julia array accesses add
# 1 at the boundary. Sets are CPythonSet (exact CPython table semantics);
# node/link ids are Python-domain ints derived from OrderedSet positions.

# ---------------------------------------------------------------------------
# walk.get_neighbors (walk.py:377)
# ---------------------------------------------------------------------------

"Port of walk.get_neighbors: 8-neighbors of idx (0-based flat) as a list."
function get_neighbors(idx::Int64, Iskel::Matrix{Bool})
    I, row_offset, col_offset = get_array(Int(idx), Iskel, 3, 3)  # window copy
    I_flat = np_flatten(I)
    neighbor_idcs = neighbors_flat(4, I_flat, 3)   # 0-based center of a 3x3
    return reglobalize_flat_idx(neighbor_idcs, (3, 3), row_offset, col_offset,
                                (size(Iskel, 1), size(Iskel, 2)))
end

# ---------------------------------------------------------------------------
# walk.walkable_neighbors (walk.py:341)
# ---------------------------------------------------------------------------

"Port of walk.walkable_neighbors: set of walkable neighbors from link end."
function walkable_neighbors(link::Vector{Int64}, Iskel::Matrix{Bool})
    idx = link[end]
    neighs = py_set_from(get_neighbors(idx, Iskel))
    # Python try/except guards malformed links; the subtraction itself is exact
    neighs = py_difference(neighs, py_set_from(link[max(end - 1, 1):end]))
    return neighs
end

# ---------------------------------------------------------------------------
# walk.idcs_no_turnaround (walk.py:201)
# ---------------------------------------------------------------------------

"Port of walk.idcs_no_turnaround: candidate steps that don't turn around."
function idcs_no_turnaround(idcs::Vector{Int64}, Iskel::Matrix{Bool})
    ncols = size(Iskel, 2)
    idxdif = idcs[1] - idcs[2]  # Python idcs[0]-idcs[1], 0-based domain

    walkdirs = nothing
    if idxdif == -ncols - 1
        walkdirs = [1, ncols, ncols + 1]
    elseif idxdif == -ncols
        walkdirs = [ncols - 1, ncols, ncols + 1]
    elseif idxdif == -ncols + 1
        walkdirs = [-1, ncols - 1, ncols]
    elseif idxdif == -1
        walkdirs = [-ncols + 1, 1, ncols + 1]
    elseif idxdif == 1
        walkdirs = [-ncols - 1, -1, ncols - 1]
    elseif idxdif == ncols - 1
        walkdirs = [-ncols, -ncols + 1, 1]
    elseif idxdif == ncols
        walkdirs = [-ncols - 1, -ncols, -ncols + 1]
    elseif idxdif == ncols + 1
        walkdirs = [-1, -ncols - 1, -ncols]
    end
    walkdirs === nothing && throw(ErrorException("walkdirs not assigned (idxdif $idxdif)"))
    return [idcs[end] + wd for wd in walkdirs]
end

# ---------------------------------------------------------------------------
# walk.cant_walk (walk.py:252)
# ---------------------------------------------------------------------------

"Port of walk.cant_walk: pixels the current walk may not enter."
function cant_walk(links::Links, linkidx0::Int, nodes::Nodes, Iskel::Matrix{Bool})
    # 1. Originating node's bp cluster (bp_cluster mutates the seed list)
    bps = bp_cluster(Int64[links.idx[linkidx0+1][1]], Iskel)
    walked = py_set_from(bps)
    # 2. Emanating links
    for bp in bps
        walked = py_or(walked, py_set_from(walkable_neighbors(Int64[bp], Iskel)))
    end
    # 3. Links walking away from the node
    nodeidx = links.idx[linkidx0+1][1]
    connlinks = nodes.conn[oset_index0(nodes.idx, nodeidx)+1]
    for cl in connlinks
        templinkidx = oset_index0(links.id, cl) + 1
        lidx = links.idx[templinkidx]
        if lidx[1] == links.idx[linkidx0+1][1]
            walked = py_or(walked, py_set_from(lidx[1:max(end - 1, 0)]))       # [:-1]
        else
            walked = py_or(walked, py_set_from(lidx[2:end]))                   # [1:]
        end
    end
    return walked
end

# ---------------------------------------------------------------------------
# walk.find_emanators (walk.py:307)
# ---------------------------------------------------------------------------

"Port of walk.find_emanators: walkable neighbors minus the bp cluster itself."
function find_emanators(bpnode::Int64, Iskel::Matrix{Bool})
    branchpoints = bp_cluster(Int64[bpnode], Iskel)
    emanators = CPythonSet{Int64}()
    for bp in branchpoints
        emanators = py_or(emanators, py_set_from(walkable_neighbors(Int64[bp], Iskel)))
    end
    return py_difference(emanators, py_set_from(branchpoints))
end

# ---------------------------------------------------------------------------
# walk.bp_cluster (walk.py:166)
# ---------------------------------------------------------------------------

"Port of walk.bp_cluster: recursive 8-connected cluster of branchpoints."
function bp_cluster(bp::Vector{Int64}, Iskel::Matrix{Bool})
    bp_neighs = walkable_neighbors(bp, Iskel)
    while !isempty(bp_neighs)
        b = py_pop!(bp_neighs)
        if is_bp(b, Iskel) == 1
            if !in(b, bp)  # list membership
                push!(bp, b)
                bp_cluster(bp, Iskel)
            end
        end
    end
    return bp
end

# ---------------------------------------------------------------------------
# walk.delete_link (walk.py:414)
# ---------------------------------------------------------------------------

"Port of walk.delete_link: remove a link (walker state) and repair node conn."
function walker_delete_link!(linkid::Int64, links::Links, nodes::Nodes)
    lid = oset_index0(links.id, linkid) + 1
    deleteat!(links.idx, lid)
    nodeidx = links.conn[lid]
    deleteat!(links.conn, lid)
    oset_remove!(links.id, linkid)
    for ni in nodeidx
        lst = nodes.conn[ni+1]
        pos = findfirst(==(linkid), lst)
        pos === nothing && throw(ErrorException("link $linkid missing from node conn"))
        deleteat!(lst, pos)
    end
    return links, nodes
end

# ---------------------------------------------------------------------------
# walk.check_dup_links (walk.py:457)
# ---------------------------------------------------------------------------

"Port of walk.check_dup_links: delete links duplicating the one just resolved."
function check_dup_links(linkid::Int64, links::Links, nodes::Nodes, links2do::OrderedSet{Int64})
    linkidx = oset_index0(links.id, linkid) + 1
    link = links.idx[linkidx]
    lconn = links.conn[linkidx][2]           # Python links['conn'][linkidx][1]
    nconn = py_set_from(nodes.conn[lconn+1]) # set(nodes['conn'][lconn])
    py_remove!(nconn, linkid)                # nconn.remove(linkid)
    for lid in collect(nconn)                # SET iteration order snapshot
        candidate = links.idx[oset_index0(links.id, lid)+1]
        if py_seteq(py_set_from(link[max(end - 1, 1):end]), py_set_from(candidate[1:min(end, 2)]))
            walker_delete_link!(lid, links, nodes)
            # links2do.remove(lid) — KeyError caught (debug log only)
            oset_discard!(links2do, lid)
        end
    end
    return links, nodes, links2do
end

# ---------------------------------------------------------------------------
# walk.handle_bp (walk.py:17)
# ---------------------------------------------------------------------------

"Port of walk.handle_bp: resolve a branchpoint during the walk."
function handle_bp(linkid::Int64, bpnode::Int64, nodes::Nodes, links::Links,
                   links2do::OrderedSet{Int64}, Iskel::Matrix{Bool})
    oset_remove!(links2do, linkid)

    doneflag = in(bpnode, nodes.idx) ? 1 : 0
    linkidx = oset_index0(links.id, linkid) + 1

    node_updater!(nodes, bpnode, linkid)
    link_updater!(links, linkid, Int64(-1), Int64(oset_index0(nodes.idx, bpnode)))

    doneflag == 1 && return links, nodes, links2do

    # Resolve the branchpoint cluster
    bp = bp_cluster(Int64[bpnode], Iskel)

    # Emanators minus the cluster and the current link's pixels (chained diffs)
    em = py_difference(py_difference(find_emanators(bpnode, Iskel), py_set_from(bp)),
                       py_set_from(links.idx[linkidx]))
    emanators = py_elements(em)
    ncols = size(Iskel, 2)

    # 4-connected emanator pairs first
    fourconn = Vector{Vector{Int64}}()
    emremove = Int64[]
    for b in bp
        for e in emanators                       # array (table) order
            a = abs(b - e)
            if a == 1 || a == ncols
                push!(fourconn, [b, e])
                push!(emremove, e)
            end
        end
    end
    emanators = [e for e in emanators if !in(e, emremove)]  # list membership

    # Diagonal emanator pairs (computed on the reduced emanator list)
    diagconn = Vector{Vector{Int64}}()
    for b in bp
        for e in emanators
            a = abs(b - e)
            if a == ncols + 1 || a == ncols - 1
                push!(diagconn, [b, e])
            end
        end
    end

    # Links between adjacent branchpoints; set-of-2 dedupe reproduces the
    # Python set(tuple(...)) order (tuple built from the 2-element set order)
    bp_pairs = Vector{CPythonSet{Int64}}()
    for b in bp
        bn = walkable_neighbors(Int64[b], Iskel)
        for bb in bp
            in(bb, bn) && push!(bp_pairs, py_set_from([b, bb]))
        end
    end
    tuple_set = CPythonSet{Tuple{Int64,Int64}}()
    for ps in bp_pairs
        el = py_elements(ps)
        py_add!(tuple_set, (el[1], el[2]))
    end
    bp_pairs_final = [Int64[t[1], t[2]] for t in tuple_set]

    for b in bp_pairs_final
        newid = maximum(links.id.items) + 1
        node_updater!(nodes, b[1], newid)
        node_updater!(nodes, b[2], newid)
        link_updater!(links, newid, b, Int64(oset_index0(nodes.idx, b[1])))
        link_updater!(links, newid, Int64(-1), Int64(oset_index0(nodes.idx, b[2])))
    end

    # Initialize fourconn links first (so they walk first), then diagonals
    for p in vcat(fourconn, diagconn)
        nodeidx = oset_index0(nodes.idx, p[1])
        donelinks = nodes.conn[nodeidx+1]
        isdone = 0
        for dl in donelinks
            dlidx = oset_index0(links.id, dl) + 1
            lidx = links.idx[dlidx]
            if py_seteq(py_set_from(lidx[max(end - 1, 1):end]), py_set_from(p))
                isdone = 1
            end
        end
        if isdone == 0
            linkid = maximum(links.id.items) + 1
            oset_add!(links2do, linkid)
            node_updater!(nodes, p[1], linkid)
            link_updater!(links, linkid, p, Int64(oset_index0(nodes.idx, p[1])))
        end
    end

    return links, nodes, links2do
end

# ---------------------------------------------------------------------------
# walk.is_bp (walk.py:523)
# ---------------------------------------------------------------------------

"NumPy `I[:k, :] = 0` (negative k wraps); 1-based Julia matrices."
function _zero_rows_before!(I::Matrix{Bool}, k::Int)
    nr = size(I, 1)
    kk = k < 0 ? k + nr : k
    kk = clamp(kk, 0, nr)
    for r in 1:kk
        I[r, :] .= false
    end
end

"NumPy `I[k:, :] = 0` (start may be past the end)."
function _zero_rows_from!(I::Matrix{Bool}, k::Int)
    nr = size(I, 1)
    kk = clamp(k, 0, nr)
    for r in kk+1:nr
        I[r, :] .= false
    end
end

"NumPy `I[:, :k] = 0` (negative k wraps)."
function _zero_cols_before!(I::Matrix{Bool}, k::Int)
    nc = size(I, 2)
    kk = k < 0 ? k + nc : k
    kk = clamp(kk, 0, nc)
    for c in 1:kk
        I[:, c] .= false
    end
end

"NumPy `I[:, k:] = 0`."
function _zero_cols_from!(I::Matrix{Bool}, k::Int)
    nc = size(I, 2)
    kk = clamp(k, 0, nc)
    for c in kk+1:nc
        I[:, c] .= false
    end
end

"Port of walk.is_bp: 1 when idx (0-based flat) is a branchpoint, else 0."
function is_bp(idx::Int64, Iskel::Matrix{Bool})
    neighs = get_neighbors(idx, Iskel)
    length(neighs) < 3 && return 0

    big_enough = 0
    size0, size1 = 7, 7
    local I::Matrix{Bool}
    local Ic::Matrix{UInt8}
    local Ilab::Matrix{Int}
    local centlabel::Int
    local cpy::Vector{Int}, cpx::Vector{Int}
    local roffset::Int, coffset::Int
    while big_enough == 0
        centidx0 = (size0 - 1) ÷ 2
        centidx1 = (size1 - 1) ÷ 2
        I, roffset, coffset = get_array(Int(idx), Iskel, size0, size1)
        Ic = im_connectivity(I)
        Ict = fill(false, size0, size1)          # np.zeros_like on a bool array
        for j in eachindex(Ic)
            Ict[j] = Ic[j] > 2
        end
        Ilab, _ = label_components(Ict; connectivity=1)

        centlabel = Ilab[centidx0+1, centidx1+1]
        cpy, cpx = Int[], Int[]
        for r in 1:size0, c in 1:size1
            if Ilab[r, c] == centlabel
                push!(cpy, r - 1)   # 0-based row
                push!(cpx, c - 1)   # 0-based col
            end
        end
        big_enough = 1
        if in(1, cpy) || in(size0 - 2, cpy)
            size0 += 4
            big_enough = 0
        end
        if in(1, cpx) || in(size1 - 2, cpx)
            size1 += 4
            big_enough = 0
        end
    end

    # Reduce the window to the connected >2 region plus a 1-pixel buffer
    # (NumPy negative-slice semantics)
    _zero_rows_before!(I, minimum(cpy) - 1)
    _zero_rows_from!(I, maximum(cpy) + 2)
    _zero_cols_before!(I, minimum(cpx) - 1)
    _zero_cols_from!(I, maximum(cpx) + 2)

    # Keep only the largest blob in case there are border stragglers
    I = largest_blobs_keep(I, 1)

    # Zero out everything outside the region of interest (mutates Ic in place)
    for r in 1:size(I, 1), c in 1:size(I, 2)
        if Ilab[r, c] != centlabel && Ic[r, c] > 2
            Ic[r, c] = 1
        end
        if !I[r, c]                     # I != 1 on a boolean array
            Ic[r, c] = 0
        end
    end

    # Trivial case: idx is the only possible branchpoint
    count(Ic .> 2) == 1 && return 1

    Ina = naxes_connectivity(I)
    Inf = nfour_connectivity(I)
    Icr = np_flatten(Ic)
    Inar = np_flatten(Ina)
    Infr = np_flatten(Inf)

    bps = isbp_parsimonious(Ic, Icr, Inar, Infr)

    # Return branchpoints to global, flat coordinates
    bps = reglobalize_flat_idx(bps, (size(Ic, 1), size(Ic, 2)), roffset, coffset,
                               (size(Iskel, 1), size(Iskel, 2)))
    return in(idx, bps) ? 1 : 0
end

# ---------------------------------------------------------------------------
# walk.isbp_parsimonious (walk.py:615)
# ---------------------------------------------------------------------------

"scipy.stats.mode (ties -> smallest value) over a nonempty vector of ints."
function _scipy_mode_smallest(vals::Vector{Int64})
    counts = Dict{Int64,Int}()
    for v in vals
        counts[v] = get(counts, v, 0) + 1
    end
    best = Int64(0)
    bestcount = -1
    for v in sort!(collect(keys(counts)))
        c = counts[v]
        if c > bestcount
            bestcount = c
            best = v
        end
    end
    return best
end

"Port of walk.isbp_parsimonious: minimal branchpoint set for a window."
function isbp_parsimonious(Ic::Matrix{UInt8}, Icr::Vector{UInt8}, Inar::Vector{UInt8},
                           Infr::Vector{UInt8})
    # Possible bps: conn>2 pixels (np.where raster order), minus edge pixels
    nr, nc = size(Ic)
    bp_poss = Int64[]
    for r in 1:nr, c in 1:nc
        Ic[r, c] > 2 && push!(bp_poss, Int64((r - 1) * nc + (c - 1)))
    end
    poss_diff = py_difference(py_set_from(bp_poss), edge_coords((nr, nc)))
    bp_poss_i = collect(poss_diff)   # list(set) — table order

    # Walk from each possible initial branchpoint
    bpsave = Vector{CPythonSet{Int64}}()
    for bpi in bp_poss_i
        push!(bpsave, isbp_walk_for_bps(Ic .!= 0, [bpi]))  # np.array(Ic, dtype=bool)
    end

    bpcounts = Int[length(b) for b in bpsave]
    bpsolo = Int64[]
    for i in eachindex(bpcounts)
        bpcounts[i] == 1 && push!(bpsolo, py_pop!(bpsave[i]))
    end

    if !isempty(bpsolo)
        naxconn = UInt8[Inar[b + 1] for b in bpsolo]
        maxnaxv = maximum(naxconn)
        maxnax = Int64[bpsolo[i] for i in eachindex(bpsolo) if naxconn[i] == maxnaxv]
        if length(maxnax) == 1
            return maxnax
        end
        fourconn = UInt8[Infr[b + 1] for b in bpsolo]
        maxfourv = maximum(fourconn)
        maxfour = Int64[bpsolo[i] for i in eachindex(bpsolo) if fourconn[i] == maxfourv]
        if length(maxfour) == 1
            return maxfour
        end
        # Highest 4-conn within the highest naxes-conn
        fourconn2 = UInt8[Infr[b + 1] for b in maxnax]
        maxfourv2 = maximum(fourconn2)
        maxfour2 = Int64[maxnax[i] for i in eachindex(maxnax) if fourconn2[i] == maxfourv2]
        return [minimum(maxfour2)]
    end

    # bp_must according to conn, naxes, nfour patterns
    keepvals = [[6, 4, 2], [5, 3, 1], [5, 3, 4], [3, 3, 2], [3, 3, 1], [4, 2, 4]]
    bp_must = Int64[]
    for kv in keepvals
        for fi in 1:length(Icr)
            if Int(Icr[fi]) == kv[1] && Int(Inar[fi]) == kv[2] && Int(Infr[fi]) == kv[3]
                push!(bp_must, Int64(fi - 1))  # np.where flat index, 0-based
            end
        end
    end

    # Special case [4, 4, 2] — exactly two matches: keep only the first
    keeps442 = Int64[]
    for fi in 1:length(Icr)
        if Int(Icr[fi]) == 4 && Int(Inar[fi]) == 4 && Int(Infr[fi]) == 2
            push!(keeps442, Int64(fi - 1))
        end
    end
    if length(keeps442) == 2
        push!(bp_must, keeps442[1])
    end

    if !isempty(bp_must)
        return isbp_walk_for_bps(Ic .!= 0, bp_must)  # np.array(Ic, dtype=bool)
    end

    # Smallest walk set wins; ties move on
    mincount = minimum(bpcounts)
    minidcs = [i for i in eachindex(bpcounts) if bpcounts[i] == mincount]
    if length(minidcs) == 1
        return bpsave[minidcs[1]]
    end

    # Most common branchpoint across all walks (set-iteration order; mode
    # ties resolve to the smallest value — scipy semantics, verified this
    # session)
    allvals = Int64[]
    for b in bpsave
        for p in collect(b)
            push!(allvals, p)
        end
    end
    modeval = _scipy_mode_smallest(allvals)
    bp_init = Int64[modeval]
    return isbp_walk_for_bps(Ic .!= 0, bp_init)  # np.array(Ic, dtype=bool)
end

# ---------------------------------------------------------------------------
# walk.isbp_walk_for_bps (walk.py:710)
# ---------------------------------------------------------------------------

"Port of walk.isbp_walk_for_bps: walk-based branchpoint selection."
function isbp_walk_for_bps(I::AbstractMatrix{Bool}, bpi_list::Vector{Int64})
    bpi = py_set_from(bpi_list)
    Ir = np_flatten(I)
    edgeidcs = edge_coords((size(I, 1), size(I, 2)))

    do_first = CPythonSet{Int64}()
    emanators = CPythonSet{Int64}()
    for bp in bpi                        # set iteration order
        emanators = py_or(emanators, py_set_from(neighbors_flat(Int(bp), Ir, size(I, 2))))
        fc = four_conn([bp], I)[1]
        for f in fc
            py_add!(do_first, f)         # update with list — add each in order
        end
    end

    walked = py_or(bpi, emanators)

    while !isempty(emanators)
        if !isempty(do_first)
            idx = py_pop!(do_first)
            py_remove!(emanators, idx)   # must exist (same invariant as Python)
        else
            idx = py_pop!(emanators)
        end

        walking = true
        while walking
            py_add!(walked, idx)
            neighs = py_difference(py_set_from(neighbors_flat(Int(idx), Ir, size(I, 2))), walked)

            if isempty(neighs)
                walking = false
            elseif length(neighs) == 1
                idx = py_pop!(neighs)
                in(idx, edgeidcs) && (walking = false)
            else
                py_add!(bpi, idx)
                fc = four_conn([idx], I)[1]
                fourconn = Int64[f for f in fc if in(f, neighs)]
                for f in fourconn
                    py_add!(do_first, f)
                end
                emanators = py_or(emanators, neighs)
                py_add!(walked, idx)
                _py_update!(walked, neighs)   # walked.update(neighs)
            end
        end
    end
    return bpi
end

"Python set.update(other): add each element of other in its table order."
function _py_update!(s::CPythonSet{Int64}, other::CPythonSet{Int64})
    for v in other
        py_add!(s, v)
    end
end

# ---------------------------------------------------------------------------
# walk.naxes_connectivity (walk.py:784)
# ---------------------------------------------------------------------------

"Port of walk.naxes_connectivity: count of connectivity axes per pixel."
function naxes_connectivity(I::Matrix{Bool})
    Ir = np_flatten(I)
    edgeidcs = edge_coords((size(I, 1), size(I, 2)))
    ncols = size(I, 2)

    allpix = py_set_from(Int64[fi - 1 for fi in 1:length(Ir) if Ir[fi]])  # 0-based
    dopix = py_difference(allpix, edgeidcs)

    # Python pairs each popped pixel with its own count (Inax[savepix] = naxes
    # aligns savepix with pop order). Counts depend only on Ir, so bind each
    # count to its pixel directly rather than relying on order agreement.
    nr, nc = size(I)
    Inax = zeros(UInt8, nr, nc)
    while !isempty(dopix)
        pix = py_pop!(dopix)
        count = 0
        if Ir[pix+2] == true || Ir[pix] == true                # Ir[pix+1]/Ir[pix-1]
            count += 1
        end
        if Ir[pix+ncols+1] == true || Ir[pix-ncols+1] == true  # Ir[pix±ncols]
            count += 1
        end
        if Ir[pix+ncols+2] == true || Ir[pix-ncols] == true    # Ir[pix+1+nc]/Ir[pix-1-nc]
            count += 1
        end
        if Ir[pix+ncols] == true || Ir[pix-ncols+2] == true    # Ir[pix-1+nc]/Ir[pix+1-nc]
            count += 1
        end
        r0, c0 = divrem(pix, ncols)  # 0-based row, col (row-major flat)
        Inax[r0+1, c0+1] = count
    end
    return Inax
end
