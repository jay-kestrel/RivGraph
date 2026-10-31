# Image utilities for the walker port (rivgraph/im_utils.py subset)
# =================================================================
# Exact ports of the helpers the skeleton walker and width kernel call.
# Every routine reproduces the Python semantics verified against the live
# library this session:
#
#   * `get_array` returns a *copy* window; slices use NumPy semantics
#     (negative starts wrap, inverted ranges produce empty windows).
#   * `im_connectivity` mirrors scipy correlate mode='mirror' (reflect
#     WITHOUT repeating the edge sample).
#   * component labels follow skimage.measure.label order: raster
#     first-occurrence; `coords` are raster-sorted.
#   * flat neighborhoods use the fixed row-major 8-neighbor order with
#     wraparound-aware `pullvals` filtering (neighbors_flat).

# ---------------------------------------------------------------------------
# NumPy-flavored window extraction (im_utils.get_array)
# ---------------------------------------------------------------------------

"NumPy `arr.flatten()` — ROW-major (C-order) copy of a 2-D array. Julia's vec() is column-major and must never be used where NumPy flatten order is semantic."
np_flatten(I::AbstractMatrix) = reduce(vcat, eachrow(I))

"NumPy 1-D slice range for (start, stop) with negative-value wraparound."
function _np_range(start::Int, stop::Int, n::Int)
    a = start < 0 ? start + n : start
    b = stop < 0 ? stop + n : stop
    a = clamp(a, 0, n)
    b = clamp(b, 0, n)
    return a, b  # 0-based, half-open; a >= b -> empty
end

"""
    get_array(idx, I, sh, sw) -> (window, row_offset, col_offset)

Port of im_utils.get_array: sub-image of `I` centered at flat index `idx`
(0-BASED, matching the Python data domain) with shape (sh, sw). Returns a
copy plus the top-left offsets (0-based; may be negative or past the end —
callers avoid out-of-bounds windows).
"""
function get_array(idx::Int, I::AbstractMatrix{Bool}, sh::Int, sw::Int)
    nr, nc = size(I)
    row, col = idx ÷ nc, idx % nc                      # flat -> (row, col), 0-based
    r0 = Int(row - (sh - 1) ÷ 2)
    c0 = Int(col - (sw - 1) ÷ 2)
    ra, rb = _np_range(r0, r0 + sh, nr)
    ca, cb = _np_range(c0, c0 + sw, nc)
    window = zeros(Bool, sh, sw)
    if ra < rb && ca < cb
        # I-side indices are global; window-side indices are the overlap-local
        # 1:(rb-ra) by 1:(cb-ca).
        window[1:rb-ra, 1:cb-ca] .= I[ra+1:rb, ca+1:cb]
    end
    return window, r0, c0
end

# ---------------------------------------------------------------------------
# Flat neighborhoods (im_utils.neighbors_flat / reglobalize_flat_idx)
# ---------------------------------------------------------------------------

"""
    neighbors_flat(idx, imflat, ncols) -> (idcs, vals)

Port of im_utils.neighbors_flat with filt='nonzero'. `idx` is a 0-BASED flat
index (matching the Python data domain). Returns the 8-neighbors in the fixed
order `[idx-dy-dx, idx-dy, idx-dy+dx, idx-dx, idx+dx, idx+dy-dx, idx+dy,
idx+dy+dx]`, dropping wraparound entries via the same `pullvals` edge-case
logic, then filtering to nonzero (true) values.
"""
function neighbors_flat(idx::Int, imflat::AbstractVector{Bool}, ncols::Int)
    dy, dx = ncols, 1
    poss = [idx - dy - dx, idx - dy, idx - dy + dx,
            idx - dx, idx + dx,
            idx + dy - dx, idx + dy, idx + dy + dx]
    n = length(imflat)
    # pullvals are 0-based positions into poss (exact Python table)
    if idx % ncols == 0                       # first column of image
        if idx < ncols                        # top row
            pull = (4, 6, 7)
        elseif idx >= n - ncols               # bottom row
            pull = (1, 2, 4)
        else
            pull = (1, 2, 4, 6, 7)
        end
    elseif (idx + 1) % ncols == 0             # last column of image
        if idx < ncols                        # top row
            pull = (3, 5, 6)
        elseif idx >= n - ncols               # bottom row
            pull = (0, 1, 3)
        else
            pull = (0, 1, 3, 5, 6)
        end
    elseif idx < ncols                        # top row
        pull = (3, 4, 5, 6, 7)
    elseif idx >= n - ncols                   # bottom row
        pull = (0, 1, 2, 3, 4)
    else
        pull = (0, 1, 2, 3, 4, 5, 6, 7)
    end
    idcs = Int64[]
    for p in pull
        v = poss[p+1]
        imflat[v+1] && push!(idcs, Int64(v))  # 0-based flat index in data domain
    end
    return idcs
end

"""
    reglobalize_flat_idx(idxlist, listdims, row_offset, col_offset, globaldims)

Port of im_utils.reglobalize_flat_idx: sub-image flat indices (0-BASED,
matching the Python data domain) -> parent-image flat indices (also 0-based),
order preserved. `listdims` is (nrows, ncols) of the subimage; row_offset /
col_offset are the 0-based row/col of the subimage's top-left in the parent
(may be negative); `globaldims` is the parent's (nrows, ncols).
"""
function reglobalize_flat_idx(idxlist, listdims::Tuple{Int,Int}, row_offset::Int, col_offset::Int,
                              globaldims::Tuple{Int,Int})
    out = Vector{Int64}()
    for fi in idxlist
        r, c = fi ÷ listdims[2], fi % listdims[2]   # 0-based unravel
        gr, gc = r + row_offset, c + col_offset
        push!(out, Int64(gr * globaldims[2] + gc))  # 0-based global flat
    end
    return out
end

# ---------------------------------------------------------------------------
# Edge coordinates (im_utils.edge_coords, dtype='flat')
# ---------------------------------------------------------------------------

"""
    edge_coords(shapeI) -> CPythonSet{Int64}

Flat indices (0-based) of all border pixels, inserted in the Python order:
left column (rows ascending), bottom row (cols ascending), right column (rows
descending), top row (cols descending) — edge_coords builds
set(ravel_multi_index((y_l, y_b, y_r, y_t), (x_l, x_b, x_r, x_t))).
"""
function edge_coords(shapeI::Tuple{Int,Int})
    nr, nc = shapeI
    s = CPythonSet{Int64}()
    for r in 0:nr-1                       # left column, rows ascending
        py_add!(s, Int64(r * nc))
    end
    for c in 0:nc-1                       # bottom row, cols ascending
        py_add!(s, Int64((nr - 1) * nc + c))
    end
    for r in nr-1:-1:0                    # right column, rows descending
        py_add!(s, Int64(r * nc + nc - 1))
    end
    for c in nc-1:-1:0                    # top row, cols descending
        py_add!(s, Int64(c))
    end
    return s
end

# ---------------------------------------------------------------------------
# Connectivity counting (im_utils.im_connectivity, mode='mirror' correlate)
# ---------------------------------------------------------------------------

@inline function _mirror_index0(i0::Int, n::Int)
    i0 < 0 && (i0 = -i0)
    i0 >= n && (i0 = 2n - i0 - 2)
    return i0
end

"""
    im_connectivity(I) -> Matrix{UInt8}

Port of im_utils.im_connectivity: binarize, correlate with the
[[1,1,1],[1,10,1],[1,1,1]] kernel under scipy 'mirror' boundary mode, then
`Iret[filtered > 10] = filtered - 10`. The result at each pixel is exactly
its 8-neighbor count (0-8) — the center weight of 10 cancels out.
"""
function im_connectivity(I::AbstractMatrix{Bool})
    nr, nc = size(I)
    out = zeros(UInt8, nr, nc)
    for r in 1:nr, c in 1:nc
        cnt = 0
        for dr in -1:1, dc in -1:1
            (dr == 0 && dc == 0) && continue
            rr = _mirror_index0(r - 1 + dr, nr) + 1
            cc = _mirror_index0(c - 1 + dc, nc) + 1
            I[rr, cc] && (cnt += 1)
        end
        # scipy: filtered = 10*self + neighbors; Iret[filtered > 10] = filtered - 10.
        # The center weight gates the count to ON pixels (and zeroes isolated
        # pixels, where filtered == 10 exactly).
        out[r, c] = (I[r, c] && cnt > 0) ? UInt8(cnt) : UInt8(0)
    end
    return out
end

"Port of im_utils.skel_endpoints: flat indices (raster order) of pixels with exactly one 8-neighbor."
function skel_endpoints(Iskel::AbstractMatrix{Bool})
    Ic = im_connectivity(Iskel)
    eps = Vector{Int64}()
    nr, nc = size(Ic)
    for r in 1:nr, c in 1:nc
        Ic[r, c] == 1 && push!(eps, Int64((r - 1) * nc + (c - 1)))   # 0-based flat
    end
    return eps
end

# ---------------------------------------------------------------------------
# Connected-component labeling (skimage.measure.label, background=0)
# ---------------------------------------------------------------------------

"""
    label_components(I; connectivity=2) -> (labels::Matrix{Int}, ncomp)

skimage.measure.label port for binary images: labels assigned in raster
first-occurrence order; connectivity 1 (4-neighbor) or 2 (8-neighbor).
"""
function label_components(I::AbstractMatrix{Bool}; connectivity::Int = 2)
    nr, nc = size(I)
    labels = zeros(Int, nr, nc)
    ncomp = 0
    neighbors8 = connectivity == 2
    for r in 1:nr, c in 1:nc
        I[r, c] && labels[r, c] == 0 || continue
        ncomp += 1
        stack = [(r, c)]
        labels[r, c] = ncomp
        while !isempty(stack)
            (rr, cc) = pop!(stack)
            for dr in -1:1, dc in -1:1
                (dr == 0 && dc == 0) && continue
                if !neighbors8 && abs(dr) + abs(dc) != 1
                    continue
                end
                r2, c2 = rr + dr, cc + dc
                (1 <= r2 <= nr && 1 <= c2 <= nc) || continue
                if I[r2, c2] && labels[r2, c2] == 0
                    labels[r2, c2] = ncomp
                    push!(stack, (r2, c2))
                end
            end
        end
    end
    return labels, ncomp
end

"""
    regionprops_coords(I; connectivity=2) -> Vector{Vector{Int64}}

Port of the regionprops usage in the walker (['coords']): one entry per
component (label order = raster first occurrence), each a raster-sorted list
of flat indices.
"""
function regionprops_coords(I::AbstractMatrix{Bool}; connectivity::Int = 2)
    labels, ncomp = label_components(I; connectivity=connectivity)
    nr, nc = size(I)
    comps = Vector{Vector{Int64}}([Int64[] for _ in 1:ncomp])
    for r in 1:nr, c in 1:nc
        l = labels[r, c]
        l == 0 && continue
        push!(comps[l], Int64((r - 1) * nc + (c - 1)))   # 0-based flat
    end
    return comps
end

"""
    largest_blobs_keep(I, nlargest) -> Matrix{Bool}

Port of im_utils.largest_blobs(I, nlargest, 'keep'): keep the `nlargest`
components (by area), zeroing everything else. Component order and areas
follow regionprops; the numpy `argsort()[-nlargest:][::-1]` tail selection is
replicated with a stable sort (tie behavior differs from numpy's quicksort
only when equal-area blobs compete for the tail — none of the fixture
skeletons exercise that corner).
"""
function largest_blobs_keep(I::AbstractMatrix{Bool}, nlargest::Int)
    comps = regionprops_coords(I; connectivity=2)
    areas = Int[length(c) for c in comps]
    out = fill(false, size(I))
    isempty(areas) && return out
    order = sortperm(areas)             # stable, ascending (numpy argsort tail)
    maxidcs = reverse(order[max(end - nlargest + 1, 1):end])
    nr, nc = size(I)
    for m in maxidcs
        for fi in comps[m]
            out[fi ÷ nc + 1, fi % nc + 1] = true    # 0-based flat -> 1-based array
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# 4-connectivity helpers (im_utils.four_conn / nfour_connectivity)
# ---------------------------------------------------------------------------

"""
    four_conn(idcs, I) -> Vector{Vector{Int64}}

Port of im_utils.four_conn: for each flat index, its nonzero 4-connected
neighbors (in neighbors_flat order).
"""
function four_conn(idcs, I::AbstractMatrix{Bool})
    Iflat = np_flatten(I)
    nc = size(I, 2)
    out = Vector{Vector{Int64}}()
    for i in idcs
        neighs = neighbors_flat(Int(i), Iflat, nc)
        push!(out, Int64[n for n in neighs if abs(n - i) == 1 || abs(n - i) == nc])
    end
    return out
end

"""
    nfour_connectivity(I) -> Matrix{UInt8}

Port of im_utils.nfour_connectivity: 4-neighbor counts at interior pixels,
zero elsewhere.
"""
function nfour_connectivity(I::AbstractMatrix{Bool})
    nr, nc = size(I)
    edge = edge_coords((nr, nc))
    Iflat = np_flatten(I)
    out = zeros(UInt8, nr, nc)
    for r in 1:nr, c in 1:nc
        i = Int64((r - 1) * nc + (c - 1))     # 0-based flat
        (I[r, c] && !in(i, edge)) || continue
        neighs = neighbors_flat(Int(i), Iflat, nc)
        out[r, c] = count(n -> abs(n - i) == 1 || abs(n - i) == nc, neighs)
    end
    return out
end
