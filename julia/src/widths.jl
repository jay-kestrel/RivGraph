# Link widths and lengths kernel (rivgraph/ln_utils.py:452, bug-for-bug)
# ======================================================================
# Port of link_widths_and_lengths. Idt is a float64 distance transform of
# the ORIGINAL (unpadded) mask; links come from the walker with pixel
# indices already unpadded. All flat indices are 0-based.

"Python's builtin max(a, b): the first argument wins any non-strict comparison."
_py_max(a::Float64, b::Float64) = b > a ? b : a

"""
    scan_mask(Imask, nrows, ncols, ray_r, ray_c, step_r, step_c) -> Float64

Port of the nested scan_mask (ln_utils.py:525-560): exact grid traversal
(Amanatides–Woo DDA) from a pixel center along (step_r, step_c) until the
ray leaves the mask or hits a zero pixel. Equal t_max steps BOTH axes
(dist = t_max_r). np.isclose(step, 0.0) uses the default atol of 1e-8.
"""
function scan_mask(Imask::AbstractMatrix{Bool}, nrows::Int, ncols::Int,
                   ray_r::Float64, ray_c::Float64, step_r::Float64, step_c::Float64)
    step_sign_r = abs(step_r) <= 1e-8 ? 0 : Int(sign(step_r))
    step_sign_c = abs(step_c) <= 1e-8 ? 0 : Int(sign(step_c))

    if step_sign_r == 0
        t_max_r = Inf
        t_delta_r = Inf
    else
        t_max_r = 0.5 / abs(step_r)
        t_delta_r = 1.0 / abs(step_r)
    end

    if step_sign_c == 0
        t_max_c = Inf
        t_delta_c = Inf
    else
        t_max_c = 0.5 / abs(step_c)
        t_delta_c = 1.0 / abs(step_c)
    end

    cur_r = Int(ray_r)
    cur_c = Int(ray_c)

    while true
        if t_max_r < t_max_c
            dist = t_max_r
            cur_r += step_sign_r
            t_max_r += t_delta_r
        elseif t_max_c < t_max_r
            dist = t_max_c
            cur_c += step_sign_c
            t_max_c += t_delta_c
        else
            dist = t_max_r
            cur_r += step_sign_r
            cur_c += step_sign_c
            t_max_r += t_delta_r
            t_max_c += t_delta_c
        end

        if cur_r < 0 || cur_r >= nrows || cur_c < 0 || cur_c >= ncols
            return dist
        end
        if Imask[cur_r+1, cur_c+1] == false
            return dist
        end
    end
end

"""
    width_along_normal(Imask, nrows, ncols, rows, cols, pix_i, fallback_width, pixlen)

Port of the nested width_along_normal: local width from the two normal rays;
falls back to the pixel's dt width for single-pixel links and zero norms.
"""
function width_along_normal(Imask::AbstractMatrix{Bool}, nrows::Int, ncols::Int,
                            rows::Vector{Int}, cols::Vector{Int},
                            pix_i::Int, fallback_width::Float64, pixlen::Float64)
    length(rows) == 1 && return fallback_width

    if pix_i == 0
        dr = rows[pix_i+2] - rows[pix_i+1]
        dc = cols[pix_i+2] - cols[pix_i+1]
    elseif pix_i == length(rows) - 1
        dr = rows[pix_i+1] - rows[pix_i]
        dc = cols[pix_i+1] - cols[pix_i]
    else
        dr = rows[pix_i+2] - rows[pix_i]
        dc = cols[pix_i+2] - cols[pix_i]
    end

    nr = -dc
    nc = dr
    norm = hypot(Float64(nr), Float64(nc))
    norm == 0 && return fallback_width

    nr = nr / norm
    nc = nc / norm

    left = scan_mask(Imask, nrows, ncols, Float64(rows[pix_i+1]), Float64(cols[pix_i+1]), nr, nc)
    right = scan_mask(Imask, nrows, ncols, Float64(rows[pix_i+1]), Float64(cols[pix_i+1]), -nr, -nc)

    return (left + right) * pixlen
end

"np.median: average of the two middle values for even lengths."
function _np_median(v::Vector{Float64})
    isempty(v) && return NaN
    s = sort(v)
    n = length(s)
    if n % 2 == 1
        return s[(n + 1) ÷ 2]
    else
        return (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2.0
    end
end

"np.argmin: index (1-based) of the FIRST minimum."
function _np_argmin(v::Vector{Float64})
    besti = 1
    bestv = v[1]
    for i in 2:length(v)
        if v[i] < bestv
            bestv = v[i]
            besti = i
        end
    end
    return besti
end

"""
    link_widths_and_lengths!(links::Links, Idt::Matrix{Float64}, pixlen::Float64=1.0) -> Links

Port of ln_utils.link_widths_and_lengths. Appends len, wid_pix, wid, wid_adj,
wid_med, len_adj, sinuosity per link. Numerical-equivalence contract: sums
are accumulated left-to-right (naive) rather than NumPy's pairwise scheme;
differences are ~1e-16 relative, within the 1e-12 stage tolerance. The
sinuosity denominator can be zero for single-pixel links — NumPy semantics
apply (0/0 = NaN, no exception).
"""
function link_widths_and_lengths!(links::Links, Idt::Matrix{Float64}, pixlen::Float64 = 1.0)
    dims = size(Idt)
    nrows, ncols = dims
    Imask = Idt .> 0.0

    links.wid_pix = Vector{Vector{Float64}}()
    links.len = Float64[]
    links.wid = Float64[]
    links.wid_adj = Vector{Float64}()      # NOTE: floats, not per-pixel arrays
    links.wid_med = Float64[]
    links.len_adj = Float64[]
    links.sinuosity = Float64[]

    width_mult = 1.1

    # Get widths at each pixel along each link
    for li in links.idx
        rows = Int[li0 ÷ ncols for li0 in li]          # np.unravel_index, 0-based
        cols = Int[li0 % ncols for li0 in li]
        dt_widths = Float64[Idt[r+1, c+1] * 2 * pixlen for (r, c) in zip(rows, cols)]
        # width_along_normal takes a 0-based pix_i (Python enumerate contract)
        widths = Float64[width_along_normal(Imask, nrows, ncols, rows, cols, i - 1, fw, pixlen)
                         for (i, fw) in enumerate(dt_widths)]
        push!(links.wid_pix, widths)
    end

    # Compute trimmed/untrimmed link widths and lengths
    for (li, widths) in zip(links.idx, links.wid_pix)
        rows = Int[li0 ÷ ncols for li0 in li]
        cols = Int[li0 % ncols for li0 in li]

        # Distances along link (cumsum of step lengths, prepended 0)
        dists = Float64[0.0]
        acc = 0.0
        for k in 1:length(li)-1
            d2 = Float64((rows[k+1] - rows[k])^2 + (cols[k+1] - cols[k])^2)
            acc += sqrt(d2)
            push!(dists, acc)
        end
        dists .*= pixlen

        # Distances along link in opposite direction
        revdists = Float64[0.0]
        acc = 0.0
        for k in length(li)-1:-1:1
            d2 = Float64((rows[k+1] - rows[k])^2 + (cols[k+1] - cols[k])^2)
            acc += sqrt(d2)
            push!(revdists, acc)
        end
        revdists .*= pixlen

        # First/last pixel at least a half-width's distance away
        startidx = _np_argmin(Float64[abs(d - widths[1] / 2 * width_mult) for d in dists]) - 1  # 0-based
        endidx = length(dists) -
                 _np_argmin(Float64[abs(d - widths[end] / 2 * width_mult) for d in revdists])  # 0-based

        if startidx >= endidx
            push!(links.wid_adj, _np_mean(widths))
            push!(links.wid_med, _np_median(widths))
            push!(links.len_adj, dists[end])
            st_dist = sqrt(Float64((rows[1] - rows[end])^2 + (cols[1] - cols[end])^2)) * pixlen
            push!(links.sinuosity, dists[end] / st_dist)
        else
            seg = widths[startidx+1:endidx]      # numpy slice [startidx:endidx]
            push!(links.wid_adj, _np_mean(seg))
            push!(links.wid_med, _np_median(seg))
            push!(links.len_adj, dists[endidx+1] - dists[startidx+1])
            st_dist = sqrt(Float64((rows[startidx+1] - rows[endidx+1])^2 +
                                   (cols[startidx+1] - cols[endidx+1])^2)) * pixlen
            push!(links.sinuosity, (dists[endidx+1] - dists[startidx+1]) / st_dist)
        end

        # Unadjusted lengths and widths
        push!(links.wid, _np_mean(widths))
        push!(links.len, dists[end])

        # Minimum width and length equal to the pixel resolution
        links.wid[end] = _py_max(pixlen, links.wid[end])
        links.len[end] = _py_max(pixlen, links.len[end])
        links.wid_adj[end] = _py_max(pixlen, links.wid_adj[end])
        links.len_adj[end] = _py_max(pixlen, links.len_adj[end])
    end

    return links
end

"np.mean: left-to-right sum divided by n (pairwise differences are ~1e-16)."
function _np_mean(v::Vector{Float64})
    isempty(v) && return NaN
    s = 0.0
    for x in v
        s += x
    end
    return s / length(v)
end
