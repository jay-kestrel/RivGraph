# Minimal .npy reader/writer (+gzip) for parity fixtures
# ======================================================
# Supports the exact set of dtypes the Python fixture dumper emits
# (np.save with allow_pickle=False): '<f8', '<i4', '<i8', '|u1', '|b1',
# C order (fortran_order=False). Packed-bit skeletons are uint8 .npy files
# that the caller unpacks (np.packbits convention, row-major).

using CodecZlib

struct NpyArray
    data::Vector{UInt8}     # raw payload bytes
    dtype::Symbol           # :f8, :i4, :i8, :u1, :b1
    shape::Vector{Int}      # row-major (C order) shape
end

nrows(a::NpyArray) = a.shape[1]
ncols(a::NpyArray) = length(a.shape) == 1 ? 1 : a.shape[2]

const _NPY_DTYPES = Dict{String,Symbol}(
    "<f8" => :f8, "<i4" => :i4, "<i8" => :i8, "|u1" => :u1, "|b1" => :b1)

function _read_npy_header(io::IO)
    # .npy v1.0 format: magic, version, header len, dict
    magic = read(io, 6)
    magic == UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')] ||
        throw(ArgumentError("not a .npy file"))
    major = read(io, UInt8)
    minor = read(io, UInt8)
    hdrlen = major == 1 ? Int(read(io, UInt16)) : Int(read(io, UInt32))
    hdr = String(read(io, hdrlen))
    return hdr
end

function _parse_header_dict(hdr::String)
    # header looks like: {'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }
    m = match(r"'descr':\s*'([^']+)'", hdr)
    m === nothing && throw(ArgumentError("missing descr in npy header"))
    descr = String(m.captures[1])
    haskey(_NPY_DTYPES, descr) || throw(ArgumentError("unsupported dtype $descr"))
    dtype = _NPY_DTYPES[descr]
    fm = match(r"'fortran_order':\s*(True|False)", hdr)
    fortran = fm === nothing ? false : fm.captures[1] == "True"
    sm = match(r"'shape':\s*\(([^)]*)\)", hdr)
    sm === nothing && throw(ArgumentError("missing shape in npy header"))
    dims = [parse(Int, strip(d)) for d in split(sm.captures[1], ',') if !isempty(strip(d))]
    return dtype, fortran, dims
end

"Read a .npy file (optionally gzip-compressed; detected by magic bytes)."
function read_npy(path::AbstractString)
    raw = read(path)
    if length(raw) >= 2 && raw[1] == 0x1f && raw[2] == 0x8b
        raw = transcode(GzipDecompressor, raw)
    end
    io = IOBuffer(raw)
    hdr = _read_npy_header(io)
    dtype, fortran, dims = _parse_header_dict(hdr)
    fortran && throw(ArgumentError("fortran-order npy not supported (fixtures are C order)"))
    bytes = read(io)
    itemsize = dtype === :f8 ? 8 : dtype === :i4 ? 4 : dtype === :i8 ? 8 : 1
    expected = prod(dims) * itemsize
    length(bytes) >= expected || throw(ArgumentError("truncated npy payload"))
    return NpyArray(bytes[1:expected], dtype, dims)
end

"""Reinterpret a C-order (row-major) flat vector as a Julia array.

Julia reshape is column-major; numpy data is row-major. For 2-D shapes this is
reshape to (cols, rows) followed by a transpose. 1-D arrays pass through.
"""
function _rowmajor_reshape(out::Vector, shape::Vector{Int})
    if length(shape) == 1
        return out
    elseif length(shape) == 2
        rows, cols = shape[1], shape[2]
        return permutedims(reshape(out, cols, rows))  # row-major (rows, cols)
    end
    throw(ArgumentError("unsupported npy dimensionality $(length(shape))"))
end

"Unpack a packbits row-major bool image dumped by the fixture generator."
function unpack_bits(a::NpyArray)
    a.dtype === :u1 || throw(ArgumentError("expected uint8 packed bits"))
    packed = reinterpret(UInt8, a.data)
    total = prod(a.shape)
    out = Array{Bool}(undef, total)
    bitpos = 0
    for (bi, b) in enumerate(packed)
        for bit in 0:7
            bitpos = (bi - 1) * 8 + bit + 1
            bitpos > total && break
            out[bitpos] = ((b >> (7 - bit)) & 0x01) == 1
        end
    end
    return _rowmajor_reshape(out, a.shape)
end

"Materialize a float64 matrix from an .npy payload (C order)."
function read_f64_matrix(path::AbstractString)
    a = read_npy(path)
    a.dtype === :f8 || throw(ArgumentError("expected float64 npy"))
    vals = reinterpret(Float64, a.data)
    return _rowmajor_reshape(collect(vals), a.shape)
end

"""Materialize a bool matrix from a packed-bits .npy (fixture skeletons).

The packed npy is a flat uint8 vector (np.packbits, MSB-first, row-major);
the true 2-D shape is not recoverable from the file, so callers pass it.
"""
function read_packed_bool_matrix(path::AbstractString, rows::Int, cols::Int)
    a = read_npy(path)
    a.dtype === :u1 || throw(ArgumentError("expected uint8 packed bits"))
    packed = reinterpret(UInt8, a.data)
    total = rows * cols
    (length(packed) == cld(total, 8)) ||
        throw(ArgumentError("packed size $(length(packed)) does not match $rows x $cols"))
    out = Array{Bool}(undef, total)
    bitpos = 0
    for (bi, b) in enumerate(packed)
        for bit in 0:7
            bitpos = (bi - 1) * 8 + bit + 1
            bitpos > total && break
            out[bitpos] = ((b >> (7 - bit)) & 0x01) == 1
        end
    end
    return permutedims(reshape(out, cols, rows))  # row-major (rows, cols)
end
