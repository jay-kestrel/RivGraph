# Julia-side benchmarks for the ported RivGraph kernels.
#
# Mirrors the Python 3-stage harness contract (bench/test_benchmark_rivgraph.py):
# 5 timed rounds, 1 warmup, identical committed Colville inputs, environment and
# input hashes recorded in the JSON artifact. Stage 1 (skel_to_graph) and stage 2
# (link_widths_and_lengths) are ported; stage 3 (assign_flow_directions) is not
# part of this port track and is reported as not_ported rather than silently
# omitted.
#
# Usage:  julia --project=julia julia/bench/benchmarks.jl [OUTPUT_JSON]
#         (default: bench/results/julia-wave1/julia-bench.json)

using BenchmarkTools
using JSON
using SHA
using Statistics

const REPO = normpath(@__DIR__, "..", "..")
const FIX = joinpath(REPO, "julia", "test", "fixtures")
const OUT = length(ARGS) >= 1 ? ARGS[1] : joinpath(REPO, "bench", "results", "julia-wave1", "julia-bench.json")

include(joinpath(@__DIR__, "..", "src", "RivGraphJulia.jl"))
using .RivGraphJulia

function sha256_file(path)
    h = SHA.SHA2_256_CTX()
    buf = Vector{UInt8}(undef, 1 << 20)
    open(path, "r") do f
        while !eof(f)
            n = readbytes!(f, buf)
            SHA.update!(h, view(buf, 1:n))
        end
    end
    return bytes2hex(SHA.digest!(h))
end

intvec(xs) = Int[x for x in xs]
intvecvec(xs) = Vector{Int}[intvec(row) for row in xs]

function links_from_ser(v::AbstractDict{String,Any})
    vals = v["values"]
    links = Links()
    links.idx = intvecvec(vals["idx"])
    links.conn = intvecvec(vals["conn"])
    links.id = OrderedSet{Int64}(intvec(vals["id"]))
    return links
end

function main()
    manifest = JSON.parsefile(joinpath(FIX, "manifest.json"))
    inputs = manifest["inputs"]

    # ---- Stage 1: skel_to_graph on the full Colville skeleton --------------
    wmeta1 = inputs["walker/colville"]["Iskel"]
    rows, cols = wmeta1["shape"]
    iskel = read_packed_bool_matrix(joinpath(FIX, "walker", "colville", wmeta1["file"]), rows, cols)
    # one warmup run, then 5 timed samples per the Python harness contract
    warm = skel_to_graph(iskel)
    stage1_samples = Float64[]
    for _ in 1:5
        t = @elapsed skel_to_graph(iskel)
        push!(stage1_samples, t)
    end

    # ---- Stage 2: link_widths_and_lengths over the pruned links -----------
    wmeta = inputs["widths/colville"]
    Idt = read_f64_matrix(joinpath(FIX, "widths", "colville", wmeta["Idt"]["file"]))
    links = links_from_ser(JSON.parsefile(joinpath(FIX, "widths", "colville", wmeta["links_in"]["file"])))
    pixlen = Float64(wmeta["pixlen"])
    link_widths_and_lengths!(links, Idt, pixlen)   # warmup
    stage2_samples = Float64[]
    for _ in 1:5
        fresh = deepcopy(links)
        t = @elapsed link_widths_and_lengths!(fresh, Idt, pixlen)
        push!(stage2_samples, t)
    end

    results = [
        Dict("name" => "test_benchmark_skel_to_graph[skel_to_graph]",
             "port_stage" => 1, "samples_s" => stage1_samples,
             "median_s" => median(stage1_samples), "min_s" => minimum(stage1_samples)),
        Dict("name" => "test_benchmark_link_widths[link_widths_and_lengths]",
             "port_stage" => 2, "samples_s" => stage2_samples,
             "median_s" => median(stage2_samples), "min_s" => minimum(stage2_samples)),
        Dict("name" => "test_benchmark_assign_flow_directions",
             "port_stage" => 3, "status" => "not_ported",
             "note" => "directionality is out of scope for wave 1; see the port spec"),
    ]

    artifact = Dict(
        "rivgraph_julia_bench_env" => Dict(
            "julia" => string(VERSION),
            "platform" => string(Sys.KERNEL, "-", Sys.ARCH),
            "machine" => string(Sys.ARCH),
            "cpus" => Sys.cpu_info() |> length,
            "rounds" => 5,
            "warmup" => 1,
            "timer" => "Base.@elapsed (monotonic)",
        ),
        "inputs" => Dict(
            "walker/colville/Iskel" => Dict(
                "file" => "walker/colville/Iskel.npy",
                "sha256" => sha256_file(joinpath(FIX, "walker", "colville", "Iskel.npy"))),
            "widths/colville/Idt" => Dict(
                "file" => "widths/colville/Idt.npy.gz",
                "sha256" => sha256_file(joinpath(FIX, "widths", "colville", "Idt.npy.gz"))),
            "widths/colville/links_in" => Dict(
                "file" => "widths/colville/links_in.json",
                "sha256" => sha256_file(joinpath(FIX, "widths", "colville", "links_in.json"))),
        ),
        "workload" => Dict(
            "stage1" => "skel_to_graph on the committed Colville skeleton",
            "stage2" => "link_widths_and_lengths on the committed pruned Colville links (pixlen=$(pixlen))",
        ),
        "results" => results,
    )

    mkpath(dirname(OUT))
    open(OUT, "w") do f
        JSON.print(f, artifact, 1)
    end
    println("wrote ", OUT)
    for r in results
        if haskey(r, "median_s")
            println(r["name"], ": median=", round(r["median_s"]; sigdigits=4), "s  min=", round(r["min_s"]; sigdigits=4), "s")
        else
            println(r["name"], ": ", r["status"])
        end
    end
end

main()
