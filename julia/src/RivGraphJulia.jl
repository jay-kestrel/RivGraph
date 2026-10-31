# RivGraphJulia — Julia port of RivGraph's core numeric functionality
# ===================================================================
# Wave-1 scope: skeleton-to-graph walker (walk.py + _skel_to_graph_python),
# link widths/lengths kernel, and the delta DAG steady-state + entropy-rate
# companions. Consumes Python-dumped .npy skeletons; parity oracle is the
# Python reference path (PYDELTARCM_NATIVE=0).

module RivGraphJulia

include("cpythonset.jl")
include("orderedset.jl")
include("npyio.jl")
include("imutils.jl")
include("lnutils.jl")
include("walk.jl")
include("mask_to_graph.jl")
include("widths.jl")
include("deltas.jl")

export skel_to_graph,
       _skel_to_graph_python,
       Links, Nodes,
       link_widths_and_lengths!,
       solve_adjacency_steady_state, delta_subN_F, nl_entropy_rate,
       AdjacencySteadyStateResult,
       CPythonSet, OrderedSet,
       read_npy, read_f64_matrix, read_packed_bool_matrix

end # module
