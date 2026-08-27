using Test
using TotalVariationImageFiltering

include("test_types.jl")
include("test_problem.jl")
include("test_operators.jl")
include("test_solver_rof.jl")
include("test_solver_pdhg.jl")
include("test_solver_gstv.jl")
include("test_api.jl")
include("test_batch_api.jl")
include("test_lambda_selection.jl")
include("test_cuda_extension.jl")
