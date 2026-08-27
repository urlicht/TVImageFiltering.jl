#!/usr/bin/env julia

using BenchmarkTools
using Dates
using Printf
using Random
using TestImages
using TotalVariationImageFiltering

if Base.find_package("CUDA") !== nothing
    @eval using CUDA
end

const DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256]
const DEFAULT_OUTPUT =
    normpath(joinpath(@__DIR__, "results", "cuda_batching_vs_no_batching.csv"))
const DEFAULT_SEED = 20260318

struct BatchBenchConfig
    samples::Int
    evals::Int
    seed::Int
    batch_sizes::Vector{Int}
    image_size::Tuple{Int,Int}
    maxiter::Int
    lambda::Float32
    tau::Float32
    noise_sigma::Float32
    output::String
end

function parse_dim2(token::AbstractString)
    m = match(r"^(\d+)x(\d+)$", lowercase(strip(token)))
    m === nothing && error("Invalid size '$token'. Use NxM (e.g. 128x128).")
    return parse(Int, m.captures[1]), parse(Int, m.captures[2])
end

function parse_batch_sizes(token::AbstractString)
    out = Int[]
    for s in split(token, ",")
        n = parse(Int, strip(s))
        n > 0 || error("Batch sizes must be positive: $n")
        push!(out, n)
    end
    isempty(out) && error("Provide at least one batch size.")
    return out
end

function parse_args(args::Vector{String})
    cfg = Dict(
        :samples => 10,
        :evals => 1,
        :seed => DEFAULT_SEED,
        :batch_sizes => DEFAULT_BATCH_SIZES,
        :image_size => (128, 128),
        :maxiter => 120,
        :lambda => 0.12f0,
        :tau => 0.1f0,
        :noise_sigma => 0.04f0,
        :output => DEFAULT_OUTPUT,
    )

    for arg in args
        if arg == "--help" || arg == "-h"
            println("""
Usage:
  julia --project=benchmark benchmark/compare_batching_cuda.jl [options]

Options:
  --samples=N                   BenchmarkTools samples per case. Default: 10
  --evals=N                     BenchmarkTools evals per sample. Default: 1
  --seed=N                      RNG seed. Default: $(DEFAULT_SEED)
  --batch-sizes=1,4,8,...       Batch sizes to compare.
                                Default: 1,4,8,16,32,64,128,256
  --size=128x128                Base image size. Default: 128x128
  --maxiter=N                   ROF iterations. Default: 120
  --lambda=0.12                 TV regularization weight. Default: 0.12
  --tau=0.1                     ROF dual step. Default: 0.1
  --noise=0.04                  Gaussian noise sigma. Default: 0.04
  --output=PATH                 CSV output path.
                                Default: benchmark/results/cuda_batching_vs_no_batching.csv
""")
            exit(0)
        elseif startswith(arg, "--samples=")
            cfg[:samples] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--evals=")
            cfg[:evals] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--seed=")
            cfg[:seed] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--batch-sizes=")
            cfg[:batch_sizes] = parse_batch_sizes(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--size=")
            cfg[:image_size] = parse_dim2(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--maxiter=")
            cfg[:maxiter] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--lambda=")
            cfg[:lambda] = Float32(parse(Float64, split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--tau=")
            cfg[:tau] = Float32(parse(Float64, split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--noise=")
            cfg[:noise_sigma] = Float32(parse(Float64, split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--output=")
            cfg[:output] = split(arg, "=", limit = 2)[2]
        else
            error("Unknown option: $arg")
        end
    end

    cfg[:samples] > 0 || error("samples must be positive")
    cfg[:evals] > 0 || error("evals must be positive")
    cfg[:maxiter] > 0 || error("maxiter must be positive")
    cfg[:lambda] > 0.0f0 || error("lambda must be positive")
    cfg[:tau] > 0.0f0 || error("tau must be positive")
    cfg[:noise_sigma] >= 0.0f0 || error("noise must be non-negative")

    return BatchBenchConfig(
        cfg[:samples],
        cfg[:evals],
        cfg[:seed],
        cfg[:batch_sizes],
        cfg[:image_size],
        cfg[:maxiter],
        cfg[:lambda],
        cfg[:tau],
        cfg[:noise_sigma],
        cfg[:output],
    )
end

function require_cuda!()
    isdefined(@__MODULE__, :CUDA) || error(
        "CUDA.jl is not available in this environment. Run: julia --project=benchmark -e 'using Pkg; Pkg.add(\"CUDA\")'",
    )
    CUDA.functional() ||
        error("CUDA is installed but no functional CUDA device is available.")
    Base.get_extension(
        TotalVariationImageFiltering,
        :TotalVariationImageFilteringCUDAExt,
    ) === nothing && error(
        "TotalVariationImageFilteringCUDAExt is not active. Ensure CUDA is loaded before TotalVariationImageFiltering.",
    )
    return nothing
end

function normalize01!(data::AbstractArray{Float32})
    lo, hi = extrema(data)
    if hi > lo
        scale = inv(hi - lo)
        @inbounds for i in eachindex(data)
            data[i] = (data[i] - lo) * scale
        end
    else
        fill!(data, 0.0f0)
    end
    return data
end

function resize_nearest(img::AbstractMatrix{Float32}, nx::Int, ny::Int)
    sx, sy = size(img)
    sx == nx && sy == ny && return copy(img)
    ix = round.(Int, range(1, sx, length = nx))
    iy = round.(Int, range(1, sy, length = ny))
    return img[ix, iy]
end

function load_base_image(image_size::Tuple{Int,Int})
    raw = TestImages.testimage("cameraman")
    img = Array(Float32.(raw))
    normalize01!(img)
    return resize_nearest(img, image_size[1], image_size[2])
end

function add_gaussian_noise(clean::AbstractArray{Float32}; seed::Int, sigma::Float32)
    rng = MersenneTwister(seed)
    noisy = copy(clean)
    @inbounds for i in eachindex(noisy)
        noisy[i] = clamp(noisy[i] + sigma * randn(rng, Float32), 0.0f0, 1.0f0)
    end
    return noisy
end

function make_batch_inputs(
    clean::AbstractMatrix{Float32},
    batch_size::Int,
    seed::Int,
    sigma::Float32,
)
    out = Matrix{Float32}[]
    sizehint!(out, batch_size)
    for i = 1:batch_size
        push!(out, add_gaussian_noise(clean; seed = seed + i, sigma = sigma))
    end
    return out
end

function run_no_batch!(
    outputs::Vector,
    problems::Vector,
    solver_cfg::TotalVariationImageFiltering.ROFConfig,
    states::Vector,
)
    @inbounds for i in eachindex(problems)
        copyto!(outputs[i], problems[i].f)
        TotalVariationImageFiltering.solve!(
            outputs[i],
            problems[i],
            solver_cfg;
            state = states[i],
        )
    end
    return nothing
end

function run_batched!(
    output,
    problem,
    solver_cfg::TotalVariationImageFiltering.ROFConfig,
    state,
)
    copyto!(output, problem.f)
    TotalVariationImageFiltering.solve!(output, problem, solver_cfg; state = state)
    return nothing
end

function summarize_trial(trial::BenchmarkTools.Trial)
    med = BenchmarkTools.median(trial)
    mn = BenchmarkTools.mean(trial)
    mnm = BenchmarkTools.minimum(trial)
    return (
        median_ms = med.time / 1e6,
        mean_ms = mn.time / 1e6,
        min_ms = mnm.time / 1e6,
        memory_bytes = med.memory,
        allocs = med.allocs,
    )
end

function write_csv(path::String, rows, cfg::BatchBenchConfig)
    mkpath(dirname(path))
    timestamp = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS")
    size_token = "$(cfg.image_size[1])x$(cfg.image_size[2])"
    open(path, "w") do io
        println(
            io,
            "run_timestamp_utc,batch_size,no_batch_median_ms,batched_median_ms,speedup_no_batch_over_batched,no_batch_mean_ms,batched_mean_ms,no_batch_min_ms,batched_min_ms,no_batch_memory_bytes,batched_memory_bytes,no_batch_allocs,batched_allocs,samples,evals,maxiter,image_size,lambda,tau,noise_sigma,seed",
        )
        for r in rows
            @printf(
                io,
                "%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%d\n",
                timestamp,
                r.batch_size,
                r.no_batch.median_ms,
                r.batched.median_ms,
                r.speedup,
                r.no_batch.mean_ms,
                r.batched.mean_ms,
                r.no_batch.min_ms,
                r.batched.min_ms,
                r.no_batch.memory_bytes,
                r.batched.memory_bytes,
                r.no_batch.allocs,
                r.batched.allocs,
                cfg.samples,
                cfg.evals,
                cfg.maxiter,
                size_token,
                cfg.lambda,
                cfg.tau,
                cfg.noise_sigma,
                cfg.seed,
            )
        end
    end
end

function main()
    cfg = parse_args(ARGS)
    require_cuda!()

    clean = load_base_image(cfg.image_size)
    solver_cfg = TotalVariationImageFiltering.ROFConfig(
        maxiter = cfg.maxiter,
        tau = cfg.tau,
        tol = 0.0f0,
        check_every = cfg.maxiter,
    )

    println("CUDA batching benchmark configuration:")
    println("  batch sizes: ", join(cfg.batch_sizes, ", "))
    println("  image size:  ", cfg.image_size[1], "x", cfg.image_size[2])
    println("  samples:     ", cfg.samples)
    println("  evals:       ", cfg.evals)
    println("  maxiter:     ", cfg.maxiter)
    println("  lambda:      ", cfg.lambda)
    println("  tau:         ", cfg.tau)
    println("  noise sigma: ", cfg.noise_sigma)
    println("  output:      ", cfg.output)
    println()

    rows = NamedTuple[]

    println(
        rpad("batch", 8),
        lpad("no_batch_ms", 14),
        lpad("batched_ms", 14),
        lpad("speedup", 10),
    )

    for batch_size in cfg.batch_sizes
        cpu_images = make_batch_inputs(
            clean,
            batch_size,
            cfg.seed + 10_000 * batch_size,
            cfg.noise_sigma,
        )
        gpu_images = [CUDA.CuArray(img) for img in cpu_images]

        problems = [
            TotalVariationImageFiltering.TVProblem(
                img;
                lambda = cfg.lambda,
                tv_mode = TotalVariationImageFiltering.IsotropicTV(),
            ) for img in gpu_images
        ]
        outputs = [similar(img) for img in gpu_images]
        states = [TotalVariationImageFiltering.ROFState(img) for img in gpu_images]

        stack_cpu = cat(cpu_images...; dims = 3)
        stack_gpu = CUDA.CuArray(stack_cpu)
        # Use large spacing on batch axis to suppress inter-sample coupling in batched mode.
        batch_problem = TotalVariationImageFiltering.TVProblem(
            stack_gpu;
            lambda = cfg.lambda,
            spacing = (1.0f0, 1.0f0, 1.0f6),
            tv_mode = TotalVariationImageFiltering.IsotropicTV(),
        )
        batch_output = similar(stack_gpu)
        batch_state = TotalVariationImageFiltering.ROFState(stack_gpu)

        run_no_batch!(outputs, problems, solver_cfg, states)
        run_batched!(batch_output, batch_problem, solver_cfg, batch_state)
        CUDA.synchronize()

        trial_no_batch = run(
            @benchmarkable begin
                run_no_batch!($outputs, $problems, $solver_cfg, $states)
                CUDA.synchronize()
            end samples = cfg.samples evals = cfg.evals
        )

        trial_batched = run(
            @benchmarkable begin
                run_batched!($batch_output, $batch_problem, $solver_cfg, $batch_state)
                CUDA.synchronize()
            end samples = cfg.samples evals = cfg.evals
        )

        no_batch_stats = summarize_trial(trial_no_batch)
        batched_stats = summarize_trial(trial_batched)
        speedup = no_batch_stats.median_ms / batched_stats.median_ms

        row = (
            batch_size = batch_size,
            no_batch = no_batch_stats,
            batched = batched_stats,
            speedup = speedup,
        )
        push!(rows, row)

        @printf(
            "%-8d%14.3f%14.3f%10.3f\n",
            batch_size,
            no_batch_stats.median_ms,
            batched_stats.median_ms,
            speedup,
        )
    end

    write_csv(cfg.output, rows, cfg)
    println()
    println("Wrote CSV: ", cfg.output)
end

main()
