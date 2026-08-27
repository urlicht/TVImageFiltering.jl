#!/usr/bin/env julia

using BenchmarkTools
using Dates
using Printf
using Random
using TestImages
using TotalVariationImageFiltering

const DEFAULT_SEED = 20260318
const DEFAULT_OUTPUT = normpath(joinpath(@__DIR__, "results", "benchmark_results.csv"))

struct BenchConfig
    backend::Symbol
    samples::Int
    evals::Int
    seed::Int
    quick::Bool
    output::Union{Nothing,String}
end

function parse_backend(token::AbstractString)
    backend = Symbol(lowercase(strip(token)))
    backend in (:cpu, :cuda, :both) || error("backend must be cpu, cuda, or both")
    return backend
end

function parse_args(args::Vector{String})
    cfg = Dict(
        :backend => :both,
        :samples => 20,
        :evals => 1,
        :seed => DEFAULT_SEED,
        :quick => false,
        :output => DEFAULT_OUTPUT,
    )

    for arg in args
        if arg == "--help" || arg == "-h"
            println(
                """
Usage:
  julia --project=benchmark benchmark/run_benchmarks.jl [options]

Options:
  --backend=cpu|cuda|both      Benchmark backend(s). Default: both
  --samples=N                  BenchmarkTools samples per case. Default: 20
  --evals=N                    BenchmarkTools evals per sample. Default: 1
  --seed=N                     RNG seed used for noise generation. Default: $(DEFAULT_SEED)
  --quick                      Smaller/faster benchmark set.
  --output=PATH                CSV output path. Default: benchmark/results/benchmark_results.csv
  --no-output                  Disable CSV output.

Backward-compatible aliases:
  --no-gpu                     Equivalent to --backend=cpu
""",
            )
            exit(0)
        elseif arg == "--quick"
            cfg[:quick] = true
        elseif arg == "--no-gpu"
            cfg[:backend] = :cpu
        elseif arg == "--no-output"
            cfg[:output] = nothing
        elseif startswith(arg, "--backend=")
            cfg[:backend] = parse_backend(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--samples=")
            cfg[:samples] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--evals=")
            cfg[:evals] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--seed=")
            cfg[:seed] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--output=")
            cfg[:output] = split(arg, "=", limit = 2)[2]
        else
            error("Unknown option: $arg")
        end
    end

    cfg[:samples] > 0 || error("samples must be positive")
    cfg[:evals] > 0 || error("evals must be positive")

    return BenchConfig(
        cfg[:backend],
        cfg[:samples],
        cfg[:evals],
        cfg[:seed],
        cfg[:quick],
        cfg[:output],
    )
end

function benchmark_cases(; quick::Bool)
    quick && return [
        (
            name = "cameraman-gaussian",
            image = "cameraman",
            mode = TotalVariationImageFiltering.IsotropicTV(),
            lambda = 0.12f0,
            gaussian_sigma = 0.04f0,
            salt_pepper_prob = 0.00f0,
            gpu = true,
        ),
        (
            name = "pirate-gaussian",
            image = "pirate",
            mode = TotalVariationImageFiltering.AnisotropicTV(),
            lambda = 0.16f0,
            gaussian_sigma = 0.06f0,
            salt_pepper_prob = 0.00f0,
            gpu = true,
        ),
    ]

    return [
        (
            name = "cameraman-gaussian",
            image = "cameraman",
            mode = TotalVariationImageFiltering.IsotropicTV(),
            lambda = 0.12f0,
            gaussian_sigma = 0.04f0,
            salt_pepper_prob = 0.00f0,
            gpu = true,
        ),
        (
            name = "pirate-gaussian",
            image = "pirate",
            mode = TotalVariationImageFiltering.AnisotropicTV(),
            lambda = 0.16f0,
            gaussian_sigma = 0.06f0,
            salt_pepper_prob = 0.00f0,
            gpu = true,
        ),
        (
            name = "woman_blonde-mixed",
            image = "woman_blonde",
            mode = TotalVariationImageFiltering.IsotropicTV(),
            lambda = 0.14f0,
            gaussian_sigma = 0.03f0,
            salt_pepper_prob = 0.02f0,
            gpu = true,
        ),
        (
            name = "mri-stack-gaussian",
            image = "mri-stack",
            mode = TotalVariationImageFiltering.IsotropicTV(),
            lambda = 0.10f0,
            gaussian_sigma = 0.02f0,
            salt_pepper_prob = 0.00f0,
            gpu = true,
        ),
        (
            name = "resolution_test_1920-large",
            image = "resolution_test_1920",
            mode = TotalVariationImageFiltering.IsotropicTV(),
            lambda = 0.10f0,
            gaussian_sigma = 0.02f0,
            salt_pepper_prob = 0.00f0,
            gpu = true,
        ),
    ]
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

function maybe_downsample(data::AbstractArray{Float32}; quick::Bool)
    quick || return data
    idx = ntuple(d -> 1:2:size(data, d), ndims(data))
    return copy(view(data, idx...))
end

function add_noise(
    clean::AbstractArray{Float32};
    seed::Int,
    gaussian_sigma::Float32,
    salt_pepper_prob::Float32,
)
    rng = MersenneTwister(seed)
    noisy = copy(clean)

    @inbounds for i in eachindex(noisy)
        noisy[i] = clamp(noisy[i] + gaussian_sigma * randn(rng, Float32), 0.0f0, 1.0f0)
    end

    if salt_pepper_prob > 0.0f0
        half = salt_pepper_prob / 2.0f0
        upper = 1.0f0 - half
        @inbounds for i in eachindex(noisy)
            r = rand(rng, Float32)
            if r < half
                noisy[i] = 0.0f0
            elseif r > upper
                noisy[i] = 1.0f0
            end
        end
    end

    return noisy
end

function load_case(case; quick::Bool, seed::Int)
    raw = TestImages.testimage(case.image)
    clean = Array(Float32.(raw))
    normalize01!(clean)
    clean = maybe_downsample(clean; quick = quick)

    noisy = add_noise(
        clean;
        seed = seed,
        gaussian_sigma = case.gaussian_sigma,
        salt_pepper_prob = case.salt_pepper_prob,
    )

    return (
        name = case.name,
        image = case.image,
        mode = case.mode,
        lambda = case.lambda,
        gaussian_sigma = case.gaussian_sigma,
        salt_pepper_prob = case.salt_pepper_prob,
        gpu = case.gpu,
        noisy = noisy,
    )
end

function prepare_cases(cfg::BenchConfig)
    prepared = NamedTuple[]

    for (i, case) in enumerate(benchmark_cases(; quick = cfg.quick))
        loaded = try
            load_case(case; quick = cfg.quick, seed = cfg.seed + i)
        catch err
            @warn "Skipping benchmark case because image loading failed." case =
                case.name image = case.image error = err
            continue
        end

        @info "Prepared benchmark case." case = loaded.name image = loaded.image size =
            size(loaded.noisy) gaussian_sigma = loaded.gaussian_sigma salt_pepper_prob =
            loaded.salt_pepper_prob
        push!(prepared, loaded)
    end

    isempty(prepared) && error(
        "No benchmark cases could be prepared from TestImages.jl. Ensure images are available/downloadable.",
    )

    return prepared
end

dims_string(shape::Tuple) = join(shape, "x")

function push_summary!(
    rows,
    backend::String,
    case_name::String,
    image_name::String,
    dims::String,
    cfg::BenchConfig,
    trial::BenchmarkTools.Trial,
)
    med = BenchmarkTools.median(trial)
    mn = BenchmarkTools.mean(trial)
    mnm = BenchmarkTools.minimum(trial)
    push!(
        rows,
        (
            backend = backend,
            case = case_name,
            image = image_name,
            dims = dims,
            median_ms = med.time / 1e6,
            mean_ms = mn.time / 1e6,
            min_ms = mnm.time / 1e6,
            memory_bytes = med.memory,
            allocs = med.allocs,
            samples = cfg.samples,
            evals = cfg.evals,
            seed = cfg.seed,
            quick = cfg.quick,
        ),
    )
end

function run_cpu_benchmarks(cfg::BenchConfig, cases, rows)
    println("Running CPU benchmarks...")

    maxiter = cfg.quick ? 120 : 240

    for case in cases
        dims = dims_string(size(case.noisy))
        problem = TotalVariationImageFiltering.TVProblem(
            case.noisy;
            lambda = case.lambda,
            tv_mode = case.mode,
        )
        solver_cfg = TotalVariationImageFiltering.ROFConfig(
            maxiter = maxiter,
            tau = 0.1f0,
            tol = 0.0f0,
            check_every = maxiter,
        )

        u = similar(case.noisy)
        state = TotalVariationImageFiltering.ROFState(case.noisy)

        TotalVariationImageFiltering.solve(problem, solver_cfg)
        trial = run(
            @benchmarkable TotalVariationImageFiltering.solve($problem, $solver_cfg) samples =
                cfg.samples evals = cfg.evals
        )
        push_summary!(rows, "cpu", "solve_allocating", case.image, dims, cfg, trial)

        copyto!(u, case.noisy)
        TotalVariationImageFiltering.solve!(u, problem, solver_cfg; state = state)
        trial = run(
            @benchmarkable begin
                copyto!($u, $(case.noisy))
                TotalVariationImageFiltering.solve!(
                    $u,
                    $problem,
                    $solver_cfg;
                    state = $state,
                )
            end samples = cfg.samples evals = cfg.evals
        )
        push_summary!(rows, "cpu", "solve_state_reuse", case.image, dims, cfg, trial)
    end
end

function maybe_load_cuda()
    Base.find_package("CUDA") === nothing && return nothing
    try
        @eval using CUDA
        cuda = Base.invokelatest(() -> getfield(@__MODULE__, :CUDA))
        Base.invokelatest(cuda.functional) || return nothing
        Base.get_extension(
            TotalVariationImageFiltering,
            :TotalVariationImageFilteringCUDAExt,
        ) === nothing && return nothing
        return cuda
    catch
        return nothing
    end
end

function run_cuda_benchmarks(cfg::BenchConfig, cases, rows)
    CUDA = maybe_load_cuda()
    if CUDA === nothing
        println(
            "Skipping CUDA benchmarks (CUDA not installed, extension inactive, or no functional CUDA device).",
        )
        return
    end

    println("Running CUDA benchmarks...")
    Base.invokelatest(_run_cuda_benchmarks_loaded, CUDA, cfg, cases, rows)
end

function _run_cuda_benchmarks_loaded(CUDA, cfg::BenchConfig, cases, rows)
    maxiter = cfg.quick ? 120 : 240
    gpu_cases = filter(case -> case.gpu, cases)

    for case in gpu_cases
        dims = dims_string(size(case.noisy))
        noisy_gpu = CUDA.CuArray(case.noisy)

        problem = TotalVariationImageFiltering.TVProblem(
            noisy_gpu;
            lambda = case.lambda,
            tv_mode = case.mode,
        )
        solver_cfg = TotalVariationImageFiltering.ROFConfig(
            maxiter = maxiter,
            tau = 0.1f0,
            tol = 0.0f0,
            check_every = maxiter,
        )

        u_gpu = similar(noisy_gpu)
        state = TotalVariationImageFiltering.ROFState(noisy_gpu)

        TotalVariationImageFiltering.solve(problem, solver_cfg)
        CUDA.synchronize()
        trial = run(
            @benchmarkable begin
                TotalVariationImageFiltering.solve($problem, $solver_cfg)
                CUDA.synchronize()
            end samples = cfg.samples evals = cfg.evals
        )
        push_summary!(rows, "cuda", "solve_allocating", case.image, dims, cfg, trial)

        copyto!(u_gpu, noisy_gpu)
        TotalVariationImageFiltering.solve!(u_gpu, problem, solver_cfg; state = state)
        CUDA.synchronize()
        trial = run(
            @benchmarkable begin
                copyto!($u_gpu, $noisy_gpu)
                TotalVariationImageFiltering.solve!(
                    $u_gpu,
                    $problem,
                    $solver_cfg;
                    state = $state,
                )
                CUDA.synchronize()
            end samples = cfg.samples evals = cfg.evals
        )
        push_summary!(rows, "cuda", "solve_state_reuse", case.image, dims, cfg, trial)
    end
end

function print_rows(rows)
    println()
    println("Benchmark results (times in ms):")
    println(
        rpad("backend", 8),
        rpad("case", 20),
        rpad("image", 24),
        rpad("dims", 16),
        lpad("median", 10),
        lpad("mean", 10),
        lpad("min", 10),
        lpad("memory", 12),
        lpad("allocs", 10),
    )

    for r in rows
        @printf(
            "%-8s%-20s%-24s%-16s%10.3f%10.3f%10.3f%12d%10d\n",
            r.backend,
            r.case,
            r.image,
            r.dims,
            r.median_ms,
            r.mean_ms,
            r.min_ms,
            r.memory_bytes,
            r.allocs,
        )
    end
end

function csv_escape(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_csv(path::String, rows)
    mkpath(dirname(path))
    timestamp = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS")
    open(path, "w") do io
        println(
            io,
            "run_timestamp_utc,backend,case,image,dims,median_ms,mean_ms,min_ms,memory_bytes,allocs,samples,evals,seed,quick",
        )
        for r in rows
            cols = (
                timestamp,
                r.backend,
                r.case,
                r.image,
                r.dims,
                @sprintf("%.6f", r.median_ms),
                @sprintf("%.6f", r.mean_ms),
                @sprintf("%.6f", r.min_ms),
                string(r.memory_bytes),
                string(r.allocs),
                string(r.samples),
                string(r.evals),
                string(r.seed),
                string(r.quick),
            )
            println(io, join(csv_escape.(cols), ","))
        end
    end
end

function main()
    cfg = parse_args(ARGS)
    cases = prepare_cases(cfg)
    rows = NamedTuple[]

    println("TotalVariationImageFiltering benchmark configuration:")
    println("  backend:  ", cfg.backend)
    println("  samples:  ", cfg.samples)
    println("  evals:    ", cfg.evals)
    println("  seed:     ", cfg.seed)
    println("  quick:    ", cfg.quick)
    println("  output:   ", something(cfg.output, "<disabled>"))

    if cfg.backend in (:cpu, :both)
        run_cpu_benchmarks(cfg, cases, rows)
    end
    if cfg.backend in (:cuda, :both)
        run_cuda_benchmarks(cfg, cases, rows)
    end

    print_rows(rows)

    if cfg.output !== nothing
        write_csv(cfg.output, rows)
        println()
        println("Wrote CSV: ", cfg.output)
    end
end

main()
