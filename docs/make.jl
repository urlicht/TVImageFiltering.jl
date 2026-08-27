using Documenter
using TotalVariationImageFiltering

const CI = get(ENV, "CI", "false") == "true"

makedocs(
    modules = [TotalVariationImageFiltering],
    sitename = "TotalVariationImageFiltering.jl",
    checkdocs = :none,
    format = Documenter.HTML(
        prettyurls = CI,
        canonical = "https://urlicht.github.io/TotalVariationImageFiltering.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quick Start" => "quick-start.md",
        "Problem & API" => "problem-and-api.md",
        "ROF Solver" => "rof-solver.md",
        "PDHG Solver" => "pdhg-solver.md",
        "Group-Sparse TV Solver" => "gstv-solver.md",
        "Lambda Selection" => "lambda-selection.md",
        "Batch & CUDA" => "batch-and-cuda.md",
        "Benchmarking" => "benchmarking.md",
        "API Reference" => "api-reference.md",
        "References" => "references.md",
    ],
)

deploydocs(
    repo = "github.com/urlicht/TotalVariationImageFiltering.jl.git",
    devbranch = "main",
    versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
)
