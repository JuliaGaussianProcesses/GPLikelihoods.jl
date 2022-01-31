using Documenter, GPLikelihoods

makedocs(;
    modules=[GPLikelihoods],
    format=Documenter.HTML(),
    pages=["Home" => "index.md", "API" => "api.md"],
    repo="https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl/blob/{commit}{path}#L{line}",
    sitename="GPLikelihoods.jl",
    authors="JuliaGaussianProcesses organization",
    assets=String[],
)

deploydocs(; repo="github.com/JuliaGaussianProcesses/GPLikelihoods.jl", push_preview=true)
