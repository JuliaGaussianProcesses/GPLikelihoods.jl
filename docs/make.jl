using Documenter, GPLikelihoods

makedocs(;
    sitename="GPLikelihoods.jl",
    format=Documenter.HTML(),
    modules=[GPLikelihoods],
    pages=["Home" => "index.md", "API" => "api.md"],
    repo="https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl/blob/{commit}{path}#L{line}",
    authors="JuliaGaussianProcesses organization",
    assets=String[],
    strict=true,
    checkdocs=:exports,
    #doctestfilters=JuliaGPsDocs.DOCTEST_FILTERS,
)

deploydocs(; repo="github.com/JuliaGaussianProcesses/GPLikelihoods.jl", push_preview=true)
