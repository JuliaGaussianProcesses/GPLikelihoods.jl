using Documenter, LatentGPs

makedocs(;
    modules=[LatentGPs],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaGaussianProcesses/LatentGPs.jl/blob/{commit}{path}#L{line}",
    sitename="LatentGPs.jl",
    authors="willtebbutt <wt0881@my.bristol.ac.uk>",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/LatentGPs.jl",
)
