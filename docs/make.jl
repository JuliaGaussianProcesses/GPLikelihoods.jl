using Documenter, NonConjugateGPs

makedocs(;
    modules=[NonConjugateGPs],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaGaussianProcesses/NonConjugateGPs.jl/blob/{commit}{path}#L{line}",
    sitename="NonConjugateGPs.jl",
    authors="willtebbutt <wt0881@my.bristol.ac.uk>",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/NonConjugateGPs.jl",
)
