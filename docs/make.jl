using Documenter

if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

Documenter.post_status(; 
    type="pending", 
    repo="github.com/JuliaGaussianProcesses/AbstractGPs.jl.git"
)

using Literate, GPLikelihoods

if ispath(joinpath(@__DIR__, "src/examples"))
    rm(joinpath(@__DIR__, "src/examples"), recursive=true)
end

for filename in readdir(joinpath(@__DIR__, "..", "examples"))
    endswith(filename, ".jl") || continue
	name = splitext(filename)[1]
    Literate.markdown(
        joinpath(@__DIR__, "..", "examples", filename),
        joinpath(@__DIR__, "src/examples");
        name = name,
        documenter=true,
    )
end

generated_examples = joinpath.("examples", filter(
    x -> endswith(x, ".md"), 
    readdir(joinpath(@__DIR__, "src", "examples"))
    )
)


DocMeta.setdocmeta!(
    GPLikelihoods,
    :DocTestSetup,
    :(GPLikelihoods, LinearAlgebra, Random);
    recursive=true,
)

makedocs(;
    modules=[GPLikelihoods],
    format=Documenter.HTML(),
    repo="https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl/blob/{commit}{path}#L{line}",
    sitename="GPLikelihoods.jl",
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => [
            generated_examples...
        ]
    ],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/GPLikelihoods.jl",
)
