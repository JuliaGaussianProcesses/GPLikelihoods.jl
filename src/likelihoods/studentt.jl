
using SpecialFunctions :: logbeta
using IrrationalConstants :: logπ, 

struct StudentTLikelihood{T<:Real, Tn :: Real} <: AbstractLikelihood
    σ²::Vector{T}
    ν::Vector{Tn}
end

function expected_loglikelihood( ::AnalyticExpectation,lik::StudentTLikelihood,q_f :: AbstractVector{<:Normal}, y :: AbstractVector{<:Real})
    f_μ = mean.(q_f)
    # Why?
    return sum(-logbeta(0.5,0.5*lik.ν) .- 0.5*logπ .- 0.5*log(lik.ν) .- log(lik.σ²) .- (0.5*(lik.ν+1))*log.(1 .+ ((y .- f_μ).^2 + var.(q_f)) / lik.σ²) )
end