
using SpecialFunctions :: logbeta
using IrrationalConstants :: logπ, 


"""
    StudentTLikelihood(σ²,ν)

Student's T likelihood with `σ²` scale and ν degrees of freedom . This is to be used if we assume that the
uncertainity associated with the data follows a Student's T distribution.

```math
    p(y|f) = \\operatorname{Student}(y | f, σ², ν)
```
"""

struct StudentTLikelihood{T<:Real, Tn :: Real} <: AbstractLikelihood
    σ²::Vector{T}
    ν::Vector{Tn}
end

function expected_loglikelihood( ::AnalyticExpectation,lik::StudentTLikelihood,q_f :: AbstractVector{<:Normal}, y :: AbstractVector{<:Real})
    f_μ = mean.(q_f)
    # Why?
    return sum(-logbeta(0.5,0.5*lik.ν) .- 0.5*logπ .- 0.5*log(lik.ν) .- log(lik.σ²) .- (0.5*(lik.ν+1))*log.(1 .+ ((y .- f_μ).^2 + var.(q_f)) / lik.σ²))
end


default_expectation_method(::StudentTLikelihood) = AnalyticExpectation()