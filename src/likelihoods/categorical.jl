"""
    CategoricalLikelihood(l=softmax)

Categorical likelihood is to be used if we assume that the 
uncertainity associated with the data follows a Categorical distribution.
```math
    p(y|f_1, f_2, \\dots, f_{n-1}) = \\operatorname{Categorical}(y | l(f_1, f_2, \\dots, f_{n-1}, 0))
```
Given an `AbstractVector` ``[f_1, f_2, ..., f_{n-1}]``, returns a `Categorical` distribution,
with probabilities given by ``l(f_1, f_2, ..., f_{n-1}, 0)``.
"""
struct CategoricalLikelihood{Tl<:AbstractLink} <: AbstractLikelihood
    invlink::Tl
end

CategoricalLikelihood(l=softmax) = CategoricalLikelihood(Link(l))

(l::CategoricalLikelihood)(f::AbstractVector{<:Real}) = Categorical(l.invlink(vcat(f, 0)))

function (l::CategoricalLikelihood)(fs::AbstractVector)
    return Product(Categorical.(l.invlink.(vcat.(fs, 0))))
end
