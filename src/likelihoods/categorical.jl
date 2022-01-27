"""
    CategoricalLikelihood(l=softmax, bijective=true)::CategoricalLikelihood{bijective,typeof(Link(k))}

Categorical likelihood is to be used if we assume that the 
uncertainty associated with the data follows a [Categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution) with `n` categories.
```math
    p(y|f_1, f_2, \\dots, f_{n-1}) = \\operatorname{Categorical}(y | l(f_1, f_2, \\dots, f_{n-1}, 0))
    p(y|f_1, f_2, \\dots, f_n) = \\operatorname{Categorical}(y | l(f_1, f_2, \\dots, f_n))
```
Two variants are possible:
## `bijective=true`
Given an `AbstractVector` ``[f_1, f_2, ..., f_{n-1}]``, returns a `Categorical` distribution,
with probabilities given by ``l(f_1, f_2, ..., f_{n-1}, 0)``.
It is used by default.

## `bijective=false`
Given an `AbstractVector` ``[f_1, f_2, ..., f_{n}]``, returns a `Categorical` distribution,
with probabilities given by ``l(f_1, f_2, ..., f_{n})``.
This variant is over-parametrized, as there are `n-1` independent parameters 
embedded in a `n` dimensional parameter space.
For more details, see the end of the section of this [Wikipedia link](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions)
where it corresponds to Variant 1 and 2.
"""
struct CategoricalLikelihood{Tb,Tl<:AbstractLink} <: AbstractLikelihood
    invlink::Tl
    CategoricalLikelihood{Tb}(invlink::Tl) where {Tb,Tl} = new{Tb,Tl}(invlink)
end

function CategoricalLikelihood(l=softmax, bijective=true)
    return CategoricalLikelihood{bijective}(Link(l))
end

CategoricalLikelihood(l::AbstractLink, bijective=true) = CategoricalLikelihood{bijective}(l)

function (l::CategoricalLikelihood{true})(f::AbstractVector{<:Real})
    return Categorical(l.invlink(vcat(f, 0)))
end
function (l::CategoricalLikelihood{false})(f::AbstractVector{<:Real})
    return Categorical(l.invlink(f))
end

function (l::CategoricalLikelihood{true})(fs::AbstractVector)
    return Product(Categorical.(l.invlink.(vcat.(fs, 0))))
end
function (l::CategoricalLikelihood{false})(fs::AbstractVector)
    return Product(Categorical.(l.invlink.(fs)))
end
