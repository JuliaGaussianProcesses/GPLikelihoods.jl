"""
    CategoricalLikelihood(l=softmax; bijective::Union{Bool,Val{false},Val{true}}=Val(true))

Categorical likelihood is to be used if we assume that the 
uncertainty associated with the data follows a [Categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution) with `n` categories.
```math
    p(y|f_1, f_2, \\dots, f_{n-1}) = \\operatorname{Categorical}(y | l(f_1, f_2, \\dots, f_{n-1}, 0))
    p(y|f_1, f_2, \\dots, f_n) = \\operatorname{Categorical}(y | l(f_1, f_2, \\dots, f_n))
```
Two variants are possible (you can use a `Bool` or `Val{true}`/`Val{false}` but
we recommend the latter for type stability).
## `bijective=true/Val{true}`
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
struct CategoricalLikelihood{Tl<:AbstractLink} <: AbstractLikelihood
    invlink::Tl
end

function CategoricalLikelihood(
    l=softmax; bijective::Union{Bool,Val{true},Val{false}}=Val(true)
)
    return CategoricalLikelihood(make_bijective(link(l), bijective))
end

function (l::CategoricalLikelihood)(f::AbstractVector{<:Real})
    return Categorical(l.invlink(f))
end

function (l::CategoricalLikelihood)(fs::AbstractVector)
    return Product(Categorical.(l.invlink.(fs)))
end
