struct SimplexVariant end

struct CurvedVariant end

"""
    CategoricalLikelihood(l=softmax, variant=SimplexVariant())

Categorical likelihood is to be used if we assume that the 
uncertainty associated with the data follows a [Categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution) with `n` categories.
```math
    p(y|f_1, f_2, \\dots, f_{n-1}) = \\operatorname{Categorical}(y | l(f_1, f_2, \\dots, f_{n-1}, 0))
    p(y|f_1, f_2, \\dots, f_n) = \\operatorname{Categorical}(y | l(f_1, f_2, \\dots, f_n))
```
Two variants are possible:
## `SimplexVariant`
Given an `AbstractVector` ``[f_1, f_2, ..., f_{n-1}]``, returns a `Categorical` distribution,
with probabilities given by ``l(f_1, f_2, ..., f_{n-1}, 0)``.
It is used by default.

## `CurvedVariant`
Given an `AbstractVector` ``[f_1, f_2, ..., f_{n}]``, returns a `Categorical` distribution,
with probabilities given by ``l(f_1, f_2, ..., f_{n})``.
The "Curved" comes from the fact that such a distribution is "curved exponential family"
as there are `n-1` independent parameters embedded in a `n` dimensional parameter space.
A lot of standard results for exponential families do not apply.
For more explanations, see this [Wikipedia link](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions) at the end of the section.
Use with caution!
"""
struct CategoricalLikelihood{Tv,Tl<:AbstractLink} <: AbstractLikelihood
    invlink::Tl
    CategoricalLikelihood{Tv}(invlink::Tl) where {Tv,Tl} = new{Tv,Tl}(invlink)
end

CategoricalLikelihood(l=softmax, variant=SimplexVariant()) = CategoricalLikelihood{typeof(variant)}(Link(l))

(l::CategoricalLikelihood{<:AbstractLink,SimplexVariant})(f::AbstractVector{<:Real}) = Categorical(l.invlink(vcat(f, 0)))
(l::CategoricalLikelihood{<:AbstractLink,CurvedVariant})(f::AbstractVector{<:Real}) = Categorical(l.invlink(f))

function (l::CategoricalLikelihood{<:AbstractLink,SimplexVariant})(fs::AbstractVector)
    return Product(Categorical.(l.invlink.(vcat.(fs, 0))))
end
function (l::CategoricalLikelihood{<:AbstractLink,CurvedVariant})(fs::AbstractVector)
    return Product(Categorical.(l.invlink.(fs)))
end