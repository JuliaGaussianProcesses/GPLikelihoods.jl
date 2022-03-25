"""
    CategoricalLikelihood(l=BijectiveSimplexLink(softmax))

Categorical likelihood is to be used if we assume that the 
uncertainty associated with the data follows a [Categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution).

Assuming a distribution with `n` categories:

## `n-1` inputs (bijective link)

One can work with a bijective transformation by wrapping a link (like `softmax`)
into a [`BijectiveSimplexLink`](@ref) and only needs `n-1` inputs:
```math
    p(y|f_1, f_2, \\dots, f_{n-1}) = \\operatorname{Categorical}(y | l(f_1, f_2, \\dots, f_{n-1}, 0))
```
The default constructor is a bijective link around `softmax`.

## `n` inputs (non-bijective link)

One can also pass directly the inputs without concatenating a `0`:
```math
    p(y|f_1, f_2, \\dots, f_n) = \\operatorname{Categorical}(y | l(f_1, f_2, \\dots, f_n))
```
This variant is over-parametrized, as there are `n-1` independent parameters 
embedded in a `n` dimensional parameter space.
For more details, see the end of the section of this [Wikipedia link](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions)
where it corresponds to Variant 1 and 2.
"""
struct CategoricalLikelihood{Tl<:AbstractLink} <: AbstractLikelihood
    invlink::Tl
end

CategoricalLikelihood(l=BijectiveSimplexLink(softmax)) = CategoricalLikelihood(link(l))

function (l::CategoricalLikelihood)(f::AbstractVector{<:Real})
    return Categorical(l.invlink(f))
end

(l::CategoricalLikelihood)(fs::AbstractVector) = Product(l.(fs))
