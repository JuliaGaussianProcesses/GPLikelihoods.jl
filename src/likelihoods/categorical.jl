
"""
    CategoricalLikelihood

Categorical likelihood is to be used if we assume that the 
uncertainity associated with the data follows a Categorical distribution.
```math
    p(y|f_1, f_2, \\dots, f_n) = Categorical(y | f_1, f_2, \\dots, f_n)
```
On calling, this would return a Categorical distribution with `f` probability of `true`.
"""
struct CategoricalLikelihood end

@functor CategoricalLikelihood

(l::CategoricalLikelihood)(f::AbstractVector{<:Real}) = Categorical(softmax(f))

(l::CategoricalLikelihood)(fs::AbstractVector) = Product(Categorical.(softmax.(fs)))
