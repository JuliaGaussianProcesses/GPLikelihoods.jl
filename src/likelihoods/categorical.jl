"""
    CategoricalLikelihood

Categorical likelihood is to be used if we assume that the 
uncertainity associated with the data follows a Categorical distribution.
```math
    p(y|f_1, f_2, \\dots, f_{n-1}) = Categorical(y | softmax(f_1, f_2, \\dots, f_{n-1}, 0))
```
On calling, this would return a Categorical distribution with `f_i` 
probability of `i` category.
"""
struct CategoricalLikelihood end

(l::CategoricalLikelihood)(f::AbstractVector{<:Real}) = Categorical(softmax(vcat(f, 0)))

(l::CategoricalLikelihood)(fs::AbstractVector) = Product(Categorical.(softmax.(vcat.(fs, 0))))
