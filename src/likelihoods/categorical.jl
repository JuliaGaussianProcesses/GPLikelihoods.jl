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
struct CategoricalLikelihood{Tl<:AbtractLink}
    invlink::Tl
end

CategoricalLikelihood(invlink::T=SoftMaxLink()) where{T<:AbstractLink} = CategoricalLikelihood{T}(invlink)

(l::CategoricalLikelihood)(f::AbstractVector{<:Real}) = Categorical(l.invlink(vcat(f, 0)))

(l::CategoricalLikelihood)(fs::AbstractVector) = Product(Categorical.(l.invlink.(vcat.(fs, 0))))
