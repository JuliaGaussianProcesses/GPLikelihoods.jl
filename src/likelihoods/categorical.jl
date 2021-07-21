"""
    CategoricalLikelihood(l::AbstractLink=SoftMaxLink())

Categorical likelihood is to be used if we assume that the 
uncertainity associated with the data follows a Categorical distribution.
```math
    p(y|f_1, f_2, \\dots, f_{n-1}) = Categorical(y | l(f_1, f_2, \\dots, f_{n-1}, 0))
```
On calling, this would return a Categorical distribution where probabilities are 
given by `l(f_1, f_2, ..., f_{n-1}, 0)`, a vector of probabilities, which sums to 1
"""
struct CategoricalLikelihood{Tl<:AbstractLink}
    invlink::Tl
end

CategoricalLikelihood() = CategoricalLikelihood(SoftMaxLink())

(l::CategoricalLikelihood)(f::AbstractVector{<:Real}) = Categorical(l.invlink(vcat(f, 0)))

(l::CategoricalLikelihood)(fs::AbstractVector) = Product(Categorical.(l.invlink.(vcat.(fs, 0))))
