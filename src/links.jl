"""
    AbstractLink

Abstract type defining maps from R^n -> X.
They can be applied by calling `link(x)`.

A series of definitions are given in http://web.pdx.edu/~newsomj/mvclass/ho_link.pdf
"""
abstract type AbstractLink end

struct ChainLink{Tls} <: AbstractLink
    links::Tls
end

(l::ChainLink)(x) = foldl((x, l) -> l(x), l.ls; init=x)

"""
    Link(f)

General construction for a link with a function `f`.
"""
struct Link{F} <: AbstractLink
    f::F
end

(l::Link)(x) = l.f(x)

Base.inv(l::Link) = Link(InverseFunctions.inverse(l.f))

# alias
const LogLink = Link{typeof(log)}
const ExpLink = Link{typeof(exp)}

const InvLink = Link{typeof(inv)}

const SqrtLink = Link{typeof(sqrt)}
const SquareLink = Link{typeof(InverseFunctions.square)}

const LogitLink = Link{typeof(logit)}
const LogisticLink = Link{typeof(logistic)}

const ProbitLink = Link{typeof(norminvcdf)}
const NormalCDFLink = Link{typeof(normcdf)}

const SoftMaxLink = Link{typeof(softmax)}

"""
    LogLink()

`log` link, f:ℝ⁺->ℝ . Its inverse is the [`ExpLink`](@ref).
"""
LogLink() = Link(log)

"""
    ExpLink()

`exp` link, f:ℝ->ℝ⁺. Its inverse is the [`LogLink`](@ref).
"""
ExpLink() = Link(exp)

"""
    InvLink()

`inv` link, f:ℝ/{0}->ℝ/{0}. It is its own inverse.
"""
InvLink() = Link(inv)

"""
    SqrtLink()

`sqrt` link, f:ℝ⁺∪{0}->ℝ⁺∪{0}. Its inverse is the [`SquareLink`](@ref).
"""
SqrtLink() = Link(sqrt)

"""
    SquareLink()

`^2` link, f:ℝ->ℝ⁺∪{0}. Its inverse is the [`SqrtLink`](@ref).
"""
SquareLink() = Link(InverseFunctions.square)

"""
    LogitLink()

`log(x/(1-x))` link, f:[0,1]->ℝ. Its inverse is the [`LogisticLink`](@ref).
"""
LogitLink() = Link(logit)

"""
    LogisticLink()

`exp(x)/(1+exp(-x))` link. f:ℝ->[0,1]. Its inverse is the [`LogitLink`](@ref).
"""
LogisticLink() = Link(logistic)

"""
    ProbitLink()

`ϕ⁻¹(y)` link, where `ϕ⁻¹` is the `invcdf` of a `Normal` distribution, f:[0,1]->ℝ.
Its inverse is the [`NormalCDFLink`](@ref).
"""
ProbitLink() = Link(norminvcdf)

"""
    NormalCDFLink()

`ϕ(y)` link, where `ϕ` is the `cdf` of a `Normal` distribution, f:ℝ->[0,1].
Its inverse is the [`ProbitLink`](@ref).
"""
NormalCDFLink() = Link(normcdf)

(::ChainLink{<:Tuple{LogLink,NormalCDFLink}})(x) = normlogcdf(x) # Specialisation for log + normal cdf

"""
    SoftMaxLink()

`softmax` link, i.e `f(x)ᵢ = exp(xᵢ)/∑ⱼexp(xⱼ)`.
f:ℝⁿ->Sⁿ⁻¹, where Sⁿ⁻¹ is an [(n-1)-simplex](https://en.wikipedia.org/wiki/Simplex)
It has no defined inverse
"""
SoftMaxLink() = Link(softmax)
