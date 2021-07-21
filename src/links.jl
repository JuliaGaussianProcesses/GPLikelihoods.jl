"""
    AbstractLink

Abstract type defining maps from R^n -> X.
They can be applied by calling `link(x)`.

A series of definitions are given in http://web.pdx.edu/~newsomj/mvclass/ho_link.pdf
"""
abstract type AbstractLink end

(f::AbstractLink)(x) = apply(f, x)

struct ChainLink{Tls} <: AbstractLink
    links::Tls
end

apply(l::ChainLink, x) = foldl((x, l) -> l(x), l.ls; init=x)

"""
    Link(f)

General construction for a link with a function `f`.
"""
struct Link{F} <: AbstractLink
    f::F
end

apply(l::Link, x) = l.f(x)

"""
    LogLink()

`log` link, f:ℝ⁺->ℝ . Its inverse is the [`ExpLink`](@ref).
"""
struct LogLink <: AbstractLink end

apply(::LogLink, x) = log(x)

Base.inv(::LogLink) = ExpLink()

"""
    ExpLink()

`exp` link, f:ℝ->ℝ⁺. Its inverse is the [`LogLink`](@ref).
"""
struct ExpLink <: AbstractLink end

apply(::ExpLink, x) = exp(x)

Base.inv(::ExpLink) = LogLink()

"""
    InvLink()

`inv` link, f:ℝ/{0}->ℝ/{0}. It is its own inverse.
"""
struct InvLink <: AbstractLink end

apply(::InvLink, x) = inv(x)

Base.inv(::InvLink) = InvLink()

"""
    SqrtLink()

`sqrt` link, f:ℝ⁺∪{0}->ℝ⁺∪{0}. Its inverse is the [`SquareLink`](@ref).
"""
struct SqrtLink <: AbstractLink end

apply(::SqrtLink, x) = sqrt(x)

Base.inv(::SqrtLink) = SquareLink()

"""
    SquareLink()

`^2` link, f:ℝ->ℝ⁺∪{0}. Its inverse is the [`SqrtLink`](@ref).
"""
struct SquareLink <: AbstractLink end

apply(::SquareLink, x) = x^2

Base.inv(::SquareLink) = SqrtLink()

"""
    LogitLink()

`log(x/(1-x))` link, f:[0,1]->ℝ. Its inverse is the [`LogisticLink`](@ref).
"""
struct LogitLink <: AbstractLink end

apply(::LogitLink, x) = logit(x)

Base.inv(::LogitLink) = LogisticLink()

"""
    LogisticLink()

`exp(x)/(1+exp(-x))` link. f:ℝ->[0,1]. Its inverse is the [`Logit`](@ref).
"""
struct LogisticLink <: AbstractLink end

apply(::LogisticLink, x) = logistic(x)

Base.inv(::LogisticLink) = LogitLink()

"""
    ProbitLink()

`ϕ⁻¹(y)` link, where `ϕ⁻¹` is the `invcdf` of a `Normal` distribution, f:[0,1]->ℝ.
Its inverse is the [`NormalCDFLink`](@ref).
"""
struct ProbitLink <: AbstractLink end

apply(::ProbitLink, x) = norminvcdf(x)

Base.inv(::ProbitLink) = NormalCDFLink()

"""
    NormalCDFLink()

`ϕ(y)` link, where `ϕ` is the `cdf` of a `Normal` distribution, f:ℝ->[0,1].
Its inverse is the [`ProbitLink`](@ref).
"""
struct NormalCDFLink <: AbstractLink end

apply(::NormalCDFLink, x) = normcdf(x)

Base.inv(::NormalCDFLink) = ProbitLink()

apply(::ChainLink{<:Tuple{LogLink,NormalCDFLink}}, x) = normlogcdf(x) # Specialisation for log + normal cdf

"""
    SoftMaxLink()

`softmax` link, i.e `f(x)ᵢ = exp(xᵢ)/∑ⱼexp(xⱼ)`.
f:ℝⁿ->Sⁿ, where Sⁿ is an n-simplex
It has no defined inverse
"""
struct SoftMaxLink <: AbstractLink end

apply(::SoftMaxLink, x::AbstractVector{<:Real}) = softmax(x)
