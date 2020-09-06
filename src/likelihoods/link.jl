abstract type AbstractLink end

(f::AbstractLink)(x) = apply(f, x)

struct Link{F} <: AbstractLink
    f::F
end

apply(l::Link, x) = l.f(x)

struct LogisticLink{T<:Real} <: AbstractLink
    λ::Vector{T}
end

LogisticLink() = LogisticLink(1.0)

LogisticLink(λ::Real) = LogisticLink([λ])

@functor LogisticLink

apply(l::LogisticLink) = first(l.λ) * logistic(x)
