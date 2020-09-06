"""
    PoissonLikelihood()

Poisson likelihood with rate as exponential of samples from GP `f`. This is to be used if
we assume that the uncertainity associated with the data follows a Poisson distribution.
"""
struct PoissonLikelihood{L<:AbstractLink}
    link::L
end

PoissonLikelihood() = PoissonLikelihood(Link(exp))

PoissonLikelihood(f::Function)  = PoissonLikelihood(Link(f))

@functor PoissonLikelihood

(l::PoissonLikelihood)(f::Real) = Poisson(l.link(f))

(l::PoissonLikelihood)(fs::AbstractVector{<:Real}) = Product(Poisson.(l.link.(fs)))
