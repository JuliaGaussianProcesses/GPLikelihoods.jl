import .Flux.trainable

Flux.trainable(lik::GaussianLikelihood) = (lik.σ²,)

Flux.trainable(lik::PoissonLikelihood) = (lik.λ,)
