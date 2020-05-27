import .Flux.trainable

Flux.trainable(lik::GaussianLikelihood) = (lik.σ²,)
