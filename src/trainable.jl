import .Flux.trainable

Flux.trainable(::Likelihood) = ()

Flux.trainable(lik::GaussianLikelihood) = (lik.σ²,)
