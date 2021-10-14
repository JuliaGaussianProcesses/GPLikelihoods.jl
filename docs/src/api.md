# API

```@index
```

## Likelihoods

```@docs
BernoulliLikelihood
CategoricalLikelihood
ExponentialLikelihood
GammaLikelihood
GaussianLikelihood
PoissonLikelihood
```

## Links

```@docs
Link
```

The rest of the links `ExpLink`, `LogisticLink`, etc., are simple aliases for the
corresponding wrapped functions in a `Link`.
For example `ExpLink == Link{::typeof(exp)}`.
