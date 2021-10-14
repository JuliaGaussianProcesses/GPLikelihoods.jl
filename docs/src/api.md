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
HeteroscedasticGaussianLikelihood
PoissonLikelihood
```

## Links

```@docs
Link
ChainLink
```

The rest of the links `ExpLink`, `LogisticLink`, etc., are simple aliases for the
corresponding wrapped functions in a `Link`.
For example `ExpLink == Link{typeof(exp)}`.

We provide nonetheless docs for them:

```@docs
LogLink
ExpLink
InvLink
SqrtLink
SquareLink
LogitLink
LogisticLink
ProbitLink
NormalCDFLink
SoftMaxLink
```