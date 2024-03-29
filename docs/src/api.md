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

### Negative Binomial

```@docs
NegativeBinomialLikelihood
NBParamSuccess
NBParamFailure
NBParamI
NBParamII
NBParamPower
```

## Links

```@docs
Link
ChainLink
BijectiveSimplexLink
```

The rest of the links [`ExpLink`](@ref), [`LogisticLink`](@ref), etc.,
are aliases for the corresponding wrapped functions in a `Link`.
For example `ExpLink == Link{typeof(exp)}`.

When passing a [`Link`](@ref) to a likelihood object, this link 
corresponds to the transformation `p=link(f)` while, as mentioned in the
[Constrained parameters](@ref) section, the statistics literature usually uses
 the denomination [**inverse link or mean function**](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function) for it.

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

## Expectations

```@docs
expected_loglikelihood
```
