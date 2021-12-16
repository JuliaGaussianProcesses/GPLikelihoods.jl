```@meta
CurrentModule = GPLikelihoods
```
# GPLikelihoods

[`GPLikelihoods.jl`](https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl) provides a practical interface to connect non-conjugate likelihoods
with Gaussian Processes.
The API is very basic: Every `AbstractLikelihood` object is a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects-1)
that takes the output of a Gaussian process as input and returns a 
`Distribution` from [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl). 

```@repl
f = 2.0;
GaussianLikelihood()(f) == Normal(2.0)
```

### Constrained parameters

The domain of some distributions parameters can be different from 
``\mathbb{R}``, the real domain.
To solve this problem, we also provide the [`Link`](@ref) type, which can be
passed to the [`Likelihood`](@ref) constructors.
Alternatively, `function`s can also directly be passed and will be wrapped in a `Link`).
For more details about which likelihoods require a [`Link`](@ref) check out their docs.
We typically named this passed link as the `invlink`.
This comes from the  statistic literature, where the "link" is defined as `f = link(y)`.

A classical example is the [`BernoulliLikelihood`](@ref) for classification, with the probability parameter ``p \in \[0, 1\]``.
The default it to use a [`logistic`](https://en.wikipedia.org/wiki/Logistic_function) transformation, but one could also use the inverse of the [`probit`](https://en.wikipedia.org/wiki/Probit) link:

```@repl
f = 2.0;
BernoulliLikelihood()(f) == Bernoulli(logistic(f))
BernoulliLikelihood(NormalCDFLink()) == Bernoulli(normalcdf(f))
```
Note that we passed the `inverse` of the `probit` function which is the `normalcdf` function.