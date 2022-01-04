```@meta
CurrentModule = GPLikelihoods
```
# GPLikelihoods

[`GPLikelihoods.jl`](https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl) provides a practical interface to connect Gaussian and non-conjugate likelihoods
to Gaussian Processes.
The API is very basic: Every `AbstractLikelihood` object is a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects-1)
taking a `Real` or an `AbstractVector` as an input and returns a 
`Distribution` from [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl).

### Single-latent vs multi-latent likelihoods

Most likelihoods, like the [`GaussianLikelihood`](@ref), only require one latent Gaussian process.
Passing a `Real` will therefore return a [`UnivariateDistribution`](https://juliastats.org/Distributions.jl/latest/univariate/),
and passing an `AbstractVector{<:Real}` will return a [multivariate product of distributions](https://juliastats.org/Distributions.jl/latest/multivariate/#Product-distributions).
```@repl
f = 2.0;
GaussianLikelihood()(f) == Normal(2.0)
fs = [2.0, 3.0, 1.5]
GaussianLikelihood()(fs) == Product([Normal(2.0), Normal(3.0), Normal(1.5)])
```

Some likelihoods, like the [`CategoricalLikelihood`](@ref), requires multiple latent Gaussian processes,
and an `AbstractVector{<:Real}` needs to be passed.
To obtain a product of distributions an `AbstractVector{<:AbstractVector{<:Real}}` has to be passed (we recommend
using [`ColVecs` and `RowVecs` from KernelFunctions.jl](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/api/#Vector-Valued-Inputs)
if you need to transform an `AbstractMatrix`).
```@repl
fs = [2.0, 3.0, 4.5];
CategoricalLikelihood()(fs) isa Categorical
Fs = [rand(3) for _ in 1:4] 
CategoricalLikelihood()(Fs) isa Product{<:Any,<:Categorical}
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
