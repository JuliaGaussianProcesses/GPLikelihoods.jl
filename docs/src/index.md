# GPLikelihoods.jl

`GPLikelihoods.jl` provides a practical interface to connect non-conjugate likelihoods
with Gaussian Processes.
The API is very basic: Every `AbstractLikelihood` object is a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects-1) that takes the output of a Gaussian process as input and returns a `Distribution` from [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl). 

```@repl
f = 2.0;
GaussianLikelihood()(f) == Normal(2.0)
```

Since the parameter domain of a lot of distributions is more restricted to the real
domain, for example the parameter of a `Bernoulli` distribution, most `AbstractLikelihood`
contain a `invlink` field which will map the latent variable to the right domain.

```@repl
f = 2.0;
BernoulliLikelihood(logistic)(f) == Bernoulli(logistic(2.0))
```

[`Link`](@ref)s can be created using the constructor `Link(l)` where `l` is a function or a functor.