# GPLikelihoods.jl

`GPLikelihoods.jl` provides a practical interface to connect non-conjugate likelihoods
with Gaussian Processes.
The API is very basic, every `AbstractLikelihood` object is functor taking a 
scalar/vector `f` and returns a `Distribution` from `Distributions.jl`. 

```@doctest
    f = 2.0
    GaussianLikelihood()(f) == Normal(2.0)
```

Since the parameter domain of a lot of distributions is more restricted to the real
domain, for example the parameter of a `Bernoulli` distribution, most `AbstractLikelihood`
contain a `invlink` field which will map the latent variable to the right domain.

[`Link`](@ref)s can be created using the constructor `Link(::Function)`.