var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/#Likelihoods","page":"API","title":"Likelihoods","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"BernoulliLikelihood\nCategoricalLikelihood\nExponentialLikelihood\nGammaLikelihood\nGaussianLikelihood\nHeteroscedasticGaussianLikelihood\nPoissonLikelihood","category":"page"},{"location":"api/#GPLikelihoods.BernoulliLikelihood","page":"API","title":"GPLikelihoods.BernoulliLikelihood","text":"BernoulliLikelihood(l=logistic)\n\nBernoulli likelihood is to be used if we assume that the  uncertainity associated with the data follows a Bernoulli distribution. The link l needs to transform the input f to the domain [0, 1]\n\n    p(yf) = operatornameBernoulli(y  l(f))\n\nOn calling, this would return a Bernoulli distribution with l(f) probability of true.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.CategoricalLikelihood","page":"API","title":"GPLikelihoods.CategoricalLikelihood","text":"CategoricalLikelihood(l=BijectiveSimplexLink(softmax))\n\nCategorical likelihood is to be used if we assume that the  uncertainty associated with the data follows a Categorical distribution.\n\nAssuming a distribution with n categories:\n\nn-1 inputs (bijective link)\n\nOne can work with a bijective transformation by wrapping a link (like softmax) into a BijectiveSimplexLink and only needs n-1 inputs:\n\n    p(yf_1 f_2 dots f_n-1) = operatornameCategorical(y  l(f_1 f_2 dots f_n-1 0))\n\nThe default constructor is a bijective link around softmax.\n\nn inputs (non-bijective link)\n\nOne can also pass directly the inputs without concatenating a 0:\n\n    p(yf_1 f_2 dots f_n) = operatornameCategorical(y  l(f_1 f_2 dots f_n))\n\nThis variant is over-parametrized, as there are n-1 independent parameters  embedded in a n dimensional parameter space. For more details, see the end of the section of this Wikipedia link where it corresponds to Variant 1 and 2.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.ExponentialLikelihood","page":"API","title":"GPLikelihoods.ExponentialLikelihood","text":"ExponentialLikelihood(l=exp)\n\nExponential likelihood with scale given by l(f).\n\n    p(yf) = operatornameExponential(y  l(f))\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.GammaLikelihood","page":"API","title":"GPLikelihoods.GammaLikelihood","text":"GammaLikelihood(α::Real=1.0, l=exp)\n\nGamma likelihood with fixed shape α.\n\n    p(yf) = operatornameGamma(y  α l(f))\n\nOn calling, this returns a Gamma distribution with shape α and scale invlink(f).\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.GaussianLikelihood","page":"API","title":"GPLikelihoods.GaussianLikelihood","text":"GaussianLikelihood(σ²)\n\nGaussian likelihood with σ² variance. This is to be used if we assume that the uncertainity associated with the data follows a Gaussian distribution.\n\n    p(yf) = operatornameNormal(y  f σ²)\n\nOn calling, this would return a normal distribution with mean f and variance σ².\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.HeteroscedasticGaussianLikelihood","page":"API","title":"GPLikelihoods.HeteroscedasticGaussianLikelihood","text":"HeteroscedasticGaussianLikelihood(l=exp)\n\nHeteroscedastic Gaussian likelihood.  This is a Gaussian likelihood whose mean and variance are functions of latent processes.\n\n    p(yf g) = operatornameNormal(y  f sqrt(l(g)))\n\nOn calling, this would return a normal distribution with mean f and variance l(g). Where l is link going from R to R^+\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.PoissonLikelihood","page":"API","title":"GPLikelihoods.PoissonLikelihood","text":"PoissonLikelihood(l=exp)\n\nPoisson likelihood with rate defined as l(f).\n\n    p(yf) = operatornamePoisson(y  θ=l(f))\n\nThis is to be used if  we assume that the uncertainity associated with the data follows a Poisson distribution.\n\n\n\n\n\n","category":"type"},{"location":"api/#Negative-Binomial","page":"API","title":"Negative Binomial","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"NegativeBinomialLikelihood\nNBParamSuccess\nNBParamFailure\nNBParamI\nNBParamII\nNBParamPower","category":"page"},{"location":"api/#GPLikelihoods.NegativeBinomialLikelihood","page":"API","title":"GPLikelihoods.NegativeBinomialLikelihood","text":"NegativeBinomialLikelihood(param::NBParam, invlink::Union{Function,Link})\n\nThere are many possible parametrizations for the Negative Binomial likelihood. We follow the convention laid out in p.137 of [^Hilbe'11] and provide some common parametrizations. The NegativeBinomialLikelihood has a special structure; the first type parameter NBParam defines in what parametrization the latent function is used, and contains the other (scalar) parameters. NBParam itself has two subtypes:\n\nNBParamProb for parametrizations where f -> p, the probability of success of a Bernoulli event\nNBParamMean for parametrizations where f -> μ, the expected number of events\n\nNBParam predefined types\n\nNBParamProb types with p = invlink(f) the probability of success or failure\n\nNBParamSuccess: Here p = invlink(f) is the probability of success. This is the definition used in Distributions.jl.\nNBParamFailure: Here p = invlink(f) is the probability of a failure\n\nNBParamMean types with μ = invlink(f) the mean/expected number of events\n\nNBParamI: Mean is linked to f and variance is given by μ(1 + α)\nNBParamII: Mean is linked to f and variance is given by μ(1 + α * μ)\nNBParamPower: Mean is linked to f and variance is given by μ(1 + α * μ^ρ)\n\nTo create a new parametrization, you need to:\n\ncreate a new type struct MyNBParam{T} <: NBParam; myparams::T; end;\ndispatch (l::NegativeBinomialLikelihood{<:MyNBParam})(f::Real), which must return a NegativeBinomial from Distributions.jl.\n\nNegativeBinomial follows the parametrization of NBParamSuccess, i.e. the first argument is the number of successes and the second argument is the probability of success.\n\nExamples\n\njulia> NegativeBinomialLikelihood(NBParamSuccess(10), logistic)(2.0)\nNegativeBinomial{Float64}(r=10.0, p=0.8807970779778824)\njulia> NegativeBinomialLikelihood(NBParamFailure(10), logistic)(2.0)\nNegativeBinomial{Float64}(r=10.0, p=0.11920292202211757)\n\njulia> d = NegativeBinomialLikelihood(NBParamI(3.0), exp)(2.0)\nNegativeBinomial{Float64}(r=2.4630186996435506, p=0.25)\njulia> mean(d) ≈ exp(2.0)\ntrue\njulia> var(d) ≈ exp(2.0) * (1 + 3.0)\ntrue\n\n[^Hilbe'11]: Hilbe, Joseph M. Negative binomial regression. Cambridge University Press, 2011.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.NBParamSuccess","page":"API","title":"GPLikelihoods.NBParamSuccess","text":"NBParamSuccess(successes)\n\nNegative Binomial parametrization with successes the number of successes and invlink(f) the probability of success. This corresponds to the definition used by Distributions.jl.\n\n  p(ytextsuccesses p=textinvlink(f)) = fracGamma(y+textsuccesses)y Gamma(textsuccesses) p^textsuccesses (1 - p)^y\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.NBParamFailure","page":"API","title":"GPLikelihoods.NBParamFailure","text":"NBParamFailure(failures)\n\nNegative Binomial parametrization with failures the number of failures and invlink(f) the probability of success. This corresponds to the definition used by Wikipedia.\n\n  p(ytextfailures p=textinvlink(f)) = fracGamma(y+textfailures)y Gamma(textfailures) p^y (1 - p)^textfailures\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.NBParamI","page":"API","title":"GPLikelihoods.NBParamI","text":"NBParamI(α)\n\nNegative Binomial parametrization with mean μ = invlink(f) and variance v = μ(1 + α) where α > 0.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.NBParamII","page":"API","title":"GPLikelihoods.NBParamII","text":"NBParamII(α)\n\nNegative Binomial parametrization with mean μ = invlink(f) and variance v = μ(1 + α * μ) where α > 0.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.NBParamPower","page":"API","title":"GPLikelihoods.NBParamPower","text":"NBParamPower(α, ρ)\n\nNegative Binomial parametrization with mean μ = invlink(f) and variance v = μ(1 + α * μ^ρ) where α > 0 and ρ > 0.\n\n\n\n\n\n","category":"type"},{"location":"api/#Links","page":"API","title":"Links","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Link\nChainLink\nBijectiveSimplexLink","category":"page"},{"location":"api/#GPLikelihoods.Link","page":"API","title":"GPLikelihoods.Link","text":"Link(f)\n\nGeneral construction for a link with a function f.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.ChainLink","page":"API","title":"GPLikelihoods.ChainLink","text":"ChainLink(links)\n\nCreate a composed chain of different links.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.BijectiveSimplexLink","page":"API","title":"GPLikelihoods.BijectiveSimplexLink","text":"BijectiveSimplexLink(link)\n\nWrapper to preprocess the inputs by adding a 0 at the end before passing it to  the link link. This is a necessary step to work with simplices. For example with the SoftMaxLink, to obtain a n-simplex leading to n+1 categories for the CategoricalLikelihood, one needs to pass n+1 latent GP. However, by wrapping the link into a BijectiveSimplexLink, only n latent are needed. \n\n\n\n\n\n","category":"type"},{"location":"api/","page":"API","title":"API","text":"The rest of the links ExpLink, LogisticLink, etc., are aliases for the corresponding wrapped functions in a Link. For example ExpLink == Link{typeof(exp)}.","category":"page"},{"location":"api/","page":"API","title":"API","text":"When passing a Link to a likelihood object, this link  corresponds to the transformation p=link(f) while, as mentioned in the Constrained parameters section, the statistics literature usually uses  the denomination inverse link or mean function for it.","category":"page"},{"location":"api/","page":"API","title":"API","text":"LogLink\nExpLink\nInvLink\nSqrtLink\nSquareLink\nLogitLink\nLogisticLink\nProbitLink\nNormalCDFLink\nSoftMaxLink","category":"page"},{"location":"api/#GPLikelihoods.LogLink","page":"API","title":"GPLikelihoods.LogLink","text":"LogLink()\n\nlog link, f:ℝ⁺->ℝ . Its inverse is the ExpLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.ExpLink","page":"API","title":"GPLikelihoods.ExpLink","text":"ExpLink()\n\nexp link, f:ℝ->ℝ⁺. Its inverse is the LogLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.InvLink","page":"API","title":"GPLikelihoods.InvLink","text":"InvLink()\n\ninv link, f:ℝ/{0}->ℝ/{0}. It is its own inverse.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.SqrtLink","page":"API","title":"GPLikelihoods.SqrtLink","text":"SqrtLink()\n\nsqrt link, f:ℝ⁺∪{0}->ℝ⁺∪{0}. Its inverse is the SquareLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.SquareLink","page":"API","title":"GPLikelihoods.SquareLink","text":"SquareLink()\n\n^2 link, f:ℝ->ℝ⁺∪{0}. Its inverse is the SqrtLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.LogitLink","page":"API","title":"GPLikelihoods.LogitLink","text":"LogitLink()\n\nlog(x/(1-x)) link, f:[0,1]->ℝ. Its inverse is the LogisticLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.LogisticLink","page":"API","title":"GPLikelihoods.LogisticLink","text":"LogisticLink()\n\n1/(1+exp(-x)) link. f:ℝ->[0,1]. Its inverse is the LogitLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.ProbitLink","page":"API","title":"GPLikelihoods.ProbitLink","text":"ProbitLink()\n\nϕ⁻¹(y) link, where ϕ⁻¹ is the invcdf of a Normal distribution, f:[0,1]->ℝ. Its inverse is the NormalCDFLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.NormalCDFLink","page":"API","title":"GPLikelihoods.NormalCDFLink","text":"NormalCDFLink()\n\nϕ(y) link, where ϕ is the cdf of a Normal distribution, f:ℝ->[0,1]. Its inverse is the ProbitLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.SoftMaxLink","page":"API","title":"GPLikelihoods.SoftMaxLink","text":"SoftMaxLink()\n\nsoftmax link, i.e f(x)ᵢ = exp(xᵢ)/∑ⱼexp(xⱼ). f:ℝⁿ->Sⁿ⁻¹, where Sⁿ⁻¹ is an (n-1)-simplex It has no defined inverse\n\n\n\n\n\n","category":"type"},{"location":"api/#Expectations","page":"API","title":"Expectations","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"expected_loglikelihood","category":"page"},{"location":"api/#GPLikelihoods.expected_loglikelihood","page":"API","title":"GPLikelihoods.expected_loglikelihood","text":"expected_loglikelihood(\n    quadrature,\n    lik,\n    q_f::AbstractVector{<:Normal},\n    y::AbstractVector,\n)\n\nThis function computes the expected log likelihood:\n\n     q(f) log p(y  f) df\n\nwhere p(y | f) is the process likelihood. This is described by lik, which should be a callable that takes f as input and returns a Distribution over y that supports loglikelihood(lik(f), y).\n\nq(f) is an approximation to the latent function values f given by:\n\n    q(f) =  p(f  u) q(u) du\n\nwhere q(u) is the variational distribution over inducing points. The marginal distributions of q(f) are given by q_f.\n\nquadrature determines which method is used to calculate the expected log likelihood.\n\nExtended help\n\nq(f) is assumed to be an MvNormal distribution and p(y | f) is assumed to have independent marginals such that only the marginals of q(f) are required.\n\n\n\n\n\nexpected_loglikelihood(::DefaultExpectationMethod, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector)\n\nThe expected log likelihood, using the default quadrature method for the given likelihood. (The default quadrature method is defined by default_expectation_method(lik), and should be the closed form solution if it exists, but otherwise defaults to Gauss-Hermite quadrature.)\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = GPLikelihoods","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Distributions\nusing GPLikelihoods\nusing StatsFuns","category":"page"},{"location":"#GPLikelihoods","page":"Home","title":"GPLikelihoods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GPLikelihoods.jl provides a practical interface to connect Gaussian and non-conjugate likelihoods to Gaussian Processes. The API is very basic: Every AbstractLikelihood object is a functor taking a Real or an AbstractVector as an input and returning a  Distribution from Distributions.jl.","category":"page"},{"location":"#Single-latent-vs-multi-latent-likelihoods","page":"Home","title":"Single-latent vs multi-latent likelihoods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Most likelihoods, like the GaussianLikelihood, only require one latent Gaussian process. Passing a Real will therefore return a UnivariateDistribution, and passing an AbstractVector{<:Real} will return a multivariate product of distributions.","category":"page"},{"location":"","page":"Home","title":"Home","text":"f = 2.0;\nGaussianLikelihood()(f) == Normal(2.0, 1e-3)\nfs = [2.0, 3.0, 1.5];\nGaussianLikelihood()(fs) isa AbstractMvNormal","category":"page"},{"location":"","page":"Home","title":"Home","text":"Some likelihoods, like the CategoricalLikelihood, require multiple latent Gaussian processes, and an AbstractVector{<:Real} needs to be passed. To obtain a product of distributions an AbstractVector{<:AbstractVector{<:Real}} has to be passed (we recommend using ColVecs and RowVecs from KernelFunctions.jl if you need to transform an AbstractMatrix).","category":"page"},{"location":"","page":"Home","title":"Home","text":"fs = [2.0, 3.0, 4.5];\nCategoricalLikelihood()(fs) isa Categorical\nFs = [rand(3) for _ in 1:4];\nCategoricalLikelihood()(Fs) isa Product{<:Any,<:Categorical}","category":"page"},{"location":"#Constrained-parameters","page":"Home","title":"Constrained parameters","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The function values f of the latent Gaussian process live in the real domain mathbbR. For some likelihoods, the domain of the distribution parameter p that is modulated by the latent Gaussian process is constrained to some subset of mathbbR, e.g. only positive values or values in an interval.","category":"page"},{"location":"","page":"Home","title":"Home","text":"To connect these two domains, a transformation from f to p is required. For this, we provide the Link type, which can be passed to the likelihood constructors.  (Alternatively, functions can also directly be passed and will be wrapped in a Link.)","category":"page"},{"location":"","page":"Home","title":"Home","text":"We typically call this passed transformation the invlink. This comes from the statistics literature, where the \"link\" is defined as f = link(p), whereas here we need p = invlink(f).","category":"page"},{"location":"","page":"Home","title":"Home","text":"For more details about which likelihoods require a Link check out their docs.","category":"page"},{"location":"","page":"Home","title":"Home","text":"A classical example is the BernoulliLikelihood for classification, with the probability parameter p in 0 1. The default is to use a logistic transformation, but one could also use the inverse of the probit link:","category":"page"},{"location":"","page":"Home","title":"Home","text":"f = 2.0;\nBernoulliLikelihood()(f) == Bernoulli(logistic(f))\nBernoulliLikelihood(NormalCDFLink())(f) == Bernoulli(normcdf(f))","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that we passed the inverse of the probit function which is the normcdf function.","category":"page"}]
}