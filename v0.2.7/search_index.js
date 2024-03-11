var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/#Likelihoods","page":"API","title":"Likelihoods","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"BernoulliLikelihood\nCategoricalLikelihood\nExponentialLikelihood\nGammaLikelihood\nGaussianLikelihood\nHeteroscedasticGaussianLikelihood\nPoissonLikelihood","category":"page"},{"location":"api/#GPLikelihoods.BernoulliLikelihood","page":"API","title":"GPLikelihoods.BernoulliLikelihood","text":"BernoulliLikelihood(l=logistic)\n\nBernoulli likelihood is to be used if we assume that the  uncertainity associated with the data follows a Bernoulli distribution. The link l needs to transform the input f to the domain [0, 1]\n\n    p(yf) = operatornameBernoulli(y  l(f))\n\nOn calling, this would return a Bernoulli distribution with l(f) probability of true.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.CategoricalLikelihood","page":"API","title":"GPLikelihoods.CategoricalLikelihood","text":"CategoricalLikelihood(l=softmax)\n\nCategorical likelihood is to be used if we assume that the  uncertainity associated with the data follows a Categorical distribution.\n\n    p(yf_1 f_2 dots f_n-1) = operatornameCategorical(y  l(f_1 f_2 dots f_n-1 0))\n\nGiven an AbstractVector f_1 f_2  f_n-1, returns a Categorical distribution, with probabilities given by l(f_1 f_2  f_n-1 0).\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.ExponentialLikelihood","page":"API","title":"GPLikelihoods.ExponentialLikelihood","text":"ExponentialLikelihood(l=exp)\n\nExponential likelihood with scale given by l(f).\n\n    p(yf) = operatornameExponential(y  l(f))\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.GammaLikelihood","page":"API","title":"GPLikelihoods.GammaLikelihood","text":"GammaLikelihood(α::Real=1.0, l=exp)\n\nGamma likelihood with fixed shape α.\n\n    p(yf) = operatornameGamma(y  α l(f))\n\nOn calling, this would return a gamma distribution with shape α and scale l(f).\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.GaussianLikelihood","page":"API","title":"GPLikelihoods.GaussianLikelihood","text":"GaussianLikelihood(σ²)\n\nGaussian likelihood with σ² variance. This is to be used if we assume that the uncertainity associated with the data follows a Gaussian distribution.\n\n    p(yf) = operatornameNormal(y  f σ²)\n\nOn calling, this would return a normal distribution with mean f and variance σ².\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.HeteroscedasticGaussianLikelihood","page":"API","title":"GPLikelihoods.HeteroscedasticGaussianLikelihood","text":"HeteroscedasticGaussianLikelihood(l=exp)\n\nHeteroscedastic Gaussian likelihood.  This is a Gaussian likelihood whose mean and variance are functions of latent processes.\n\n    p(yf g) = operatornameNormal(y  f sqrt(l(g)))\n\nOn calling, this would return a normal distribution with mean f and variance l(g). Where l is link going from R to R^+\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.PoissonLikelihood","page":"API","title":"GPLikelihoods.PoissonLikelihood","text":"PoissonLikelihood(l=exp)\n\nPoisson likelihood with rate defined as l(f).\n\n    p(yf) = operatornamePoisson(y  θ=l(f))\n\nThis is to be used if  we assume that the uncertainity associated with the data follows a Poisson distribution.\n\n\n\n\n\n","category":"type"},{"location":"api/#Links","page":"API","title":"Links","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Link\nChainLink","category":"page"},{"location":"api/#GPLikelihoods.Link","page":"API","title":"GPLikelihoods.Link","text":"Link(f)\n\nGeneral construction for a link with a function f.\n\n\n\n\n\n","category":"type"},{"location":"api/","page":"API","title":"API","text":"The rest of the links ExpLink, LogisticLink, etc., are aliases for the corresponding wrapped functions in a Link. For example ExpLink == Link{typeof(exp)}.","category":"page"},{"location":"api/","page":"API","title":"API","text":"When passing a Link to an AbstractLikelihood, this link  corresponds to the transformation p=link(f) while, as mentioned in the Constrained parameters section, the statistics literature usually use  the denomination inverse link or mean function for it.","category":"page"},{"location":"api/","page":"API","title":"API","text":"LogLink\nExpLink\nInvLink\nSqrtLink\nSquareLink\nLogitLink\nLogisticLink\nProbitLink\nNormalCDFLink\nSoftMaxLink","category":"page"},{"location":"api/#GPLikelihoods.LogLink","page":"API","title":"GPLikelihoods.LogLink","text":"LogLink()\n\nlog link, f:ℝ⁺->ℝ . Its inverse is the ExpLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.ExpLink","page":"API","title":"GPLikelihoods.ExpLink","text":"ExpLink()\n\nexp link, f:ℝ->ℝ⁺. Its inverse is the LogLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.InvLink","page":"API","title":"GPLikelihoods.InvLink","text":"InvLink()\n\ninv link, f:ℝ/{0}->ℝ/{0}. It is its own inverse.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.SqrtLink","page":"API","title":"GPLikelihoods.SqrtLink","text":"SqrtLink()\n\nsqrt link, f:ℝ⁺∪{0}->ℝ⁺∪{0}. Its inverse is the SquareLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.SquareLink","page":"API","title":"GPLikelihoods.SquareLink","text":"SquareLink()\n\n^2 link, f:ℝ->ℝ⁺∪{0}. Its inverse is the SqrtLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.LogitLink","page":"API","title":"GPLikelihoods.LogitLink","text":"LogitLink()\n\nlog(x/(1-x)) link, f:[0,1]->ℝ. Its inverse is the LogisticLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.LogisticLink","page":"API","title":"GPLikelihoods.LogisticLink","text":"LogisticLink()\n\n1/(1+exp(-x)) link. f:ℝ->[0,1]. Its inverse is the LogitLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.ProbitLink","page":"API","title":"GPLikelihoods.ProbitLink","text":"ProbitLink()\n\nϕ⁻¹(y) link, where ϕ⁻¹ is the invcdf of a Normal distribution, f:[0,1]->ℝ. Its inverse is the NormalCDFLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.NormalCDFLink","page":"API","title":"GPLikelihoods.NormalCDFLink","text":"NormalCDFLink()\n\nϕ(y) link, where ϕ is the cdf of a Normal distribution, f:ℝ->[0,1]. Its inverse is the ProbitLink.\n\n\n\n\n\n","category":"type"},{"location":"api/#GPLikelihoods.SoftMaxLink","page":"API","title":"GPLikelihoods.SoftMaxLink","text":"SoftMaxLink()\n\nsoftmax link, i.e f(x)ᵢ = exp(xᵢ)/∑ⱼexp(xⱼ). f:ℝⁿ->Sⁿ⁻¹, where Sⁿ⁻¹ is an (n-1)-simplex It has no defined inverse\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = GPLikelihoods","category":"page"},{"location":"#GPLikelihoods","page":"Home","title":"GPLikelihoods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GPLikelihoods.jl provides a practical interface to connect Gaussian and non-conjugate likelihoods to Gaussian Processes. The API is very basic: Every AbstractLikelihood object is a functor taking a Real or an AbstractVector as an input and returns a  Distribution from Distributions.jl.","category":"page"},{"location":"#Single-latent-vs-multi-latent-likelihoods","page":"Home","title":"Single-latent vs multi-latent likelihoods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Most likelihoods, like the GaussianLikelihood, only require one latent Gaussian process. Passing a Real will therefore return a UnivariateDistribution, and passing an AbstractVector{<:Real} will return a multivariate product of distributions.","category":"page"},{"location":"","page":"Home","title":"Home","text":"f = 2.0;\nGaussianLikelihood()(f) == Normal(2.0)\nfs = [2.0, 3.0, 1.5]\nGaussianLikelihood()(fs) == Product([Normal(2.0), Normal(3.0), Normal(1.5)])","category":"page"},{"location":"","page":"Home","title":"Home","text":"Some likelihoods, like the CategoricalLikelihood, requires multiple latent Gaussian processes, and an AbstractVector{<:Real} needs to be passed. To obtain a product of distributions an AbstractVector{<:AbstractVector{<:Real}} has to be passed (we recommend using ColVecs and RowVecs from KernelFunctions.jl if you need to transform an AbstractMatrix).","category":"page"},{"location":"","page":"Home","title":"Home","text":"fs = [2.0, 3.0, 4.5];\nCategoricalLikelihood()(fs) isa Categorical\nFs = [rand(3) for _ in 1:4] \nCategoricalLikelihood()(Fs) isa Product{<:Any,<:Categorical}","category":"page"},{"location":"#Constrained-parameters","page":"Home","title":"Constrained parameters","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The domain of some distributions parameters can be different from  mathbbR, the real domain. To solve this problem, we also provide the Link type, which can be passed to the Likelihood constructors. Alternatively, functions can also directly be passed and will be wrapped in a Link). For more details about which likelihoods require a Link check out their docs. We typically named this passed link as the invlink. This comes from the  statistic literature, where the \"link\" is defined as f = link(y).","category":"page"},{"location":"","page":"Home","title":"Home","text":"A classical example is the BernoulliLikelihood for classification, with the probability parameter p in 0 1. The default it to use a logistic transformation, but one could also use the inverse of the probit link:","category":"page"},{"location":"","page":"Home","title":"Home","text":"f = 2.0;\nBernoulliLikelihood()(f) == Bernoulli(logistic(f))\nBernoulliLikelihood(NormalCDFLink()) == Bernoulli(normalcdf(f))","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that we passed the inverse of the probit function which is the normalcdf function.","category":"page"}]
}
