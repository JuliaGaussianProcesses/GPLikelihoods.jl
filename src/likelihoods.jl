
"""
    gaussian(fx::LatentGP, y)

Returns gaussian log-likelihood of output `y` w.r.t to the given `LatentGP`.  
"""
function gaussian(fx::LatentGP, y)
    return Distributions.logpdf(MvNormal(mean(fx), cov(fx) + fx.σ² * I), y)
end

