"""
    inverse(f)

Return the inverse of function `f`.

This is an internal function to avoid type piracies such as `inv(::typeof(exp))`. At some
point `inv` might support standard Julia functions which would allow us to remove `inverse`.
"""
inverse(f)

inverse(::typeof(exp)) = log
inverse(::typeof(log)) = exp

inverse(::typeof(log1pexp)) = logexpm1
inverse(::typeof(logexpm1)) = log1pexp

inverse(::typeof(inv)) = inv

square(x) = x^2
inverse(::typeof(sqrt)) = square
inverse(::typeof(square)) = sqrt

inverse(::typeof(logit)) = logistic
inverse(::typeof(logistic)) = logit

inverse(::typeof(normcdf)) = norminvcdf
inverse(::typeof(norminvcdf)) = normcdf
