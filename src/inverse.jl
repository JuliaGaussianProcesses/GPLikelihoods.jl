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
