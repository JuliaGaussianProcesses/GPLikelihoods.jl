@testset "link" begin
    x = rand()
    xs = rand(3)

    # Generic link
    f = sin
    l = GPLikelihoods.link(f)
    @test l == Link(f)
    @test l(x) == f(x)
    l = GPLikelihoods.link(ExpLink())
    @test l == ExpLink()

    # Log
    l = LogLink()
    @test l(x) == log(x)
    @test inv(l) == ExpLink()
    @test inv(inv(l)) == l

    # Exp
    l = ExpLink()
    @test l(x) == exp(x)
    @test inv(l) == LogLink()
    @test inv(inv(l)) == l

    # Sqrt
    l = SqrtLink()
    @test l(x) == sqrt(x)
    @test inv(l) == SquareLink()
    @test inv(inv(l)) == l

    # Square
    l = SquareLink()
    @test l(x) == x^2
    @test inv(l) == SqrtLink()
    @test inv(inv(l)) == l

    # Logit
    l = LogitLink()
    @test l(x) == logit(x)
    @test inv(l) isa LogisticLink
    @test inv(inv(l)) == l

    # Logistic
    l = LogisticLink()
    @test l(x) == logistic(x)
    @test inv(l) isa LogitLink
    @test inv(inv(l)) == l

    # Probit
    l = ProbitLink()
    @test l(x) == norminvcdf(x)
    @test inv(l) == NormalCDFLink()
    @test inv(inv(l)) == l

    # NormalCDF
    l = NormalCDFLink()
    @test l(x) == normcdf(x)
    @test inv(l) == ProbitLink()
    @test inv(inv(l)) == l

    # SoftMax
    l = SoftMaxLink()
    @test l(xs) == softmax(xs)
end
