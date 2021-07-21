@testset "link" begin
    x = rand()
    xs = rand(3)

    # Generic link
    f = sin
    l = Link(f)
    @test l(x) == f(x)

    # Log
    l = LogLink()
    @test l(x) == log(x)
    @test inv(l) == ExpLink()

    # Exp
    l = ExpLink()
    @test l(x) == exp(x)
    @test inv(l) == LogLink()
    
    # Sqrt
    l = SqrtLink()
    @test l(x) == sqrt(x)
    @test inv(l) == SquareLink()

    # Square
    l = SquareLink()
    @test l(x) == x^2
    @test inv(l) == SqrtLink()

    # Logit
    l = LogitLink()
    @test l(x) == logit(x)
    @test inv(l) isa LogisticLink

    # Logistic
    l = LogisticLink()
    @test l(x) == logistic(x)
    @test inv(l) isa LogitLink

    # Probit
    l = ProbitLink()
    @test l(x) == norminvcdf(x)
    @test inv(l) == NormalCDFLink()

    # NormalCDF
    l = NormalCDFLink()
    @test l(x) == normcdf(x)
    @test inv(l) == ProbitLink()

    # SoftMax
    l = SoftMaxLink()
    @test l(xs) == softmax(xs)

end
