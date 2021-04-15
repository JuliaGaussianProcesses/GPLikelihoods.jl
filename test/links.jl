@testset "link" begin
    using GPLikelihoods: AbstractLink, Link, LogisticLink
    f = sin
    l = Link(f)
    x = rand()
    @test l(x) == f(x)

    l = LogisticLink()
    @test l(x) == logistic(x)
    @test inv(l) isa LogitLink
    l = LogitLink()
    @test l(x) == logit(x)
    @test inv(l) isa LogisticLink
end
