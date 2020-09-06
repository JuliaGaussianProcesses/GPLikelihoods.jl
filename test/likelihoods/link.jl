@testset "link" begin
    using GPLikelihoods: AbstractLink, Link, LogisticLink
    f = sin
    l = Link(f)
    x = rand()
    @test l(x) == f(x)

    λ = 2.0
    l = LogisticLink(λ)
    @test l(x) == λ * logistic(x)
    l = LogisticLink()
    @test l(x) == logistic(x)
end
