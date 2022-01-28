@testset "CategoricalLikelihood" begin
    Catbij = CategoricalLikelihood{true,SoftMaxLink}
    for args in ((), (softmax,), (SoftMaxLink(),)),
        kwargs in ((), (; bijective=true), (; bijective=Val(true)))

        @test CategoricalLikelihood(args...; kwargs...) isa Catbij
        if kwargs != (; bijective=true) # Inferred does not pass with Bool keyword
            @inferred CategoricalLikelihood(args...; kwargs...)
        end
    end

    Catnonbij = CategoricalLikelihood{false,SoftMaxLink}
    for args in ((), (softmax,), (SoftMaxLink(),)),
        kwargs in ((; bijective=false), (; bijective=Val(false)))

        @test CategoricalLikelihood(args...; kwargs...) isa Catnonbij
        if kwargs != (; bijective=false) # Inferred does not pass with Bool keyword
            @inferred CategoricalLikelihood(args...; kwargs...)
        end
    end

    OUT_DIM = 4
    lik_bijective = CategoricalLikelihood()
    test_interface(lik_bijective, Categorical, OUT_DIM)
    lik_nonbijective = CategoricalLikelihood(softmax; bijective=Val(false))
    test_interface(lik_nonbijective, Categorical, OUT_DIM)
end
