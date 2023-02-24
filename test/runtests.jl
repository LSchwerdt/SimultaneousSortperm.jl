using SimultaneousSortperm
using Test
using Random

Random.seed!(0xdeadbeef)

@testset "SimultaneousSortperm.jl" begin
    for n in [(0:31)..., 100, 999, 1000, 1001]
        for T in [UInt16, Int, Float64], rev in [false, true], lt in [isless, >]
            for order in [Base.Order.Forward, Base.Order.Reverse], by in [identity, x->x÷100]

                skiptest = VERSION < v"1.9" && T != Int && by != identity && n > 20
                # broken depending on rng -> infeasible to list all combinations

                v = rand(T,n)
                pref = sortperm(v, lt=lt, by=by, rev=rev, order=order)
                vref = sort(v, lt=lt, by=by, rev=rev, order=order)

                p = ssortperm(v, lt=lt, by=by, rev=rev, order=order)
                @test p == pref
                p == pref || println((T,rev,lt,order,by))

                p .= 0
                ssortperm!(p, v, lt=lt, by=by, rev=rev, order=order)
                @test p == pref
                v2 = copy(v)
                p = ssortperm!(v2, lt=lt, by=by, rev=rev, order=order)
                @test p == pref
                if VERSION >= v"1.7"
                    @test v2 == vref skip=skiptest
                elseif !skiptest
                    @test v2 == vref
                end

                v2 = copy(v)
                p .= 0
                ssortperm!!(p, v2, lt=lt, by=by, rev=rev, order=order)
                @test p == pref
                if VERSION >= v"1.7"
                    @test v2 == vref skip=skiptest
                elseif !skiptest
                    @test v2 == vref
                end
            end
        end
    end
end
