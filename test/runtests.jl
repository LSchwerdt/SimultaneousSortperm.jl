using SimultaneousSortperm
using Test
using Random
using OffsetArrays

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

@testset "OffsetArrays" begin
    for n in [(0:31)..., 100, 999, 1000, 1001]
        for offset in [-n , -1, 0, 1, n]
            v = OffsetArray(rand(Int,n), (1:n).+offset) 
            pref = sortperm(v)
            vref = sort(v)

            p = ssortperm(v)
            @test p == pref

            v2 = copy(v)
            p .= 0
            ssortperm!!(p, v2)
            @test p == pref
            @test v2 == vref
        end
    end
end

randnans(n) = reinterpret(Float64,[rand(UInt64)|0x7ff8000000000000 for i=1:n])

function randn_with_nans(n,p)
    v = randn(n)
    x = findall(rand(n).<p)
    v[x] = randnans(length(x))
    return v
end

@testset "rand_with_NaNs and negative Floats" begin
    for n in [(0:31)..., 100, 999, 1000, 1001]
        v = randn_with_nans(n,0.1)
        vo = OffsetArray(rand(Int,n), (1:n).+100)
        for order in [Base.Order.Forward, Base.Order.Reverse]

            pref = sortperm(v, order=order)
            vref = sort(v, order=order)

            p = ssortperm(v, order=order)
            @test p == pref

            v2 = copy(v)
            p .= 0
            ssortperm!!(p, v2, order=order)
            @test p == pref
            @test reinterpret(UInt64,v2) == reinterpret(UInt64,vref)

            # offset
            pref = sortperm(vo, order=order)
            vref = sort(vo, order=order)

            p = ssortperm(vo, order=order)
            @test p == pref

            v2o = copy(vo)
            p .= 0
            ssortperm!!(p, v2o, order=order)
            @test p == pref
            @test v2o == vref
        end
    end
end;

@testset "missing" begin
    for n in [(1:31)..., 100, 999, 1000, 1001]
        v = [rand(1:100) < 50 ? missing : randn_with_nans(1,0.1)[1] for _ in 1:n]
        vo = OffsetArray(rand(Int,n), (1:n).+100)
        for order in [Base.Order.Forward, Base.Order.Reverse]
            pref = sortperm(v, order=order)
            vref = sort(v, order=order)

            p = ssortperm(v, order=order)
            @test p == pref

            v2 = copy(v)
            p .= 0
            ssortperm!!(p, v2, order=order)
            @test p == pref
            im_v2 = ismissing.(v2)
            im_vref = ismissing.(vref)
            @test im_v2 == im_vref
            if any(.!im_vref) > 0
                @test reinterpret(UInt64,Float64.(v2[.!im_v2])) == reinterpret(UInt64,Float64.(vref[.!im_vref]))
            end

            # offset
            pref = sortperm(vo, order=order)
            vref = sort(vo, order=order)

            p = ssortperm(vo, order=order)
            @test p == pref

            v2o = copy(vo)
            p .= 0
            ssortperm!!(p, v2o, order=order)
            @test p == pref
            @test v2o == vref
        end
    end
end;
