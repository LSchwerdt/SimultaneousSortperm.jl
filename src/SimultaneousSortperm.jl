__precompile__()

module SimultaneousSortperm

include("pdqsort.jl")

using StructArrays
using StaticArrays

using Base.Order
using Base: copymutable

export ssortperm!!, ssortperm!, ssortperm

function sort_equal_subarrays!(v, vs, o::Ordering, offsets_l, offsets_r)
    i = firstindex(v)
    while i < lastindex(v)
        if !lt(o, v[i], v[i+1]) 
            first_equal = i
            while i < lastindex(v) && !lt(o, v[i], v[i+1])
                i += 1
            end
            last_equal = i
            lo = first_equal
            hi = last_equal
            pdqsort_loop!(vs, lo, hi, BranchlessPatternDefeatingQuicksortAlg(), Base.Order.By(x->x[2]), log2i(hi + 1 - lo), offsets_l, offsets_r)
        end
        i += 1
    end
end

function allocate_index_vector(v)
    ax = axes(v, 1)
    similar(Vector{eltype(ax)}, ax)
end

"""
    ssortperm!!(ix, v; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    Like [`ssortperm`](@ref), but also sorts v and accepts a preallocated index vector or array `ix` with the same `axes` as `v`.
    `ix` is initialized to contain the values `LinearIndices(v)`.
# Examples
```jldoctest
julia> v = [3, 1, 2]; ix = [0,0,0];
julia> ssortperm!(ix, v);
julia> ix
3-element Vector{Int64}:
 2
 3
 1
julia> v
3-element Vector{Int64}:
 1
 2
 3
```
"""
function ssortperm!!(ix, v; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    ix .= LinearIndices(v)
    vs = StructArray{Tuple{eltype(v),eltype(ix)}}(val=v, ix=ix)
    o = ord(lt, by, rev ? true : nothing, order)
    lo = firstindex(vs)
    hi = lastindex(vs)
    a = BranchlessPatternDefeatingQuicksortAlg()
    offsets_l = MVector{PDQ_BLOCK_SIZE, Int}(undef)
    offsets_r = MVector{PDQ_BLOCK_SIZE, Int}(undef)
    pdqsort_loop!(vs, lo, hi, a, Base.Order.By(x->x[1], o), log2i(hi + 1 - lo), offsets_l, offsets_r)
    sort_equal_subarrays!(v, vs, o, offsets_l, offsets_r)
    ix
end

"""
    ssortperm!(ix, v; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    Like [`ssortperm`](@ref), but accepts a preallocated index vector or array `ix` with the same `axes` as `v`.
    `ix` is initialized to contain the values `LinearIndices(v)`.
# Examples
```jldoctest
julia> v = [3, 1, 2]; ix = [0,0,0];
julia> ssortperm!(ix, v);
julia> ix
3-element Vector{Int64}:
 2
 3
 1
julia> v[ix]
3-element Vector{Int64}:
 1
 2
 3
```
"""
function ssortperm!(ix, v; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    vv = copymutable(v)
    ssortperm!!(ix, vv, lt=lt, by=by, rev=rev, order=order)
end

"""
    ssortperm!(v; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    Like [`ssortperm`](@ref), but also sorts v.
# Examples
```jldoctest
julia> v = [3, 1, 2];
julia> ssortperm!(v)
3-element Vector{Int64}:
 2
 3
 1
julia> v
3-element Vector{Int64}:
 1
 2
 3
```
"""
function ssortperm!(v; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    ix = allocate_index_vector(v)
    ssortperm!!(ix, v, lt=lt, by=by, rev=rev, order=order)
end

"""
    ssortperm(v; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
Return a permutation vector `p` that puts `v[p]` in sorted order.
The order is specified using the same keywords as [`sort!`](@ref). The permutation is guaranteed to be stable.
See also [`ssortperm!`](@ref),[`ssortperm!!`](@ref),
 [`ssortperm!`](@ref), [`sortperm!`](@ref), [`partialsortperm`](@ref), [`invperm`](@ref), [`indexin`](@ref).
# Examples
```jldoctest
julia> v = [3, 1, 2];
julia> p = ssortperm(v)
3-element Vector{Int64}:
 2
 3
 1
julia> v[p]
3-element Vector{Int64}:
 1
 2
 3
```
"""
function ssortperm(v; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    ix = allocate_index_vector(v)
	vv = copymutable(v)
    ssortperm!!(ix, vv, lt=lt, by=by, rev=rev, order=order)
end

end
