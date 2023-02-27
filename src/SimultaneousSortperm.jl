__precompile__()

module SimultaneousSortperm

include("pdqsort.jl")

using StructArrays
using StaticArrays

using Base.Order
using Base: copymutable, uinttype

export ssortperm!!, ssortperm!, ssortperm

# send_to end from Base.sort
# copied here for compatibility with earlier Julia versions
"""
    send_to_end!(f::Function, v::AbstractVector; [lo, hi])
Send every element of `v` for which `f` returns `true` to the end of the vector and return
the index of the last element for which `f` returns `false`.
`send_to_end!(f, v, lo, hi)` is equivalent to `send_to_end!(f, view(v, lo:hi))+lo-1`
Preserves the order of the elements that are not sent to the end.
"""
function send_to_end!(f::F, v::AbstractVector; lo=firstindex(v), hi=lastindex(v)) where F <: Function
    i = lo
    @inbounds while i <= hi && !f(v[i])
        i += 1
    end
    j = i + 1
    @inbounds while j <= hi
        if !f(v[j])
            v[i], v[j] = v[j], v[i]
            i += 1
        end
        j += 1
    end
    i - 1
end
"""
    send_to_end!(f::Function, v::AbstractVector, o::DirectOrdering[, end_stable]; lo, hi)
Return `(a, b)` where `v[a:b]` are the elements that are not sent to the end.
If `o isa ReverseOrdering` then the "end" of `v` is `v[lo]`.
If `end_stable` is set, the elements that are sent to the end are stable instead of the
elements that are not
"""
@inline send_to_end!(f::F, v::AbstractVector, ::ForwardOrdering, end_stable=false; lo, hi) where F <: Function =
    end_stable ? (lo, hi-send_to_end!(!f, view(v, hi:-1:lo))) : (lo, send_to_end!(f, v; lo, hi))
@inline send_to_end!(f::F, v::AbstractVector, ::ReverseOrdering, end_stable=false; lo, hi) where F <: Function =
    end_stable ? (send_to_end!(!f, v; lo, hi)+1, hi) : (hi-send_to_end!(f, view(v, hi:-1:lo))+1, hi)

after_zero(::ForwardOrdering, x) = !signbit(x)
after_zero(::ReverseOrdering, x) = signbit(x)
is_concrete_IEEEFloat(T::Type) = T <: Base.IEEEFloat && isconcretetype(T)

# Missing optimization end from Base.sort
# copied here for compatibility with earlier Julia versions
struct WithoutMissingVector{T, U} <: AbstractVector{T}
    data::U
    function WithoutMissingVector(data; unsafe=false)
        if !unsafe && any(ismissing, data)
            throw(ArgumentError("data must not contain missing values"))
        end
        new{nonmissingtype(eltype(data)), typeof(data)}(data)
    end
end
Base.@propagate_inbounds function Base.getindex(v::WithoutMissingVector, i)
    out = v.data[i]
    @assert !(out isa Missing)
    out::eltype(v)
end
Base.@propagate_inbounds function Base.setindex!(v::WithoutMissingVector, x, i)
    v.data[i] = x
    v
end
Base.size(v::WithoutMissingVector) = size(v.data)


function sort_equal_subarrays!(v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
    i = lo
    while i < hi
        if !lt(o, v[i], v[i+1]) 
            first_equal = i
            while i < hi && !lt(o, v[i], v[i+1])
                i += 1
            end
            last_equal = i
            pdq_loop!(vs, first_equal, last_equal, Base.Order.By(x->x[2]), offsets_l, offsets_r)
        end
        i += 1
    end
end

function allocate_index_vector(v)
    ax = axes(v, 1)
    similar(Vector{eltype(ax)}, ax)
end

pdq_loop!(v, lo, hi, o, offsets_l, offsets_r) = pdqsort_loop!(v, lo, hi, BranchlessPatternDefeatingQuicksortAlg(), o, log2i(hi + 1 - lo), offsets_l, offsets_r)

function _sortperm_inplace_Missing_optimization!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
    was_unstable = false
    if nonmissingtype(eltype(v)) != eltype(v) && o isa DirectOrdering && hi>lo
        lo, hi = send_to_end!(x->ismissing(x[1]), vs, o, true; lo, hi)
        v = WithoutMissingVector(v, unsafe=true)
        vs = StructArray{Tuple{eltype(v),eltype(ix)}}(val=v, ix=ix)
        was_unstable = true
    end
    _sortperm_IEEEFloat_optimization!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r, was_unstable)
end

function _sortperm_IEEEFloat_optimization!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r, was_unstable)
    if is_concrete_IEEEFloat(eltype(v)) && o isa DirectOrdering
        lo2, hi2 = send_to_end!(x->isnan(x[1]), vs, o, true; lo, hi)
        if was_unstable # sort NaNs by index, because previous optimization was unstable 
            # only one of them does work depending on Forward/Reverse Ordering
            pdq_loop!(vs, hi2+1, hi, Base.Order.By(x->x[2]), offsets_l, offsets_r)
            pdq_loop!(vs, lo, lo2-1, Base.Order.By(x->x[2]), offsets_l, offsets_r)
        end
        lo, hi = lo2, hi2
        ivType = uinttype(eltype(v))
        iv = reinterpret(ivType, v)
        ivs = StructArray{Tuple{ivType,eltype(ix)}}(val=iv, ix=ix)
        j = send_to_end!(x -> after_zero(o, x[1]), vs; lo, hi)
        _sortperm!!(ix, iv, ivs, lo, j, Reverse, offsets_l, offsets_r)
        _sortperm!!(ix, iv, ivs, j+1, hi, Forward, offsets_l, offsets_r)
    else
        _sortperm!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
    end
end

function _sortperm!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
    pdq_loop!(vs, lo, hi, Base.Order.By(x->x[1], o), offsets_l, offsets_r)
    sort_equal_subarrays!(v, vs, lo, hi, o, offsets_l, offsets_r)
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
    offsets_l = MVector{PDQ_BLOCK_SIZE, Int}(undef)
    offsets_r = MVector{PDQ_BLOCK_SIZE, Int}(undef)
    _sortperm_inplace_Missing_optimization!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
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
    ssortperm!(ix, v, lt=lt, by=by, rev=rev, order=order)
end

end
