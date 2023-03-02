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
    if nonmissingtype(eltype(v)) != eltype(v) && o isa DirectOrdering && hi>lo
        lo, hi = send_to_end!(x->ismissing(x[1]), vs, o, true; lo, hi)
        v_nomissing = WithoutMissingVector(v, unsafe=true)
        vs_nomissing = StructArray{Tuple{eltype(v_nomissing),eltype(ix)}}(val=v_nomissing, ix=ix)
        _sortperm_IEEEFloat_optimization!!(ix, v_nomissing, vs_nomissing, lo, hi, o::Ordering, offsets_l, offsets_r, true)
    else
        _sortperm_IEEEFloat_optimization!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r, false)
    end
    
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
        _sortperm_short_string_optimization!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
    end
end

function uintmap_string(s,T)
    x = zero(T)
    len = ncodeunits(s)
    lenType = sizeof(T)
    uselen = min(len,lenType) 
    shift = 8lenType
    for i = 1:uselen
        shift -= 8
        @inbounds x |= T(codeunit(s, i)) << shift        
    end
    x
end

function uinttype_of_size(x)
    @assert 1 <= x <= 16
    (UInt8, UInt16, UInt32, UInt32,
    UInt64, UInt64,UInt64, UInt64,
    UInt128, UInt128, UInt128, UInt128,
    UInt128, UInt128, UInt128, UInt128)[x]
end

function uintmap_strings!(v, vs::AbstractArray{String})
    @inbounds for i in eachindex(v)
        v[i] = uintmap_string(vs[i],eltype(v))
    end
end

function uintmap_strings(vs::AbstractArray{String}, T::Type)
    v = similar(Vector{T}, axes(vs))
    uintmap_strings!(v, vs)
    v
end

function _sortperm_short_string_optimization!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
    if eltype(v) === String && o isa DirectOrdering
        maxlength = mapreduce(ncodeunits, max, v)
        if maxlength > 16 || maxlength < 1
            return _sortperm!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
        end
        T = uinttype_of_size(maxlength)
        vu = uintmap_strings(v,T)
        vsu = StructArray{Tuple{T,eltype(ix),String}}(val=vu, ix=ix, s=v)
        _sortperm!!(ix, vu, vsu, lo, hi, o::Ordering, offsets_l, offsets_r) 
    else
        _sortperm!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
    end
end

function _sortperm!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
    pdq_loop!(vs, lo, hi, Base.Order.By(x->x[1], o), offsets_l, offsets_r)
    sort_equal_subarrays!(v, vs, lo, hi, o, offsets_l, offsets_r)
end

function _sortperm_Missing_optimization!(ix, v, o::Ordering)
    lo = firstindex(v)
    hi = lastindex(v)
    if nonmissingtype(eltype(v)) != eltype(v) && o isa DirectOrdering && hi>lo
        lo_i, hi_i = lo, hi
        offset = firstindex(v)-1;
        for (i,x) in zip(eachindex(v).-offset, v)
            if ismissing(x) == (o isa ReverseOrdering) # should i go at the beginning?
                ix[lo_i] = i
                lo_i += 1
            else
                ix[hi_i] = i
                hi_i -= 1
            end
        end
        reverse!(ix, lo_i, hi)
        if o isa ReverseOrdering
            lo = lo_i
        else
            hi = hi_i
        end
        vv = similar(WithoutMissingVector(v, unsafe=true))
        for i in lo:hi
            vv[i] = v[ix[i]]
        end
    else
        ix .= LinearIndices(v)
        vv = copymutable(v)
    end
    vs = StructArray{Tuple{eltype(vv),eltype(ix)}}(val=vv, ix=ix) 
    offsets_l = MVector{PDQ_BLOCK_SIZE, Int}(undef)
    offsets_r = MVector{PDQ_BLOCK_SIZE, Int}(undef)
    _sortperm_IEEEFloat_optimization!!(ix, vv, vs, lo, hi, o::Ordering, offsets_l, offsets_r, false)
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
    o = ord(lt, by, rev ? true : nothing, order)
    _sortperm_Missing_optimization!(ix, v, o::Ordering)
    ix
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
