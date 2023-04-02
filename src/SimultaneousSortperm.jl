__precompile__()

module SimultaneousSortperm

include("pdqsort.jl")

using StructArrays
using StaticArrays
using Base.Order
using Base: copymutable, uinttype, sub_with_overflow, add_with_overflow

export ssortperm!!, ssortperm!, ssortperm

const SSORTPERM_INPLACE_SMALL_THRESHOLD = 40
const SSORTPERM_SMALL_THRESHOLD = 80

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
Base.axes(v::WithoutMissingVector) = axes(v.data)


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


function _sortperm!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
    pdq_loop!(vs, lo, hi, Base.Order.By(x->x[1], o), offsets_l, offsets_r)
    sort_equal_subarrays!(v, vs, lo, hi, o, offsets_l, offsets_r)
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

# map string to UInt starting at firstcodeunit.
# Maps at most sizeof(T) - 1 codeunits.
# Use uANS to encode 257 Symbols per codeunint.
# Extra Symbol is below '\0' to distinguish "" from "\0".
# uANS: https://arxiv.org/abs/1311.2540
function uintmap_string(s::String, ::Type{T}, firstcodeunit::Int) where T<:Unsigned
    last_possible_codeunint = sizeof(T) + firstcodeunit - 2
    len = ncodeunits(s)
    lastcodeunint = min(len, last_possible_codeunint)
    x = zero(T)
    i = firstcodeunit
    while i <= lastcodeunint
        x *= T(257)
        @inbounds x += T(codeunit(s, i)) + T(1)
        i += 1
    end
    while i <= last_possible_codeunint
        x *= T(257)
        i += 1
    end
    x, len
end

uintmap_string(s::String, ::Type{T}) where T<:Unsigned = uintmap_string(s, T, 1)[1]

# uinttype that can fit min(x,8) codeunints
function uinttype_of_size(x::Integer)
    @assert 1 <= x
    (UInt16, UInt32, UInt32, UInt64,
    UInt64, UInt64, UInt64, UInt128)[min(x,8)]
end

# map codeunits starting at firstcodeunit for some strings (from recursion depth 2 on)
function uintmap_strings!(v, vs::AbstractVector{String}, lo::Int, hi::Int, ix, firstcodeunint)
    maxlength = 0
    @inbounds for i in lo:hi
        v[i], len = uintmap_string(vs[ix[i]], eltype(v), firstcodeunint)
        maxlength = max(maxlength, len)
    end
    maxlength
end

# map first codeunits of some strings (after missing optimization)
function uintmap_strings(vs::AbstractVector{String}, ::Type{T}, lo::Int, hi::Int, ix) where T
    v = similar(Vector{T}, axes(vs))
    @inbounds for i in lo:hi
        v[i] = uintmap_string(vs[ix[i]],T)
    end
    v
end

# map first codeunits of all strings
function uintmap_strings(vs::AbstractVector{String}, ::Type{T}) where T
    map(s->uintmap_string(s, T), vs)
end

maxncodeunints(v::AbstractVector{String}) = mapreduce(ncodeunits, max, v, init=0)
maxncodeunints(v::AbstractVector{Union{String,Missing}}) = mapreduce(x-> ismissing(x) ? 0 : ncodeunits(x), max, v, init=0)

# optimization for short strings (length <=45 codeunints)
function _sortperm_type_optimization!(ix, v::Union{AbstractVector{String},AbstractVector{Union{String,Missing}}}, lo::Int, hi::Int, o::Ordering, vcontainsmissing::Bool)
    if eltype(v) === String && o isa DirectOrdering && 1 <= (maxlength = maxncodeunints(v)) <= 45
        # map strings to UInts for faster comparisons
        # this can be slower for strings with long common prefixes -> use only if maxlength <= 45
        T = uinttype_of_size(maxlength)
        if vcontainsmissing
            vu = uintmap_strings(v, T, lo, hi, ix)
        else
            vu = uintmap_strings(v, T)
        end
        vs = StructArray{Tuple{T,eltype(ix)}}(val=vu, ix=ix)
        offsets_l = MVector{PDQ_BLOCK_SIZE, Int}(undef)
        offsets_r = MVector{PDQ_BLOCK_SIZE, Int}(undef)
        _sortperm_string!!(ix, vu, vs, v, lo, hi, T, 1, maxlength, o::Ordering, offsets_l, offsets_r)
    else
        _sortperm_finish_Missing_optimization!(ix, v, lo::Int, hi::Int, o::Ordering, vcontainsmissing::Bool)
    end
end

function _sortperm_type_optimization!(ix, v::AbstractVector{Bool}, lo::Int, hi::Int, o::Ordering, vcontainsmissing::Bool)
    if o isa DirectOrdering
        _sortperm_bool_optimization!(ix, v, lo, hi, o)
    else
        _sortperm_finish_Missing_optimization!(ix, v, lo::Int, hi::Int, o::Ordering, vcontainsmissing::Bool)
    end
end

# from Base.sort
function _sortperm_type_optimization!(ix, v::AbstractVector{<:Integer}, lo::Int, hi::Int, o::Ordering, vcontainsmissing::Bool)
    if o === Forward
        n = length(v)
        if n > 1
            min, max = extrema(v)
            (diff, o1) = sub_with_overflow(max, min)
            (rangelen, o2) = add_with_overflow(diff, oneunit(diff))
            if !(o1 || o2)::Bool && rangelen < div(n,2)
                return sortperm_int_range!(ix, v, rangelen, min)
            end
        end
    end
    _sortperm_finish_Missing_optimization!(ix, v, lo::Int, hi::Int, o::Ordering, vcontainsmissing::Bool)
end

# sortperm for vectors of few unique integers
# based on code from Base.sort
function sortperm_int_range!(ix, v::AbstractVector{<:Integer}, rangelen, minval)
    offs = 1 - minval

    counts = fill(0, rangelen+1)
    counts[1] = firstindex(v)
    @inbounds for i = eachindex(v)
        counts[v[i] + offs + 1] += 1
    end

    #cumsum!(counts, counts)
    @inbounds for i = 2:lastindex(counts)
        counts[i] += counts[i-1]
    end

    @inbounds for i = eachindex(ix)
        label = v[i] + offs
        ix[counts[label]] = i
        counts[label] += 1
    end

    return ix
end

# based on BoolOptimization from Base.sort
function _sortperm_bool_optimization!(ix, v::AbstractVector{Bool}, lo::Int, hi::Int, o::Ordering)
    first = lt(o, false, true) ? false : lt(o, true, false) ? true : return ix
    count = 0
    @inbounds for i in lo:hi
        if v[i] == first
            count += 1
        end
    end
    j = lo
    k = lo + count
    @inbounds for i in lo:hi
        if v[i] == first
            ix[j] = i
            j += 1
        else
            ix[k] = i
            k += 1
        end
    end
    ix
end

# fallback if no allocating optimization for special types is used
_sortperm_type_optimization!(ix, v::AbstractVector, lo::Int, hi::Int, o::Ordering, vcontainsmissing::Bool) = 
_sortperm_finish_Missing_optimization!(ix, v, lo, hi, o, vcontainsmissing)


function _sortperm_finish_Missing_optimization!(ix, v, lo::Int, hi::Int, o::Ordering, vcontainsmissing::Bool)
    offsets_l = MVector{PDQ_BLOCK_SIZE, Int}(undef)
    offsets_r = MVector{PDQ_BLOCK_SIZE, Int}(undef)
    if vcontainsmissing
        vv = similar(WithoutMissingVector(v, unsafe=true)) 
        for i in lo:hi
            vv[i] = v[ix[i]]
        end
    else
        vv = copymutable(v)
    end
    vs = StructArray{Tuple{eltype(vv),eltype(ix)}}(val=vv, ix=ix)
    _sortperm_IEEEFloat_optimization!!(ix, vv, vs, lo, hi, o::Ordering, offsets_l, offsets_r, false)
end

function _sortperm_string!!(ix, v, vs, v_string, lo, hi, ::Type{T}, firstcodeunint, maxlength, o::Ordering, offsets_l, offsets_r) where T
    if firstcodeunint > maxlength
        return _sortperm!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
    end
    pdq_loop!(vs, lo, hi, Base.Order.By(x->x[1], o), offsets_l, offsets_r)
    # sort equal subarrays by next codeunints
    i = lo
    while i < hi
        if !lt(o, v[i], v[i+1])
            first_equal = i
            while i < hi && !lt(o, v[i], v[i+1])
                i += 1
            end
            last_equal = i
            maxlength = uintmap_strings!(v, v_string, first_equal, last_equal, ix, firstcodeunint+sizeof(T)-1)
            _sortperm_string!!(ix, v, vs, v_string, first_equal, last_equal, T, firstcodeunint+sizeof(T)-1, maxlength, o::Ordering, offsets_l, offsets_r)
        end
        i += 1
    end

end

function _sortperm_Missing_optimization!(ix, v, o::Ordering)
    lo = firstindex(v)
    hi = lastindex(v)
    if nonmissingtype(eltype(v)) != eltype(v) && o isa DirectOrdering && hi>lo
        lo_i, hi_i = lo, hi
        for (i,x) in zip(eachindex(v), v)
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
        hi <= lo && return # all items were missing
        vcontainsmissing = true
    else
        ix .= LinearIndices(v)
        vcontainsmissing = false
    end
    _sortperm_type_optimization!(ix, v, lo, hi, o::Ordering, vcontainsmissing)
end

function _sortperm_inplace_small_optimization!(ix, v, vs, lo, hi, o::Ordering)
    if hi + 1 <= lo + SSORTPERM_INPLACE_SMALL_THRESHOLD
        # InsertionSort is stable. No need for sort_equal_subarrays!
        sort!(vs, lo, hi, InsertionSort, Base.Order.By(x->x[1], o))
        return ix
    end
    offsets_l = MVector{PDQ_BLOCK_SIZE, Int}(undef)
    offsets_r = MVector{PDQ_BLOCK_SIZE, Int}(undef)
    _sortperm_inplace_Missing_optimization!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
end

function _sortperm_small_optimization(ix, v, o::Ordering)
    if length(v) <= SSORTPERM_SMALL_THRESHOLD
        return sortperm!(ix, v, order=o)
    end
    _sortperm_Missing_optimization!(ix, v, o)
end

"""
    ssortperm!!(ix::AbstractVector{Int}, v::AbstractVector; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
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
function ssortperm!!(ix::AbstractVector{Int}, v::AbstractVector; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    axes(ix) == axes(v) || throw(ArgumentError("index array must have the same size/axes as the source array, $(axes(ix)) != $(axes(v))"))
    ix .= LinearIndices(v)
    vs = StructArray{Tuple{eltype(v),eltype(ix)}}(val=v, ix=ix)
    o = ord(lt, by, rev ? true : nothing, order)
    lo = firstindex(vs)
    hi = lastindex(vs)
    _sortperm_inplace_small_optimization!(ix, v, vs, lo, hi, o::Ordering)
    ix
end

"""
    ssortperm!(ix::AbstractVector{Int}, v::AbstractVector; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)

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
function ssortperm!(ix::AbstractVector{Int}, v::AbstractVector; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    axes(ix) == axes(v) || throw(ArgumentError("index array must have the same size/axes as the source array, $(axes(ix)) != $(axes(v))"))
    o = ord(lt, by, rev ? true : nothing, order)
    _sortperm_small_optimization(ix, v, o)
    ix
end

"""
    ssortperm!(v::AbstractVector; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)

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
function ssortperm!(v::AbstractVector; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    ix = allocate_index_vector(v)
    ssortperm!!(ix, v, lt=lt, by=by, rev=rev, order=order)
end

"""
    ssortperm(v::AbstractVector; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)

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
function ssortperm(v::AbstractVector; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
    ix = allocate_index_vector(v)
    ssortperm!(ix, v, lt=lt, by=by, rev=rev, order=order)
end

# ssortperm with dims

function ssortperm_chunks!!(ix, v, vs, n, o)
    fst = firstindex(vs)
    lst = lastindex(vs)
    if n <= SSORTPERM_INPLACE_SMALL_THRESHOLD
        for lo = fst:n:lst
            hi=lo+n-1
            sort!(vs, lo, hi, InsertionSort, Base.Order.By(x->x[1], o))
        end
    else
        offsets_l = MVector{PDQ_BLOCK_SIZE, Int}(undef)
        offsets_r = MVector{PDQ_BLOCK_SIZE, Int}(undef)
        for lo = fst:n:lst
            hi=lo+n-1
            _sortperm_inplace_Missing_optimization!!(ix, v, vs, lo, hi, o::Ordering, offsets_l, offsets_r)
        end
    end
    ix
end

function ssortperm_dim1!!(ix::AbstractArray{Int}, A::AbstractArray, o)
    Av = vec(A)
    ixv = vec(ix)
    vs = StructArray{Tuple{eltype(A),eltype(ix)}}(val=Av, ix=ixv)
    n = length(axes(A, 1))
    ssortperm_chunks!!(ixv, Av, vs, n, o)
end

"""
    ssortperm!!(ix::AbstractArray{Int}, A::AbstractArray; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward, dims::Integer)

Like [`ssortperm!`](@ref), but also sorts A and accepts a preallocated index vector or array `ix` with the same `axes` as `A`.
`ix` is initialized to contain the values `LinearIndices(A)`.

# Examples
```jldoctest
julia> A = [8 7; 5 6]; p = zeros(Int,2, 2);
julia> ssortperm!!(p, A; dims=1); p
2×2 Matrix{Int64}:
 2  4
 1  3
julia> A
2×2 Matrix{Int64}:
 5  6
 8  7
julia> ssortperm!!(p, A; dims=2); p
2×2 Matrix{Int64}:
 1  3
 4  2
julia> A
2×2 Matrix{Int64}:
 5  6
 7  8
```
"""
function ssortperm!!(ix::AbstractArray{Int}, A::AbstractArray; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward, dims::Integer)
    axes(ix) == axes(A) || throw(ArgumentError("index array must have the same size/axes as the source array, $(axes(ix)) != $(axes(A))"))
    dim = dims
    ix .= LinearIndices(A)
    o = ord(lt, by, rev ? true : nothing, order)    
    if dim == 1
        ssortperm_dim1!!(ix::AbstractArray{Int}, A::AbstractArray, o)
    else
        pdims = (dim, setdiff(1:ndims(A), dim)...)  # put the selected dimension first
        Ap = permutedims(A, pdims)
        ixp = permutedims(ix, pdims)
        ssortperm_dim1!!(ixp, Ap, o)
        permutedims!(A, Ap, invperm(pdims))
        permutedims!(ix, ixp, invperm(pdims))
    end
    ix
end

"""
    ssortperm!(ix::AbstractArray{Int}, A::AbstractArray; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward, dims::Integer)

Like [`ssortperm`](@ref), but accepts a preallocated index vector or array `ix` with the same `axes` as `A`.
`ix` is initialized to contain the values `LinearIndices(A)`.

# Examples
```jldoctest
julia> A = [8 7; 5 6]; p = zeros(Int,2, 2);
julia> ssortperm!(p, A; dims=1); p
2×2 Matrix{Int64}:
 2  4
 1  3
julia> sortperm!(p, A; dims=2); p
2×2 Matrix{Int64}:
 3  1
 2  4
```
"""
function ssortperm!(ix::AbstractArray{Int}, A::AbstractArray; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward, dims::Integer)
    axes(ix) == axes(A) || throw(ArgumentError("index array must have the same size/axes as the source array, $(axes(ix)) != $(axes(A))"))
    dim = dims
    ix .= LinearIndices(A)
    o = ord(lt, by, rev ? true : nothing, order)
    # do not use ssortperm!! to avoid one copy of A if dims > 1
    if dim == 1
        Ac = copymutable(A)
        ssortperm_dim1!!(ix, Ac, o)
    else
        pdims = (dim, setdiff(1:ndims(A), dim)...)  # put the selected dimension first
        Ap = permutedims(A, pdims)
        ixp = permutedims(ix, pdims)
        ssortperm_dim1!!(ixp, Ap, o)
        permutedims!(ix, ixp, invperm(pdims))
    end
    ix
end

"""
    ssortperm!(A::AbstractArray; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward, dims::Integer)

Return a permutation vector or array `I` that puts `A[I]` in sorted order along the given dimension, and sort A.
If `A` has more than one dimension, then the `dims` keyword argument must be specified. The order is specified
using the same keywords as [`ssortperm!`](@ref). The permutation is guaranteed to be stable.
See also [`sortperm!`](@ref), [`partialsortperm`](@ref), [`invperm`](@ref), [`indexin`](@ref).
To sort slices of an array, refer to [`sortslices`](@ref).

# Examples
```jldoctest
julia> A = [8 7; 5 6]
2×2 Matrix{Int64}:
 8  7
 5  6
julia> ssortperm!(A, dims = 1)
2×2 Matrix{Int64}:
 2  4
 1  3
julia> A
2×2 Matrix{Int64}:
 5  6
 8  7
 julia> ssortperm!(A, dims = 2)
 2×2 Matrix{Int64}:
 1  3
 4  2
 julia> A
 2×2 Matrix{Int64}:
 5  6
 7  8
```
"""
function ssortperm!(A::AbstractArray; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward, dims::Integer)
    ix = copymutable(LinearIndices(A))
    ssortperm!!(ix, A, lt=lt, by=by, rev=rev, order=order, dims=dims)
end

"""
    ssortperm(A::AbstractArray; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward, dims::Integer)

Return a permutation vector or array `I` that puts `A[I]` in sorted order along the given dimension.
If `A` has more than one dimension, then the `dims` keyword argument must be specified. The order is specified
using the same keywords as [`ssortperm!`](@ref). The permutation is guaranteed to be stable.
See also [`sortperm!`](@ref), [`partialsortperm`](@ref), [`invperm`](@ref), [`indexin`](@ref).
To sort slices of an array, refer to [`sortslices`](@ref).

# Examples
```jldoctest
julia> A = [8 7; 5 6]
2×2 Matrix{Int64}:
 8  7
 5  6
julia> ssortperm(A, dims = 1)
2×2 Matrix{Int64}:
 2  4
 1  3
julia> ssortperm(A, dims = 2)
2×2 Matrix{Int64}:
 3  1
 2  4
```
"""
function ssortperm(A::AbstractArray; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward, dims::Integer)
    ix = copymutable(LinearIndices(A))
    ssortperm!(ix, A, lt=lt, by=by, rev=rev, order=order, dims=dims)
end

end
