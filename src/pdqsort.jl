using Base.Sort
using Base.Order
using Base: Cartesian

import Base.Sort: sort!
using SortingAlgorithms: HeapSortAlg
import StaticArrays: MVector


abstract type PatternDefeatingQuicksortAlg  <: Algorithm end
struct BranchyPatternDefeatingQuicksortAlg  <: PatternDefeatingQuicksortAlg end
struct BranchlessPatternDefeatingQuicksortAlg  <: PatternDefeatingQuicksortAlg end

function maybe_optimize(x::Algorithm) 
    isdefined(Base.Sort, :InitialOptimizations) ? Base.Sort.InitialOptimizations(x) : x
end

"""
    BranchyPatternDefeatingQuicksortAlg

Quicksort with improved performance on special input patterns.

Presorted inputs (including reverse and almost presorted ones), as well as inputs with many duplicates are
sorted in less than n log n time.
The code is based closely on the original C++ implementation by Orson Peters (see References).

Characteristics:
 - *not stable* does not preserve the ordering of elements which
   compare equal (e.g. "a" and "A" in a sort of letters which
   ignores case).
 - *in-place* in memory.
 - *`n log n` garuanteed runtime* by falling back to heapsort for pathological inputs.

## References
 - https://arxiv.org/pdf/2106.05123.pdf
 - https://github.com/orlp/pdqsort
"""
const BranchyPatternDefeatingQuicksort  = maybe_optimize(BranchyPatternDefeatingQuicksortAlg())
const BranchyPdqSort  = BranchyPatternDefeatingQuicksort

"""
    BranchlessPatternDefeatingQuicksortAlg

Quicksort with improved performance on special input patterns.

Presorted inputs (including reverse and almost presorted ones), as well as inputs with many duplicates are
sorted in less than n log n time. Uses branchless block partitioning scheme, which is faster for simple types.
The code is based closely on the original C++ implementation by Orson Peters (see References).

Characteristics:
 - *not stable* does not preserve the ordering of elements which
   compare equal (e.g. "a" and "A" in a sort of letters which
   ignores case).
 - *constant* auxilary memory (approximately 1KiB on 64-bit systems).
 - *`n log n` garuanteed runtime* by falling back to heapsort for pathological inputs.

## References
 - https://arxiv.org/pdf/2106.05123.pdf
 - https://github.com/orlp/pdqsort
 - https://dl.acm.org/doi/10.1145/3274660
 - http://arxiv.org/abs/1604.06697

"""
const BranchlessPatternDefeatingQuicksort  = maybe_optimize(BranchlessPatternDefeatingQuicksortAlg())
const BranchlessPdqSort  = BranchlessPatternDefeatingQuicksort

const PDQ_SMALL_THRESHOLD = 32
const PDQ_NINTHER_THRESHOLD = 128
const PDQ_PARTIAL_INSERTION_SORT_LIMIT = 8
const PDQ_BLOCK_SIZE = 64



"""
    unguarded_insertion_sort!(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)

Sorts v[lo:hi] using insertion sort with the given ordering. Assumes
v[lo-1] is an element smaller than or equal to any element in v[lo:hi].
"""
function unguarded_insertion_sort!(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)
    lo_plus_1 = (lo + 1)::Integer
    @inbounds for i = lo_plus_1:hi
        j = i
        x = v[i]
        while true
            y = v[j-1]
            if !(lt(o, x, y)::Bool)
                break
            end
            v[j] = y
            j -= 1
        end
        v[j] = x
    end
    v
end

"""
    partial_insertion_sort!(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)

Attempts to use insertion sort on v[lo:hi]. Will return false if more than
PDQ_PARTIAL_INSERTION_SORT_LIMIT elements were moved, and abort sorting. Otherwise it will
successfully sort and return true.
"""
function partial_insertion_sort!(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)
    limit = 0
    lo_plus_1 = (lo + 1)::Integer
    @inbounds for i = lo_plus_1:hi
        j = i
        x = v[i]
        while j > lo
            y = v[j-1]
            if !(lt(o, x, y)::Bool)
                break
            end
            v[j] = y
            j -= 1
        end
        v[j] = x
        limit += i - j
        limit > PDQ_PARTIAL_INSERTION_SORT_LIMIT  && return false
    end
    return true
end

"""
    partition_right!(v::AbstractVector, lo::Integer, hi::Integer, a::BranchlessPatternDefeatingQuicksortAlg, o::Ordering, offsets_l::AbstractVector{Integer}, offsets_r::AbstractVector{Integer})

Partitions v[lo:hi] around pivot v[lo] using ordering o.

Elements equal to the pivot are put in the right-hand partition. Returns the position of the pivot
after partitioning and whether the passed sequence already was correctly partitioned. Assumes the
pivot is a median of at least 3 elements and that v[lo:hi] is at least PDQ_SMALL_THRESHOLD long.
Uses branchless partitioning.
"""
function partition_right!(v::AbstractVector, lo::Integer, hi::Integer, a::BranchlessPatternDefeatingQuicksortAlg, o::Ordering, offsets_l::AbstractVector{Int}, offsets_r::AbstractVector{Int})
    # output:
    # v[lo:pivot_index-1] < pivot
    # v[pivot_index] == pivot
    # v[pivot_index+1:hi] >= pivot
    @inbounds begin
        pivot = v[lo]

        # swap pointers
        # v[lo] is pivot -> start at lo + 1
        left = lo + 1
        right = hi
        # Find the first element greater than or equal than the pivot (the median of 3 guarantees
        # this exists).
        while lt(o, v[left], pivot)
            left += 1
        end
        # Find the first element strictly smaller than the pivot. We have to guard this search if
        # there was no element before v[left].
        if left - 1 == lo
            while left < right && !lt(o, v[right], pivot)
                right -= 1
            end
        else
            while !lt(o, v[right], pivot)
                right -= 1
            end
        end

        # If the first pair of elements that should be swapped to partition are the same element,
        # the passed in sequence already was correctly partitioned.
        was_already_partitioned = left >= right
        if !was_already_partitioned
            v[left], v[right] = v[right], v[left]
            left += 1
            right -= 1

            offsets_l_base = left
            offsets_r_base = right
            start_l = 0; start_r = 0
            num_l = 0; num_r = 0

            while left < right + 1
                # Fill up offset blocks with elements that are on the wrong side.
                # First we determine how much elements are considered for each offset block.
                num_unknown = right - left + 1
                left_split = num_l == 0 ? (num_r == 0 ? num_unknown ÷ 2 : num_unknown) : 0
                right_split = num_r == 0 ? (num_unknown - left_split) : 0

                # Fill the offset blocks.
                if left_split >= PDQ_BLOCK_SIZE
                    i = 0
                    while i < PDQ_BLOCK_SIZE
                        Cartesian.@nexprs 8 _ ->
                        begin
                            offsets_l[num_l+1] = i
                            num_l += Int(!lt(o, v[left], pivot))
                            left += 1
                            i += 1
                        end
                    end
                else
                    for i in 0:left_split-1
                        offsets_l[num_l+1] = i
                        num_l += Int(!lt(o, v[left], pivot))
                        left += 1
                    end
                end
                if right_split  >= PDQ_BLOCK_SIZE
                    i = 0
                    while i < PDQ_BLOCK_SIZE
                        Cartesian.@nexprs 8 _ ->
                        begin
                            offsets_r[num_r+1] = i
                            num_r += Int(lt(o, v[right], pivot))
                            right -= 1
                            i += 1
                        end
                    end
                else
                    for i in 0:right_split-1
                        offsets_r[num_r+1] = i
                        num_r += Int(lt(o, v[right], pivot))
                        right -= 1
                    end
                end

                # Swap elements and update block sizes and left/right boundaries.
                num = min(num_l, num_r)
                for i = 1:num
                    swap!(v, offsets_l_base + offsets_l[i+start_l], offsets_r_base - offsets_r[i+start_r])
                end
                num_l -= num; num_r -= num
                start_l += num; start_r += num

                if num_l == 0
                    start_l = 0
                    offsets_l_base = left
                end

                if num_r == 0
                    start_r = 0
                    offsets_r_base = right
                end
            end

            # We have now fully identified [left, right)'s proper position. Swap the last elements.
            if num_l > 0
                while num_l > 0
                    swap!(v, offsets_l_base + offsets_l[start_l+num_l], right)
                    num_l -= 1
                    right -= 1
                end
                left = right + 1
            end
            if num_r > 0
                while num_r > 0
                    swap!(v, left, offsets_r_base - offsets_r[start_r+num_r])
                    num_r -= 1
                    left += 1
                end
                right = left
            end

        end

        # Put the pivot in the right place.
        pivot_index = left - 1
        v[lo] = v[pivot_index]
        v[pivot_index] = pivot
    end
    return pivot_index, was_already_partitioned
end

"""
    partition_right!(v::AbstractVector, lo::Integer, hi::Integer, a::BranchyPatternDefeatingQuicksortAlg, o::Ordering, _, _)

Partitions v[lo:hi] around pivot v[lo] using ordering o.

Elements equal to the pivot are put in the right-hand partition. Returns the position of the pivot
after partitioning and whether the passed sequence already was correctly partitioned. Assumes the
pivot is a median of at least 3 elements and that v[lo:hi] is at least PDQ_SMALL_THRESHOLD long.
"""
function partition_right!(v::AbstractVector, lo::Integer, hi::Integer, a::BranchyPatternDefeatingQuicksortAlg, o::Ordering, _, _)
    # output:
    # v[lo:pivot_index-1] < pivot
    # v[pivot_index] == pivot
    # v[pivot_index+1:hi] >= pivot
    @inbounds begin
        pivot = v[lo]

        # swap pointers
        # v[lo] is pivot
        left = lo + 1
        right = hi
        # Find the left element greater than or equal than the pivot (the median of 3 guarantees
        # this exists).
        while lt(o, v[left], pivot)
            left += 1
        end
        # Find the first element strictly smaller than the pivot. We have to guard this search if
        # there was no element before v[left].
        if left - 1 == lo
            while left < right && !lt(o, v[right], pivot)
                right -= 1
            end
        else
            while !lt(o, v[right], pivot)
                right -= 1
            end
        end

        # If the first pair of elements that should be swapped to partition are the same element,
        # the passed in sequence already was correctly partitioned.
        was_already_partitioned = left >= right

        # Keep swapping pairs of elements that are on the wrong side of the pivot. Previously
        # swapped pairs guard the searches, which is why the first iteration is special-cased
        # above.
        while left < right
            swap!(v, left, right)
            left += 1
            right -= 1
            while lt(o, v[left], pivot)
                left += 1
            end
            while !lt(o, v[right], pivot)
                right -= 1
            end
        end

        # Put the pivot in the right place.
        pivot_index = left - 1
        v[lo] = v[pivot_index]
        v[pivot_index] = pivot

    end
    return pivot_index, was_already_partitioned
end

"""
    partition_left!(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)

Partitions v[lo:hi] around pivot v[lo] using ordering o.

Similar function to the one above, except elements equal to the pivot are put to the left of
the pivot and it doesn't check or return if the passed sequence already was partitioned.
Since this is rarely used (the many equal case), and in that case pdqsort already has O(n)
performance, no block quicksort is applied here for simplicity.
"""
function partition_left!(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)
    # output:
    # v[lo:pivot_index-1] <= pivot
    # v[pivot_index] == pivot
    # v[pivot_index+1:hi] > pivot
    
    @inbounds begin
        pivot = v[lo]
        left = lo + 1
        right = hi
        
        while lt(o, pivot, v[right])
            right -= 1
        end
        if right == hi
            while left < right && !lt(o, pivot, v[left])
                left += 1
            end
        else
            while !lt(o, pivot, v[left])
                left += 1
            end
        end
        
        while left < right
            swap!(v, left, right)
            while lt(o, pivot, v[right])
                right -= 1
            end
            while !lt(o, pivot, v[left])
                left += 1
            end
        end
        
        # Put the pivot in the right place.
        pivot_index = right
        v[lo] = v[pivot_index]
        v[pivot_index] = pivot
    end
    return pivot_index
end

# midpoint was added to Base.sort in version 1.4 and later moved to Base
# -> redefine for compatibility with earlier versions
midpoint(lo::Integer, hi::Integer) = lo + ((hi - lo) >>> 0x01)

@inline function swap!(v::AbstractVector, i::Integer, j::Integer)
    v[i], v[j] = v[j], v[i]
end

@inline function sort2!(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)
    lt(o, v[hi], v[lo]) && swap!(v, lo, hi)
end

@inline function sort3!(v::AbstractVector, lo::Integer, m::Integer, hi::Integer, o::Ordering)
    sort2!(v, lo,  m, o)
    sort2!(v,  m, hi, o)
    sort2!(v, lo,  m, o)
end

@inline function selectpivot_ninther!(v::AbstractVector, lo::Integer, m::Integer, hi::Integer, o::Ordering)
    sort3!(v, lo  , m  , hi  , o)
    sort3!(v, lo+1, m-1, hi-1, o)
    sort3!(v, lo+2, m+1, hi-2, o)
    sort3!(v, m-1, m, m+1, o)
    swap!(v, lo, m)
end

@inline function swap3consecutive!(v::AbstractVector, i::Integer, j::Integer)
    swap!(v, i,   j)
    swap!(v, i+1, j+1)
    swap!(v, i+2, j+2)
end

# swap first 3 and last 3 elements each with 3 pseudorandomly chosen consecutive elements from v[lo+3:hi-3]
function breakpatterns!(v::AbstractVector, lo::Integer, hi::Integer)    
    # correct because hi+1-lo > PDQ_SMALL_THRESHOLD > 8
    len8 = hi - lo - 6 # length minus 8
    swap_lo = typeof(len8)(hash(hi) % len8) + lo + 3
    swap_hi = typeof(len8)(hash(lo) % len8) + lo + 3
    swap3consecutive!(v, lo, swap_lo)
    swap3consecutive!(v, hi-2, swap_hi)
end

pdqsort_loop!(v::AbstractVector, lo::Integer, hi::Integer, a::BranchlessPatternDefeatingQuicksortAlg, o::Ordering, bad_allowed::Integer, offsets_l::Nothing, offsets_r::Nothing, leftmost=true) =
pdqsort_loop!(v, lo, hi, a, o, bad_allowed, MVector{PDQ_BLOCK_SIZE, Int}(undef), MVector{PDQ_BLOCK_SIZE, Int}(undef), leftmost)

function pdqsort_loop!(v::AbstractVector, lo::Integer, hi::Integer, a::PatternDefeatingQuicksortAlg, o::Ordering, bad_allowed::Integer, offsets_l, offsets_r, leftmost=true)
    # Use a while loop for tail recursion elimination.
    @inbounds while true
        len = hi - lo + 1
        # Insertion sort is faster for small arrays.
        if len <= PDQ_SMALL_THRESHOLD
            if leftmost
                sort!(v, lo, hi, InsertionSort, o)
            else
                unguarded_insertion_sort!(v, lo, hi, o)
            end
            return v
        end
        
        # Choose pivot as median of 3 or pseudomedian of 9.
        # use hi+1 to ensure reverse sorted list is swapped perfectly
        m = midpoint(lo, hi+1)
        if len > PDQ_NINTHER_THRESHOLD
            selectpivot_ninther!(v, lo, m, hi, o)
        else
            sort3!(v, m, lo, hi, o)
        end
        # If v[lo - 1] is the end of the right partition of a previous partition operation
        # there is no element in v[lo:hi] that is smaller than v[lo - 1]. Then if our
        # pivot compares equal to v[lo - 1] we change strategy, putting equal elements in
        # the left partition, greater elements in the right partition. We do not have to
        # recurse on the left partition, since it's sorted (all equal).
        if !leftmost && !lt(o, v[lo-1], v[lo])
            lo = partition_left!(v, lo, hi, o) + 1
            continue
        end
        
        # Partition and get results.
        pivot_index, was_already_partitioned = partition_right!(v, lo, hi, a, o, offsets_l, offsets_r)
        
        # Check for a highly unbalanced partition.
        len_r = pivot_index - lo;
        len_l = hi - pivot_index;
        is_highly_unbalanced = len_r < len ÷ 8 || len_l < len ÷ 8
        
        if is_highly_unbalanced
            # If we had too many bad partitions, switch to heapsort to guarantee O(n log n).
            bad_allowed -= 1
            if bad_allowed <= 0
                sort!(v, lo, hi, HeapSortAlg(), o)
                return v
            end
            # If we got a highly unbalanced partition we shuffle elements to break adverse patterns.
            len_r > PDQ_SMALL_THRESHOLD && breakpatterns!(v, lo, pivot_index - 1)
            len_l > PDQ_SMALL_THRESHOLD && breakpatterns!(v, pivot_index + 1, hi)
        else
            # If we were decently balanced and we tried to sort an already partitioned
            # sequence try to use insertion sort.
            if was_already_partitioned &&
                partial_insertion_sort!(v, lo, pivot_index, o) &&
                partial_insertion_sort!(v, pivot_index + 1, hi, o)
                return v
            end
        end
        
        # Sort the left partition first using recursion and do tail recursion elimination for
        # the right-hand partition.
        pdqsort_loop!(v, lo, pivot_index-1, a, o, bad_allowed, offsets_l, offsets_r, leftmost)
        lo = pivot_index + 1
        leftmost = false
    end
end

# integer logarithm base two, ignoring sign
function log2i(n::Integer)
    sizeof(n) << 3 - leading_zeros(abs(n))
end

sort!(v::AbstractVector, lo::Int, hi::Int, a::PatternDefeatingQuicksortAlg, o::Ordering) =
pdqsort_loop!(v, lo, hi, a, o, log2i(hi + 1 - lo), nothing, nothing)

#=
This implementation of pattern-defeating quicksort is based on the original code from Orson Peters,
available at https://github.com/orlp/pdqsort.
Original license notice:
"""
Copyright (c) 2021 Orson Peters <orsonpeters@gmail.com>

This software is provided 'as-is', without any express or implied warranty. In no event will the
authors be held liable for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial
applications, and to alter it and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the
   original software. If you use this software in a product, an acknowledgment in the product
   documentation would be appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be misrepresented as
   being the original software.

3. This notice may not be removed or altered from any source distribution.
"""
=#