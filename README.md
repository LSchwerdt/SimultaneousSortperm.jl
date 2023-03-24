# SimultaneousSortperm

[![Build Status](https://github.com/LSchwerdt/SimultaneousSortperm.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LSchwerdt/SimultaneousSortperm.jl/actions/workflows/CI.yml?query=branch%3Amain)

The `SimultaneousSortperm` package provides functions that mimic `sortperm` and `sortperm!`, but achieve better performance for large input sizes by simultaneously sorting the data and index vector.
First the data is sorted using the unstable Pattern-Defeating-Quicksort algorithm while simultaneously moving the corresponding indices.
In a second pass, all subarrays with equal data elements are sorted according to their indices to ensure stability.

The following functions are exported:

- `ssortperm(v)` – Return a permutation vector `p` that puts `v[p]` in sorted order.
- `ssortperm!(ix, v)` – Modify vector `ix` so that `v[ix]` is in sorted order.
- `ssortperm!(v)` – Sort `v` and return the permutation vector `p` that was used to put `v` in sorted order.
- `ssortperm!!(ix, v)` – Sort `v` and modify the vector `ix`, so that it contains the permutation which was used to put `v` in sorted order.

## Benchmarks

<img src="https://github.com/LSchwerdt/MiscJulia/blob/11bc3588da5d93ee0b91d58012b0b41dc7ffcab3/benchmark_ssortperm/Intel_7820x/Int64.svg">

More benchmark results can be found [here](https://github.com/LSchwerdt/MiscJulia/tree/master/benchmark_ssortperm).

## Roadmap

- Use pattern-defeating-quicksort from SortingAlgorithms when PR is merged.
- Implement `dims` keyword (added to `sortperm` in Julia 1.9).
- Dispatch to `sortperm` / `sortperm!` when they are faster (very small inputs)?
- Contribute to Base / SortingAlgorithms
- (Include option to use different sorting algorithms?)