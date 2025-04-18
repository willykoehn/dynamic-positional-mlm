module Cooccurrence

using LinearAlgebra

"""
    compute_cooccurrence_matrix(ids::Vector{Int}, vocab_size::Int; window::Int=4)

Builds a symmetric co-occurrence matrix from token IDs using a sliding window.
"""
function compute_cooccurrence_matrix(ids::Vector{Int}, vocab_size::Int; window::Int=4)
    mat = zeros(Float32, vocab_size, vocab_size)
    for i in 1:length(ids)
        center = ids[i]
        for offset in -window:window
            j = i + offset
            if j != i && 1 ≤ j ≤ length(ids)
                context = ids[j]
                mat[center, context] += 1
                mat[context, center] += 1
            end
        end
    end
    mat ./= sum(mat; dims=2) .+ eps()
    return mat
end

"""
    project_encoding(mat::Matrix{Float32}, d_model::Int) -> Matrix{Float32}

Projects token co-occurrence vectors into embedding space.
"""
function project_encoding(mat::Matrix{Float32}, d_model::Int)
    W = randn(Float32, d_model, size(mat, 2)) * 0.01
    return mat * W'
end

end

