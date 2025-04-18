module TransformerModel

using Flux: Dense, Chain, LayerNorm, Embedding, relu
using ..Attention: MultiheadAttention

"""
    struct TransformerBlock

Transformer encoder block with multi-head attention and feedforward layers.
"""
struct TransformerBlock
    mha::MultiheadAttention
    ff::Chain
    norm1::LayerNorm
    norm2::LayerNorm
end

function TransformerBlock(d_model::Int, heads::Int, ff_dim::Int)
    mha = MultiheadAttention(d_model, heads)
    ff = Chain(Dense(d_model, ff_dim, relu), Dense(ff_dim, d_model))
    return TransformerBlock(mha, ff, LayerNorm(d_model), LayerNorm(d_model))
end

function (block::TransformerBlock)(x::Matrix{Float32})
    x = block.norm1(x .+ block.mha(x))
    x = block.norm2(x .+ block.ff(x))
    return x
end

"""
    struct MaskedLM

Transformer-based masked language model with externally defined positional encodings.
"""
mutable struct MaskedLM
    embedding::Embedding
    pos_enc::Matrix{Float32}
    layers::Vector{TransformerBlock}
    output::Dense
end

"""
    MaskedLM(vocab_size, d_model, heads, ff_dim, layers, pos_enc)

Creates a masked language model with dynamic positional encoding.
"""
function MaskedLM(vocab_size::Int, d_model::Int, heads::Int, ff_dim::Int, layers::Int, pos_enc::Matrix{Float32})
    embedding = Embedding(vocab_size, d_model)
    blocks = [TransformerBlock(d_model, heads, ff_dim) for _ in 1:layers]
    output = Dense(d_model, vocab_size)
    return MaskedLM(embedding, pos_enc, blocks, output)
end

"""
    (m::MaskedLM)(ids)

Forward pass for a sequence of token IDs.
"""
function (m::MaskedLM)(ids::Vector{Int})
    x = m.embedding(ids) .+ transpose(m.pos_enc[ids, :])
    for block in m.layers
        x = block(x)
    end
    return m.output(x)
end

end

