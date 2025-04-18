module Tokenizer

using StatsBase

"""
    struct TokenizerState

Minimal whitespace tokenizer with special token handling.
"""
mutable struct TokenizerState
    token2id::Dict{String, Int}
    id2token::Dict{Int, String}
    pad_id::Int
    mask_id::Int
    unk_id::Int
end

"""
    TokenizerState(corpus::String)

Creates vocabulary from the corpus and initializes special tokens.
"""
function TokenizerState(corpus::String)
    specials = ["<pad>", "<mask>", "<unk>"]
    tokens = split(corpus)
    vocab = union(tokens, specials)
    token2id = Dict(tok => i for (i, tok) in enumerate(vocab))
    id2token = Dict(i => tok for (tok, i) in token2id)
    return TokenizerState(
        token2id,
        id2token,
        token2id["<pad>"],
        token2id["<mask>"],
        token2id["<unk>"]
    )
end

"""
    tokenize(t::TokenizerState, text::String) -> Vector{Int}

Converts text to token IDs using tokenizer vocabulary.
"""
function tokenize(t::TokenizerState, text::String)
    return [get(t.token2id, tok, t.unk_id) for tok in split(text)]
end

"""
    detokenize(t::TokenizerState, ids::Vector{Int}) -> String

Converts token IDs back to readable text.
"""
function detokenize(t::TokenizerState, ids::Vector{Int})
    return join([get(t.id2token, id, "<unk>") for id in ids], " ")
end

end

