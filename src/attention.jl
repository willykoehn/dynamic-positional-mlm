module Attention

using Flux: Dense, softmax

"""
    struct MultiheadAttention

Basic multi-head attention with internal projection layers.
"""
struct MultiheadAttention
    w_q::Dense
    w_k::Dense
    w_v::Dense
    w_o::Dense
    num_heads::Int
    head_dim::Int
end

"""
    MultiheadAttention(d_model::Int, num_heads::Int)

Construct a MultiheadAttention layer with model dimension d_model split into num_heads.
"""
function MultiheadAttention(d_model::Int, num_heads::Int)
    head_dim = div(d_model, num_heads)
    return MultiheadAttention(
        Dense(d_model, d_model),  # W_q
        Dense(d_model, d_model),  # W_k
        Dense(d_model, d_model),  # W_v
        Dense(d_model, d_model),  # W_o
        num_heads,
        head_dim
    )
end

"""
    (m::MultiheadAttention)(x::Matrix{Float32})

Apply multi-head attention to input tensor x of shape (d_model, seq_len).
"""
function (m::MultiheadAttention)(x::Matrix{Float32})
    # x: (d_model, seq_len)
    d_model, seq_len = size(x)
    h = m.num_heads
    d_k = m.head_dim

    # Project inputs
    Q = m.w_q(x)  # (d_model, seq_len)
    K = m.w_k(x)
    V = m.w_v(x)

    # Reshape: (d_model, seq_len) → (d_k, h, seq_len)
    function split_heads(t)
        reshape(t, d_k, h, seq_len)
    end

    Qh = split_heads(Q)
    Kh = split_heads(K)
    Vh = split_heads(V)

    # Output tensor: (d_k, h, seq_len)
    output = Array{Float32}(undef, d_k, h, seq_len)

    for i in 1:h
        Qi = Qh[:, i, :]               # (d_k, seq_len)
        Ki = Kh[:, i, :]               # (d_k, seq_len)
        Vi = Vh[:, i, :]               # (d_k, seq_len)

        scores = Qi' * Ki ./ sqrt(d_k) # (seq_len, seq_len)
        weights = softmax(scores; dims=2)
        output[:, i, :] = (Vi * weights')  # (d_k, seq_len)
    end

    # Reshape back: (d_k, h, seq_len) → (d_model, seq_len)
    concat = reshape(output, d_k * h, seq_len)
    return m.w_o(concat)  # (d_model, seq_len)
end

end
