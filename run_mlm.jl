using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using StatsBase: countmap
using Random
using BSON: @save, @load

corpus = read("corpus.txt", String)

# Define the minimal tokenizer and vocabulary logic
function build_vocab(corpus)
    token_counts = countmap(split(corpus))
    vocab = sort(collect(keys(token_counts)), by=x->token_counts[x], rev=true)
    token2id = Dict(token => id for (id, token) in enumerate(vocab))
    id2token = Dict(id => token for (id, token) in enumerate(vocab))
    return token2id, id2token
end

function tokenize(text, token2id)
    return [get(token2id, token, token2id["<unk>"]) for token in split(text)]
end

function detokenize(ids, id2token)
    return join([get(id2token, id, "<unk>") for id in ids], " ")
end

# Define the dynamic positional encoding
function dynamic_positional_encoding(seq_len, vocab_size, token_cooccurrence)
    pos_encoding = zeros(Float32, seq_len, vocab_size)
    for i in 1:seq_len
        for j in 1:vocab_size
            pos_encoding[i, j] = token_cooccurrence[j][i]
        end
    end
    return pos_encoding
end

# Define the transformer model
struct Transformer
    embedding
    pos_encoding
    transformer
    output
end

function Transformer(vocab_size, embed_size, num_heads, num_layers, ff_size, token_cooccurrence)
    embedding = Embedding(vocab_size, embed_size)
    pos_encoding = dynamic_positional_encoding(512, vocab_size, token_cooccurrence)
    transformer = Chain([TransformerBlock(embed_size, num_heads, ff_size) for _ in 1:num_layers]...)
    output = Dense(embed_size, vocab_size)
    return Transformer(embedding, pos_encoding, transformer, output)
end

struct TransformerBlock
    attention
    feedforward
    layernorm1
    layernorm2
end

function TransformerBlock(embed_size, num_heads, ff_size)
    attention = MultiheadAttention(embed_size, num_heads)
    feedforward = Chain(Dense(embed_size, ff_size, relu), Dense(ff_size, embed_size))
    layernorm1 = LayerNorm(embed_size)
    layernorm2 = LayerNorm(embed_size)
    return TransformerBlock(attention, feedforward, layernorm1, layernorm2)
end

function (t::Transformer)(x)
    x = t.embedding(x)
    x = x .+ t.pos_encoding[1:size(x, 1), :]
    for block in t.transformer
        x = block.layernorm1(x .+ block.attention(x))
        x = block.layernorm2(x .+ block.feedforward(x))
    end
    return t.output(x)
end

# Define the masked language model
function masked_language_model(model, input_ids, mask_token_id)
    input_ids = copy(input_ids)
    mask_indices = findall(x -> x != mask_token_id, input_ids)
    masked_input_ids = copy(input_ids)
    masked_input_ids[mask_indices] .= mask_token_id
    masked_input_ids = onehotbatch(masked_input_ids, 1:vocab_size)
    output = model(masked_input_ids)
    return output, input_ids[mask_indices]
end

# Define the training loop
function train!(model, data, opt, mask_token_id)
    for (input_ids, _) in data
        output, target_ids = masked_language_model(model, input_ids, mask_token_id)
        loss = logitcrossentropy(output, onehotbatch(target_ids, 1:vocab_size))
        Flux.back!(loss)
        opt()
    end
end

# Define the evaluation loop
function evaluate(model, data, mask_token_id)
    total_loss = 0.0
    total_tokens = 0
    for (input_ids, _) in data
        output, target_ids = masked_language_model(model, input_ids, mask_token_id)
        loss = logitcrossentropy(output, onehotbatch(target_ids, 1:vocab_size))
        total_loss += loss * length(target_ids)
        total_tokens += length(target_ids)
    end
    return exp(total_loss / total_tokens)
end

# Define the data preparation
function prepare_data(corpus, token2id, batch_size, seq_len)
    token_ids = tokenize(corpus, token2id)
    num_batches = length(token_ids) ÷ (batch_size * seq_len)
    token_ids = token_ids[1:num_batches * batch_size * seq_len]
    data = [(token_ids[(i-1)*seq_len+1:i*seq_len], nothing) for i in 1:batch_size:num_batches*batch_size]
    return data
end

# Define the token co-occurrence frequency
function token_cooccurrence(corpus, token2id)
    token_ids = tokenize(corpus, token2id)
    cooccurrence = Dict(id => zeros(Int, 512) for id in 1:length(token2id))
    for i in 1:length(token_ids)-1
        cooccurrence[token_ids[i]][token_ids[i+1]] += 1
    end
    return cooccurrence
end

# Example usage
corpus = "This is a test corpus. This corpus is used to test the masked language model."
token2id, id2token = build_vocab(corpus)
vocab_size = length(token2id)
mask_token_id = token2id["<mask>"]
token_cooccurrence = token_cooccurrence(corpus, token2id)

model = Transformer(vocab_size, 128, 4, 2, 256, token_cooccurrence)
opt = ADAM(0.001)

train_data = prepare_data(corpus, token2id, 2, 10)
@epochs 10 train!(model, train_data, opt, mask_token_id)

eval_data = prepare_data(corpus, token2id, 2, 10)
perplexity = evaluate(model, eval_data, mask_token_id)
println("Perplexity: $perplexity")
