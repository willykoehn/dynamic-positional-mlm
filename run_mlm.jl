using Pkg; Pkg.activate(".")
using Flux

include("src/DynamicPositionalMLM.jl")
using .DynamicPositionalMLM

# Load and tokenize the corpus
corpus = read("corpus.txt", String)
tokenizer = DynamicPositionalMLM.Tokenizer.TokenizerState(corpus)
token_ids = DynamicPositionalMLM.Tokenizer.tokenize(tokenizer, corpus)

# Build co-occurrence matrix and dynamic positional encoding
vocab_size = length(tokenizer.token2id)
cooccur = DynamicPositionalMLM.Cooccurrence.compute_cooccurrence_matrix(token_ids, vocab_size)
pos_enc = Float32.(DynamicPositionalMLM.Cooccurrence.project_encoding(cooccur, 64))

# Initialize the model
model = DynamicPositionalMLM.TransformerModel.MaskedLM(vocab_size, 64, 4, 128, 2, pos_enc)

# Create masked input and compute loss
masked_input, target_ids = DynamicPositionalMLM.Training.mask_input(token_ids, tokenizer.mask_id)
loss = DynamicPositionalMLM.Training.compute_loss(model, masked_input, target_ids, vocab_size)

println("Perplexity: ", exp(loss))

