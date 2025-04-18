# Dynamic Positional Masked Language Model (Julia + Flux)

This project implements a masked language model (MLM) in Julia using Flux.jl, with a custom transformer architecture and dynamic positional encoding derived from token co-occurrence statistics. It is a fully self-contained implementation that avoids external tokenization or data pipeline libraries, and is structured for modularity, clarity, and extensibility.

## Features

- Minimal whitespace tokenizer with special token support
- Vocabulary and token ID mapping built directly from the corpus
- Dynamic positional encodings based on token co-occurrence frequency
- Manually implemented multi-head attention layer using Flux primitives
- Layer-normalized transformer blocks with custom forward logic
- Forward-only training pipeline that computes cross-entropy loss and perplexity
- Fully modular design with documented components and isolated submodules

## Technologies Used

- [Flux.jl](https://fluxml.ai/) – deep learning framework for Julia
- [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl) – token frequency and co-occurrence stats
- [OneHotArrays.jl](https://github.com/FluxML/OneHotArrays.jl) – one-hot vector representation
- [BSON.jl](https://github.com/JuliaIO/BSON.jl) – optional model serialization (not required to run)

## Usage

Clone the repository and run the model with:

```bash
julia --project=. run_mlm.jl
```

Ensure you have a `corpus.txt` file in the root directory containing a whitespace-separated synthetic or natural language corpus.

To ensure reproducibility, you can optionally fix the random seed at the top of `run_mlm.jl` using:

```julia
using Random
Random.seed!(42)
```

## Output

On each run, the script tokenizes the input corpus, applies masking, computes predictions via the transformer model, and prints the perplexity of the predicted output relative to the ground truth.

```
Perplexity: 27.64
```

Perplexity will vary depending on corpus size, masking randomness, and model initialization.

## Project Structure

- `src/tokenizer.jl` – minimal whitespace tokenizer and vocabulary logic
- `src/cooccurrence.jl` – co-occurrence matrix and projection to positional encodings
- `src/attention.jl` – custom multi-head attention layer
- `src/transformer.jl` – transformer block and masked language model struct
- `src/training.jl` – masking and loss computation
- `run_mlm.jl` – entry point for the end-to-end model

## License

This project is released under the MIT License.
