module Training

using Flux: onehotbatch, logitcrossentropy

"""
    mask_input(ids::Vector{Int}, mask_id::Int; prob::Float64=0.15)

Applies random masking to tokens. Returns (masked_input, target).
"""
function mask_input(ids::Vector{Int}, mask_id::Int; prob::Float64=0.15)
    masked = copy(ids)
    target = fill(0, length(ids))
    for i in eachindex(ids)
        if rand() < prob
            target[i] = ids[i]
            masked[i] = mask_id
        end
    end
    return masked, target
end

"""
    compute_loss(model, input, target, vocab_size)

Computes average cross-entropy loss over masked tokens only.
"""
function compute_loss(model, input, target, vocab_size)
    preds = model(input)
    loss = 0.0
    count = 0
    for i in eachindex(target)
        if target[i] != 0
            target_oh = onehotbatch([target[i]], 1:vocab_size)
            loss += logitcrossentropy(preds[:, i], target_oh)
            count += 1
        end
    end
    return loss / max(count, 1)
end

end

