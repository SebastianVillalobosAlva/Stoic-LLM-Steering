def make_stoic_token_metric(tokenizer, stoic_words, neutral_words):
    """Logit difference between Stoic-associated and neutral tokens
    at the last position. Pass to ModelLens discover_circuit/patching."""

    def first_id(w):
        return tokenizer.encode(" " + w.lstrip(), add_special_tokens=False)[0]

    stoic_ids = [first_id(w) for w in stoic_words]
    neutral_ids = [first_id(w) for w in neutral_words]

    def metric(output):
        logits = output.logits if hasattr(output, "logits") else output
        last = logits[:, -1, :]
        return (last[:, stoic_ids].mean() - last[:, neutral_ids].mean()).item()

    return metric


def make_steering_projection_metric(steering_vector, layer_idx):
    """Project the final-token hidden state at `layer_idx` onto the steering
    direction — measures movement *along the Stoic axis*, not toward specific
    tokens. Requires the forward pass to expose hidden states."""
    v = steering_vector / (steering_vector.norm() + 1e-10)

    def metric(output):
        hs = output.hidden_states[layer_idx + 1][:, -1, :]  # (batch, hidden)
        return (hs @ v.to(hs.dtype)).mean().item()

    return metric
