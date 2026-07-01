import json
from pathlib import Path
from contextlib import contextmanager
import torch


@contextmanager
def _hooks(handles):
    """Collect hook handles and guarantee removal on exit."""
    try:
        yield handles
    finally:
        for h in handles:
            h.remove()


class ActivationExtractor:
    """
    Extracts CAA steering vectors at the SAME site they are injected:
    the MLP output of a decoder block.

    SteeringRunner injects at `model.model.layers[L].mlp`, so we extract
    there too. This is what makes "layer L is where the Stoic direction
    lives" a meaningful claim — the old code extracted at the final layer
    (hidden_states[-1]) and injected at layer 8, so the layer sweep was
    really "best injection site for a final-layer vector," not "best layer."
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_pairs(self, pairs_file):
        with open(pairs_file) as f:
            data = json.load(f)
        pairs = data["pairs"] if isinstance(data, dict) else data
        print(f"Loaded {len(pairs)} pairs from {pairs_file}")
        return pairs

    def _mlp_module(self, layer_idx):
        return self.model.model.layers[layer_idx].mlp

    def extract_activations_multi(self, text, layers):
        """
        Single forward pass. Capture the MLP output at each requested layer,
        averaged over sequence length.

        Returns: dict {layer_idx: tensor(hidden_dim,)}
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)

        captured = {}

        def make_hook(L):
            def hook_fn(module, inp, out):
                # LLaMA MLP returns a plain tensor: (batch, seq_len, hidden_dim)
                captured[L] = out.detach()

            return hook_fn

        handles = []
        with _hooks(handles):
            for L in layers:
                handles.append(self._mlp_module(L).register_forward_hook(make_hook(L)))
            with torch.no_grad():
                self.model(**inputs)

        # Mean over sequence length -> (hidden_dim,)
        return {L: captured[L].mean(dim=1).squeeze(0) for L in layers}

    def extract_activations(self, text, layer_idx):
        """Single-layer convenience wrapper."""
        return self.extract_activations_multi(text, [layer_idx])[layer_idx]

    def compute_layered_steering_vectors(self, pairs_file, layers):
        """
        Compute a steering vector at EACH layer in `layers`, from the same
        contrastive pairs. One forward pass per text (not per text * layer),
        so building all 7 sweep layers costs the same as building one.

        Returns: dict {layer_idx: steering_vector(hidden_dim,)}
        """
        pairs = self.load_pairs(pairs_file)
        sums = {L: None for L in layers}

        print(
            f"\nComputing steering vectors at layers {layers} "
            f"from {len(pairs)} pairs..."
        )
        for i, pair in enumerate(pairs, 1):
            print(f"Processing pair {i}/{len(pairs)}...", end="\r")
            stoic = self.extract_activations_multi(pair["stoic_text"], layers)
            neutral = self.extract_activations_multi(pair["neutral_text"], layers)
            for L in layers:
                diff = stoic[L] - neutral[L]
                sums[L] = diff if sums[L] is None else sums[L] + diff

        vectors = {L: (sums[L] / len(pairs)) for L in layers}
        print(f"\n✓ Steering vectors computed at {len(layers)} layers")
        for L in layers:
            print(f"  layer {L}: shape {tuple(vectors[L].shape)}")
        return vectors

    def compute_steering_vector(self, pairs_file, layer_idx=None):
        """
        Single-layer steering vector. layer_idx is REQUIRED — the old
        implicit `-1` (final layer) default is what caused extraction and
        injection to happen at different layers. Use
        compute_layered_steering_vectors() to build all sweep layers at once.
        """
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required. Extract at the same layer you inject "
                "(SteeringRunner injects at layers[L].mlp)."
            )
        return self.compute_layered_steering_vectors(pairs_file, [layer_idx])[layer_idx]

    def save_steering_vectors(self, vectors, output_path):
        """Save a {layer_idx: vector} dict to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vectors, output_path)
        print(f"✓ Saved vectors for layers {sorted(vectors)} to {output_path}")
