import json
from pathlib import Path


class ActivationExtractor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_pairs(self, pairs_file):
        """Load the neutral pairs from JSON file"""
        with open(pairs_file) as f:
            data = json.load(f)

        pairs = data["pairs"]
        print(f"Loaded {len(pairs)} pairs from {pairs_file}")

        return pairs

    def extract_activations(self, text, layer_idx=-1):
        """
        Extract activations for a single text at a specific layer.

        Args:
            text: Input text string
            layer_idx: Which layer to extract from (-1 = last layer)

        Returns:
            Tensor of activations
        """
        import torch

        # Tokenize the text
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.model.device)

        # Forward pass - get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract from specified layer
        activations = outputs.hidden_states[
            layer_idx
        ]  # Shape: (1, seq_len, hidden_dim)

        # Average over sequence length
        activations = activations.mean(dim=1)  # Shape: (1, hidden_dim)

        return activations.squeeze(0)  # Shape: (hidden_dim,)

    def extract_pair_difference(self, stoic_text, neutral_text, layer_idx=-1):
        """
        Extract activation difference for a contrastive pair.

        Returns:
            Difference vector: stoic_activations - neutral_activations
        """
        stoic_act = self.extract_activations(stoic_text, layer_idx)
        neutral_act = self.extract_activations(neutral_text, layer_idx)
        difference = stoic_act - neutral_act

        return difference

    def compute_steering_vector(self, pairs_file, layer_idx=-1):
        """
        Compute steering vector by averaging differences across all pairs.

        Returns:
            Single steering vector (averaged across all pairs)
        """
        import torch

        pairs = self.load_pairs(pairs_file)
        differences = []

        print(f"\nComputing steering vector from {len(pairs)} pairs...")

        for i, pair in enumerate(pairs, 1):
            print(f"Processing pair {i}/{len(pairs)}...", end="\r")

            diff = self.extract_pair_difference(
                pair["stoic_text"], pair["neutral_text"], layer_idx
            )
            differences.append(diff)

        # Stack all differences and compute mean
        all_diffs = torch.stack(differences)  # Shape: (30, 2048)
        steering_vector = all_diffs.mean(dim=0)  # Shape: (2048,)

        print(f"\n✓ Steering vector computed!")
        print(f"  Shape: {steering_vector.shape}")

        return steering_vector

    def save_steering_vector(self, steering_vector, output_path):
        """Save steering vector to disk"""
        import torch
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(steering_vector, output_path)
        print(f"✓ Saved steering vector to {output_path}")
