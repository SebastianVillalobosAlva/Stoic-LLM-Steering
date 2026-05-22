import torch
from pathlib import Path
from stoic_llm.config import (
    LAYER_IDX,
    COEFFICIENT,
    DEFAULT_PROMPTS,
    TEMPERATURE,
    MAX_TOKENS,
    DEVICE,
)


class SteeringRunner:
    def __init__(
        self,
        file_path,
        model,
        tokenizer,
        layer=LAYER_IDX,
        steering_location=DEVICE,
        coefficient=COEFFICIENT,
        prompts=DEFAULT_PROMPTS,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        do_sample=True,
    ) -> None:
        self.layer_idx = layer
        self.steering_location = steering_location
        self.coefficient = coefficient
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.file = file_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.do_sample = do_sample
        self.author = str(self.file).split("/")[-1].split("_")[0]

        # Hook state
        self._hook_handle = None
        self.steering_vector = None

    def _load_steering_vector(self):
        self.steering_vector = torch.load(
            self.file, map_location=self.steering_location, weights_only=True
        )

    def _steering_hook(self, module, input, output):
        return output + self.coefficient * self.steering_vector

    def _register_hook(self):
        # Remove existing hook first to prevent stacking
        self._remove_hook()

        self._hook_handle = self.model.model.layers[
            self.layer_idx
        ].mlp.register_forward_hook(self._steering_hook)

    def _remove_hook(self):
        """Remove the current steering hook if one exists."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def set_coefficient(self, coefficient):
        """Update steering strength. Re-registers hook with new coefficient."""
        self.coefficient = coefficient
        if self._hook_handle is not None:
            self._register_hook()

    def set_layer(self, layer_idx):
        """Switch which layer the steering vector is applied to."""
        self.layer_idx = layer_idx
        if self._hook_handle is not None:
            self._register_hook()

    def load_author(self, file_path):
        """Switch to a different author's steering vector."""
        self.cleanup()
        self.file = file_path
        self.author = str(self.file).split("/")[-1].split("_")[0]
        self.steering_vector = None

    def run_model_with_hook(self, return_output=False, **generate_kwargs):
        if self.steering_vector is None:
            self._load_steering_vector()
        if self._hook_handle is None:
            self._register_hook()

        results = []
        for prompt in self.prompts:
            if not return_output:
                print(f"\nPrompt: {prompt}")
                print("-" * 70)

            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                **generate_kwargs,
            )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if return_output:
                results.append(generated)
            else:
                print(generated)

        return results if return_output else None

    def cleanup(self):
        """Remove hook and clear state. Call when done steering."""
        self._remove_hook()
        self.steering_vector = None

    def __del__(self):
        self.cleanup()
