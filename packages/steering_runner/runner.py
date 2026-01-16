import torch
from pathlib import Path
from .config import (
    LAYER_IDX,
    STEERING_LOCATION,
    COEFFICIENT,
    PROMPTS,
    TEMPERATURE,
    MAX_TOKENS,
    DO_SAMPLE,
)


class SteeringRunner:
    def __init__(
        self,
        file_path,
        model,
        tokenizer,
        layer=LAYER_IDX,
        steering_location=STEERING_LOCATION,
        coefficient=COEFFICIENT,
        prompts=PROMPTS,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        do_sample=DO_SAMPLE,
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
        self.hook_setup = False
        self.author = str(self.file).split("/")[-1].split("_")[0]

    def _load_steering_vector(self):
        self.steering_vector = torch.load(
            self.file, map_location=self.steering_location, weights_only=True
        )

    def _steering_hook(self, module, input, output):
        return output + self.coefficient * self.steering_vector

    def _register_hook(self):
        self.model.model.layers[self.layer_idx].mlp.register_forward_hook(
            self._steering_hook
        )

    def run_model_with_hook(self):
        if not self.hook_setup:
            self._load_steering_vector()
            self._register_hook()
            self.hook_setup = True

        for prompt in self.prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 70)

            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
            )
            print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
