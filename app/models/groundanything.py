import json
from pathlib import Path

from PIL import Image

from .groundingpi import GroundingPI
from app.core.registry import register_model


@register_model("groundanything_vlm")
class GroundAnythingVLM(GroundingPI):
    pass


@register_model("groundanything")
class GroundAnything(GroundingPI):
    def _load_transformers(self):
        import torch
        import transformers
        from accelerate import init_empty_weights
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        from .groundanything_impl.hybrid import MASK_TOKEN
        from .groundanything_impl.vlm import GAMQwen3DLM

        if transformers.__version__ != "5.7.0":
            raise RuntimeError(
                "GroundAnything requires the bundled Transformers 5.7.0 "
                "fork. Install the groundingpi extra."
            )
        model_path = self.params.get(
            "model_path", "GroundingPI/GroundAnything"
        )
        if not Path(model_path).is_dir():
            model_path = snapshot_download(repo_id=model_path)
        self.device = self.params.get("device", "cuda")
        dtype = getattr(torch, self.params.get("dtype", "bfloat16"))
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        config = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        config.auto_map = json.loads(
            (Path(model_path) / "config.json").read_text()
        )["auto_map"]
        tokenizer = self.processor.tokenizer
        mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        if tokenizer.encode(MASK_TOKEN, add_special_tokens=False) != [mask_id]:
            raise ValueError(
                "GroundAnything checkpoint needs an atomic mask token"
            )
        config.text_config.vocab_size = len(tokenizer)
        with init_empty_weights():
            native = transformers.AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                experts_implementation="eager",
                attn_implementation=self.params.get(
                    "attn_implementation", "sdpa"
                ),
            )
            model = GAMQwen3DLM(
                native,
                mask_token_id=mask_id,
                im_end_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                block_size=32,
            )
        state = load_file(str(Path(model_path) / "model.safetensors"))
        model.load_state_dict(state, strict=True, assign=True)
        del state
        self.model = model.eval().to(device=self.device, dtype=dtype)

    def _generate(self, image, prompt, params):
        import torch

        from .groundanything_impl.inputs import prepare_qwen3_inputs

        inputs = prepare_qwen3_inputs(
            self.processor,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": Image.fromarray(image[:, :, ::-1]),
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        max_tokens = params.get(
            "max_new_tokens", self.params.get("max_new_tokens", 4096)
        )
        if inputs["input_ids"].shape[1] + max_tokens > self.params.get(
            "max_model_len", 16384
        ):
            raise ValueError(
                "Prompt plus max_new_tokens exceeds max_model_len"
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        eos = self.model.im_end_token_id
        with torch.inference_mode():
            generated = self.model.speculative_generate(
                **inputs,
                max_new_tokens=max_tokens,
                inference_block_size=32,
                stop_token_ids=(eos,),
                no_repeat_ngram_size=0,
            )[0].tolist()
        stopped = bool(generated and generated[-1] == eos)
        if stopped:
            generated.pop()
        content = self.processor.tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop" if stopped else "length",
                }
            ]
        }
