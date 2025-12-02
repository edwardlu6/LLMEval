# model.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import os
import torch


# Optional but recommended: avoid tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------- Base API ---------
class ModelClient:
    def generate(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

# --------- HF (local) backend ---------
# Lazy import so non-HF providers still work without transformers installed.
def _import_hf():
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    import torch
    return AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, torch

# simple in-process cache so we don’t reload weights for every call
_MODEL_CACHE: dict[Tuple, Any] = {}
_TOKENIZER_CACHE: dict[str, Any] = {}

class HFLocalClient(ModelClient):
    def __init__(self, spec: dict):
        """
        Expected spec keys (examples):
          id: "hf_local" | "transformers" | "huggingface"
          model: "deepseek-ai/deepseek-llm-7b-chat"
          trust_remote_code: bool
          device_map: "auto" | "cuda" | "cpu"
          load_in_4bit: bool
          bnb_4bit_compute_dtype: "float16" | "bfloat16"
          bnb_4bit_quant_type: "nf4" | "fp4"
          bnb_4bit_use_double_quant: bool
          use_chat_template: bool
          max_new_tokens: int
          temperature: float
          top_p: float
          token: hf_token (optional if model gated)
        """
        self.spec = spec
        self.model_name = spec.get("model")
        if not self.model_name:
            raise ValueError("HFLocalClient requires spec['model'].")

        (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, torch) = _import_hf()

        trust_remote_code = bool(spec.get("trust_remote_code", True))
        device_map = spec.get("device_map", "auto")

        # Quantization config (optional)
        quant_cfg = None
        if spec.get("load_in_4bit", False):
            compute = str(spec.get("bnb_4bit_compute_dtype", "float16")).lower()
            compute_dtype = torch.float16 if compute in ("fp16", "float16") else torch.bfloat16
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=spec.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(spec.get("bnb_4bit_use_double_quant", True)),
            )

        # Cache key: model + minimal loading knobs
        key = (
            self.model_name,
            bool(spec.get("load_in_4bit", False)),
            str(spec.get("bnb_4bit_quant_type", "nf4")),
            str(spec.get("bnb_4bit_compute_dtype", "float16")),
            bool(spec.get("bnb_4bit_use_double_quant", True)),
            str(device_map),
            bool(trust_remote_code),
        )

        # Tokenizer (cached by model name)
        tok = _TOKENIZER_CACHE.get(self.model_name)
        if tok is None:
            tok = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=trust_remote_code,
                token=spec.get("token", None),
            )
            _TOKENIZER_CACHE[self.model_name] = tok
        self.tokenizer = tok

        # Model (cached by key)
        mdl = _MODEL_CACHE.get(key)
        if mdl is None:
            load_kwargs = {
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": True,
                "token": spec.get("token", None),
            }
            if quant_cfg is not None:
                load_kwargs["quantization_config"] = quant_cfg

            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
            _MODEL_CACHE[key] = mdl
        self.model = mdl

        # Generation defaults
        self.max_new_tokens = int(spec.get("max_new_tokens", 128))
        self.temperature = float(spec.get("temperature", 0.2))
        self.top_p = float(spec.get("top_p", 0.9))
        self.use_chat_template = bool(spec.get("use_chat_template", True))

    def _format(self, prompt: str) -> str:
        # Use only the user message. No system message.
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # appends assistant header if the template has one
            )
        return prompt


    def _postprocess(self, text: str) -> str:
        """Strip echoed chat preface and trim."""
        # Cut at any subsequent role headers that sometimes appear
        for s in self.spec.get("stop_strings", ["\nUser:", "\n\nUser:", "\nSystem:", "\n\nSystem:"]):
            if s in text:
                text = text.split(s, 1)[0]

        # Keep only assistant part if template echoed "Assistant:"
        # use rsplit to be robust if the word appears earlier
        if "Assistant:" in text:
            text = text.rsplit("Assistant:", 1)[-1]

        return text.strip()

    def generate(self, prompt: str):

      # Avoid padding warnings
      try:
          self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
      except Exception:
          pass

      # 1) Build chat-formatted prompt
      sys_msg = self.spec.get("system_prompt", "Answer concisely.")
      if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
          messages = [
              {"role": "system", "content": sys_msg},
              {"role": "user", "content": prompt},
          ]
          text_in = self.tokenizer.apply_chat_template(
              messages, tokenize=False, add_generation_prompt=True
          )
      else:
          text_in = prompt

      # 2) Tokenize and move to model device
      inputs = self.tokenizer(text_in, return_tensors="pt")
      device = next(self.model.parameters()).device
      inputs = {k: v.to(device) for k, v in inputs.items()}
      prompt_len = inputs["input_ids"].shape[1]

      # 3) Generate
      with torch.inference_mode():
          out = self.model.generate(
              **inputs,
              max_new_tokens=int(self.max_new_tokens),
              temperature=float(self.temperature),
              top_p=float(self.top_p),
              do_sample=float(self.temperature) > 0,
          )

      # 4) Decode ONLY the newly generated tokens (after the prompt)
      gen_only_ids = out[0][prompt_len:]
      text_out = self.tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()

      # Optional: cut on simple stop strings if your pipeline benefits
      for s in self.spec.get("stop_strings", []):
          if s in text_out:
              text_out = text_out.split(s, 1)[0].strip()

      return {"text": text_out, "usage": {}}


# --------- Factory ---------
def make_client(spec: dict) -> ModelClient:
    provider = (spec.get("id") or spec.get("backend") or spec.get("provider") or "").lower()
    if provider in {"hf_local", "hf-local", "transformers", "huggingface"}:
        return HFLocalClient(spec)
    

    if provider in {"mixtral", "mistral", "mix"}: 
        return HFLocalClient(spec)

    if provider.startswith("qwen"): 
        return HFLocalClient(spec)

    # TODO: add real providers here (OpenAI, Together, DeepSeek API, etc.)
    # Fallback (kept for debugging clarity)
    from typing import Dict as _Dict
    class _Echo(ModelClient):
        def __init__(self, model_id: str): self.model_id = model_id
        def generate(self, prompt: str) -> _Dict:
            return {"text": f"[{provider or 'echo'}] {prompt[:120]} ...", "usage": {}}
    return _Echo(provider or "echo")

