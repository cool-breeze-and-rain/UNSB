import json
import os

import torch
import torch.nn.functional as F


def load_prompt_map(prompt_path):
    if not prompt_path:
        return {}
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file must contain a JSON object: {prompt_path}")
    return data


def infer_prompt_key(dataroot, prompt_map, explicit_key=""):
    if explicit_key:
        return explicit_key
    dataroot_upper = os.path.abspath(dataroot).upper()
    for key in prompt_map:
        if key.upper() in dataroot_upper:
            return key
    return ""


def resolve_prompt_condition(opt):
    if not getattr(opt, "use_prompt_condition", False):
        return "", ""
    prompt_map = load_prompt_map(opt.prompt_path)
    prompt_key = infer_prompt_key(opt.dataroot, prompt_map, getattr(opt, "prompt_key", ""))
    if not prompt_key:
        available = ", ".join(sorted(prompt_map.keys())) or "<empty>"
        raise ValueError(
            f"Unable to infer prompt key from dataroot '{opt.dataroot}'. "
            f"Use --prompt_key explicitly. Available keys: {available}"
        )
    if prompt_key not in prompt_map:
        available = ", ".join(sorted(prompt_map.keys())) or "<empty>"
        raise KeyError(f"Prompt key '{prompt_key}' not found in {opt.prompt_path}. Available keys: {available}")
    prompt_text = str(prompt_map[prompt_key]).strip()
    if not prompt_text:
        raise ValueError(f"Prompt text for key '{prompt_key}' is empty.")
    return prompt_key, prompt_text


def encode_prompt_text(prompt_text, embed_dim, device, dtype=torch.float32):
    prompt_bytes = prompt_text.encode("utf-8")
    if not prompt_bytes:
        return torch.zeros(embed_dim, device=device, dtype=dtype)

    values = torch.tensor(list(prompt_bytes), device=device, dtype=dtype)
    values = values / 255.0
    positions = torch.arange(1, values.numel() + 1, device=device, dtype=dtype)
    freqs = torch.arange(1, embed_dim + 1, device=device, dtype=dtype)
    phase = positions[:, None] * freqs[None, :] / max(values.numel(), 1)
    sin_term = torch.sin(phase) * values[:, None]
    cos_term = torch.cos(phase) * values[:, None]
    embedding = sin_term.mean(dim=0) + cos_term.mean(dim=0)
    embedding = F.normalize(embedding.unsqueeze(0), dim=1).squeeze(0)
    return embedding
