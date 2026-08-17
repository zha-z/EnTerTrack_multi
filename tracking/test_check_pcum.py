import torch

checkpoint_path = "/data/zjy/EnTeR-Track-main/output/pcum_ablation_current/checkpoints/train/entertrack/pcum_ablation_current_full/EnTeRTrack_ep0040.pth.tar"

ckpt = torch.load(checkpoint_path, map_location="cpu")

if isinstance(ckpt, dict):
    if "net" in ckpt:
        state = ckpt["net"]
    elif "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
else:
    raise TypeError(type(ckpt))

pcum_keys = [
    key for key in state.keys()
    if "pcum" in key.lower()
    or "prompt" in key.lower()
    or "fusion" in key.lower()
    or "align" in key.lower()
]

print("Total keys:", len(state))
print("Potential PCUM keys:", len(pcum_keys))

for key in pcum_keys[:100]:
    value = state[key]
    shape = tuple(value.shape) if hasattr(value, "shape") else None
    print(key, shape)

import torch

def tensor_stats(name, tensor):
    tensor = tensor.detach().float().cpu()

    print(f"\n{name}")
    print("shape:", tuple(tensor.shape))
    print("mean:", tensor.mean().item())
    print("std:", tensor.std(unbiased=False).item())
    print("min:", tensor.min().item())
    print("max:", tensor.max().item())
    print("abs mean:", tensor.abs().mean().item())
    print("abs max:", tensor.abs().max().item())
    print("norm:", tensor.norm().item())


keys_to_check = [
    "pcum.fusion.residual_scale",
    "pcum.fusion.prompt_proj.weight",
    "pcum.fusion.prompt_proj.bias",
    "pcum.fusion.gate.weight",
    "pcum.fusion.gate.bias",
    "pcum.fusion.film.weight",
    "pcum.fusion.film.bias",
]

for key in keys_to_check:
    if key in state:
        tensor_stats(key, state[key])
    else:
        print("\nMISSING:", key)


# film 输出为384维，如果代码中确实按前后192维拆成gamma/beta，
# 可以额外观察两部分bias。
film_bias_key = "pcum.fusion.film.bias"

if film_bias_key in state:
    bias = state[film_bias_key].detach().float().cpu()

    if bias.numel() == 384:
        gamma_bias = bias[:192]
        beta_bias = bias[192:]

        tensor_stats("film gamma bias half", gamma_bias)
        tensor_stats("film beta bias half", beta_bias)