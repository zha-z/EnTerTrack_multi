import torch
from torch import nn
import torch.nn.functional as F


REMOTE_AGGREGATION_MODES = (
    "mean",
    "confidence_softmax",
    "confidence_sigmoid",
)


def validate_remote_aggregation(mode):
    mode = str(mode).lower()
    if mode not in REMOTE_AGGREGATION_MODES:
        raise ValueError("Unsupported remote aggregation: %s" % mode)
    return mode


def _as_layer_list(tokens):
    if isinstance(tokens, (list, tuple)):
        return list(tokens)
    return [tokens]


def _state_confidence(states, batch_size, device, dtype):
    if states is None:
        return torch.ones(batch_size, 1, device=device, dtype=dtype)

    if isinstance(states, torch.Tensor):
        conf = states.to(device=device, dtype=dtype)
        if conf.dim() == 1:
            conf = conf.view(-1, 1)
        return conf[:, :1].clamp(0.0, 1.0)

    if isinstance(states, dict):
        for key in ("confidence", "score", "max_score"):
            if key in states:
                conf = states[key]
                if not torch.is_tensor(conf):
                    conf = torch.as_tensor(conf, device=device, dtype=dtype)
                conf = conf.to(device=device, dtype=dtype)
                if conf.dim() == 0:
                    conf = conf.repeat(batch_size)
                return conf.view(batch_size, -1)[:, :1].clamp(0.0, 1.0)

    return torch.ones(batch_size, 1, device=device, dtype=dtype)


def _get_prompt_from_features(features):
    if features is None:
        return None
    if torch.is_tensor(features):
        return features
    for key in ("prompt", "local_prompt", "aligned_prompt", "search_tokens", "search"):
        if key in features and torch.is_tensor(features[key]):
            return features[key]
    return None


def build_pseudo_remote_prompts(local_prompt=None, features=None, noise_std=0.02,
                                use_batch_roll=True):
    """Construct pseudo remote prompts without real communication.

    If a batch has multiple samples, another sample is used as the pseudo remote
    view. For single-sample batches, a lightly perturbed copy is used as the
    fallback remote prompt.
    """
    if local_prompt is None:
        local_prompt = _get_prompt_from_features(features)
    if local_prompt is None:
        raise ValueError("local_prompt or prompt-like features are required")
    if local_prompt.dim() not in (3, 4):
        raise ValueError("local_prompt must have shape [B, M, C] or [B, V, M, C]")

    prompt = local_prompt
    if prompt.dim() == 4:
        if prompt.shape[1] > 1:
            remote = prompt[:, 1]
            local = prompt[:, 0]
            if remote.shape != local.shape:
                remote = F.interpolate(
                    remote.transpose(1, 2),
                    size=local.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            return remote
        prompt = prompt[:, 0]

    B = prompt.shape[0]
    if use_batch_roll and B > 1:
        remote = torch.roll(prompt, shifts=1, dims=0)
    else:
        remote = prompt.clone()

    if noise_std and noise_std > 0:
        remote = remote + torch.randn_like(remote) * float(noise_std)

    return remote


class PromptConsistencyLoss(nn.Module):
    """Cosine consistency loss between local and pseudo remote prompts."""

    def __init__(self, loss_type="cosine", stop_gradient_teacher=True):
        super().__init__()
        if loss_type != "cosine":
            raise ValueError("Unsupported prompt consistency loss: %s" % loss_type)
        self.loss_type = loss_type
        self.stop_gradient_teacher = bool(stop_gradient_teacher)

    def forward(self, local_prompt, remote_prompt):
        if local_prompt is None or remote_prompt is None:
            raise ValueError("local_prompt and remote_prompt are required")
        if isinstance(remote_prompt, (list, tuple)):
            remote_prompt = torch.stack(remote_prompt, dim=1)
        if remote_prompt.dim() == 4:
            remote_prompt = remote_prompt.mean(dim=1)
        if local_prompt.dim() != 3 or remote_prompt.dim() != 3:
            raise ValueError("prompts must have shape [B, M, C]")

        if remote_prompt.shape[1] != local_prompt.shape[1]:
            remote_prompt = F.interpolate(
                remote_prompt.transpose(1, 2),
                size=local_prompt.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        if self.stop_gradient_teacher:
            remote_prompt = remote_prompt.detach()

        local = F.normalize(local_prompt, dim=-1)
        remote = F.normalize(remote_prompt, dim=-1)
        return 1.0 - F.cosine_similarity(local, remote, dim=-1).mean()


class SaliencyTokenSelector(nn.Module):
    """Select top-k salient tokens from search/template features."""

    def __init__(self, topk=16, source="feature_norm"):
        super().__init__()
        if source not in ("attention_score", "feature_norm", "confidence_score"):
            raise ValueError("Unsupported saliency source: %s" % source)
        self.topk = int(topk)
        self.source = source

    def _prepare_score(self, score, num_tokens):
        if score is None:
            return None
        if score.dim() == 4:
            score = score.flatten(2).mean(dim=1)
        elif score.dim() == 3:
            if score.shape[-1] == num_tokens:
                score = score.mean(dim=1)
            elif score.shape[1] == num_tokens:
                score = score.mean(dim=-1)
            else:
                score = score.flatten(1)
        elif score.dim() > 4:
            score = score.flatten(1)

        if score.shape[-1] != num_tokens:
            score = F.interpolate(
                score.unsqueeze(1).float(),
                size=num_tokens,
                mode="linear",
                align_corners=False,
            ).squeeze(1).to(dtype=score.dtype)
        return score

    def forward(self, search_feature, template_feature=None,
                attention_score=None, confidence_score=None, source=None):
        if search_feature.dim() != 3:
            raise ValueError("search_feature must have shape [B, N, C]")

        tokens = search_feature
        search_len = search_feature.shape[1]
        if template_feature is not None:
            if template_feature.dim() != 3:
                raise ValueError("template_feature must have shape [B, N, C]")
            tokens = torch.cat([template_feature, search_feature], dim=1)

        B, N, C = tokens.shape
        saliency_source = source or self.source

        if saliency_source == "attention_score":
            score = self._prepare_score(attention_score, N)
            if score is None:
                score = tokens.norm(dim=-1)
        elif saliency_source == "confidence_score":
            score = self._prepare_score(confidence_score, N)
            if score is None:
                score = tokens.norm(dim=-1)
        elif saliency_source == "feature_norm":
            score = tokens.norm(dim=-1)
        else:
            raise ValueError("Unsupported saliency source: %s" % saliency_source)

        k = min(self.topk, N)
        topk_score, topk_index = torch.topk(score, k=k, dim=1)
        gather_index = topk_index.unsqueeze(-1).expand(B, k, C)
        selected = torch.gather(tokens, dim=1, index=gather_index)
        is_search = topk_index >= (N - search_len)

        return {
            "tokens": selected,
            "indices": topk_index,
            "scores": topk_score,
            "is_search": is_search,
        }


class MultiLayerPromptEncoder(nn.Module):
    """Encode selected single-layer or multi-layer tokens into prompt tokens."""

    def __init__(self, input_dim, prompt_dim=None, num_prompts=4, max_layers=8,
                 max_tokens=256, num_heads=4, depth=1, mlp_ratio=2.0):
        super().__init__()
        prompt_dim = int(prompt_dim or input_dim)
        self.input_dim = int(input_dim)
        self.prompt_dim = prompt_dim
        self.num_prompts = int(num_prompts)
        self.max_layers = int(max_layers)
        self.max_tokens = int(max_tokens)

        self.input_proj = nn.Linear(self.input_dim, self.prompt_dim)
        self.layer_embed = nn.Embedding(self.max_layers, self.prompt_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, self.prompt_dim))
        self.prompt_queries = nn.Parameter(torch.randn(1, self.num_prompts, self.prompt_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.prompt_dim,
            nhead=num_heads,
            dim_feedforward=int(self.prompt_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.prompt_dim)

    def forward(self, tokens):
        layers = _as_layer_list(tokens)
        encoded_layers = []
        for layer_id, layer_tokens in enumerate(layers):
            if layer_tokens.dim() != 3:
                raise ValueError("tokens must have shape [B, N, C]")
            B, N, _ = layer_tokens.shape
            x = self.input_proj(layer_tokens)
            pos = self.pos_embed[:, :N]
            if N > self.max_tokens:
                pos = F.interpolate(
                    self.pos_embed.transpose(1, 2),
                    size=N,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            layer_index = min(layer_id, self.max_layers - 1)
            x = x + pos + self.layer_embed.weight[layer_index].view(1, 1, -1)
            encoded_layers.append(x)

        x = torch.cat(encoded_layers, dim=1)
        x = self.encoder(x)
        x = self.norm(x)

        query = self.prompt_queries.expand(x.shape[0], -1, -1)
        attn = torch.matmul(query, x.transpose(1, 2)) / (self.prompt_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        prompts = torch.matmul(attn, x)
        return prompts


class RemotePromptAggregator(nn.Module):
    """Aggregate UAV-level prompts using detached prediction reliability."""

    _QUALITY_KEYS = (
        "per_remote_score",
        "per_remote_apce",
        "per_remote_bbox_score",
        "per_remote_motion_reliability",
    )

    def __init__(self, mode="mean", temperature=0.25, eps=1e-6,
                 min_quality=0.0, diagnostics=True):
        super().__init__()
        self.mode = validate_remote_aggregation(mode)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.min_quality = float(min_quality)
        self.diagnostics = bool(diagnostics)
        if self.temperature <= 0:
            raise ValueError("REMOTE_WEIGHT_TEMPERATURE must be positive")
        if self.eps <= 0:
            raise ValueError("REMOTE_WEIGHT_EPS must be positive")

    def _stack_remote(self, remote_prompt):
        if isinstance(remote_prompt, (list, tuple)):
            remote_prompt = torch.stack(remote_prompt, dim=1)
        if remote_prompt.dim() == 3:
            remote_prompt = remote_prompt.unsqueeze(1)
        if remote_prompt.dim() != 4:
            raise ValueError(
                "remote_prompt must have shape [B, M, C] or [B, R, M, C]"
            )
        return remote_prompt

    def _state_tensor(self, remote_state, key, batch_size, num_remote,
                      device, dtype):
        if not isinstance(remote_state, dict) or key not in remote_state:
            return None
        value = remote_state[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.detach().to(device=device)
        if value.dim() == 0:
            value = value.view(1, 1)
        elif value.dim() == 1:
            if value.numel() == num_remote:
                value = value.view(1, num_remote)
            elif value.numel() == batch_size:
                value = value.view(batch_size, 1)
            else:
                return None
        else:
            value = value.reshape(value.shape[0], -1)
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1)
        if value.shape != (batch_size, num_remote):
            return None
        if key == "per_remote_valid":
            return value.to(dtype=torch.bool)
        return value.to(dtype=dtype)

    def _quality(self, remote_state, batch_size, num_remote, device, dtype):
        valid = self._state_tensor(
            remote_state,
            "per_remote_valid",
            batch_size,
            num_remote,
            device,
            dtype,
        )
        if valid is None:
            valid = torch.ones(
                batch_size, num_remote, device=device, dtype=torch.bool
            )

        log_sum = torch.zeros(batch_size, num_remote, device=device, dtype=dtype)
        count = torch.zeros_like(log_sum)
        for key in self._QUALITY_KEYS:
            metric = self._state_tensor(
                remote_state, key, batch_size, num_remote, device, dtype
            )
            if metric is None:
                continue
            available = torch.isfinite(metric)
            clamped = metric.clamp(min=self.eps, max=1.0)
            log_sum = log_sum + torch.where(
                available, clamped.log(), torch.zeros_like(clamped)
            )
            count = count + available.to(dtype=dtype)

        has_quality = count > 0
        quality = torch.zeros_like(log_sum)
        quality = torch.where(
            has_quality,
            torch.exp(log_sum / count.clamp_min(1.0)),
            quality,
        )
        eligible = valid & has_quality & (quality >= self.min_quality)
        return quality, valid, eligible

    def _fallback_weights(self, valid, dtype):
        valid_weight = valid.to(dtype=dtype)
        valid_count = valid_weight.sum(dim=1, keepdim=True)
        uniform_valid = valid_weight / valid_count.clamp_min(1.0)
        uniform_all = torch.full_like(valid_weight, 1.0 / valid_weight.shape[1])
        return torch.where(valid_count > 0, uniform_valid, uniform_all)

    def forward(self, remote_prompt, remote_state=None):
        prompts = self._stack_remote(remote_prompt)
        batch_size, num_remote = prompts.shape[:2]
        quality, valid, eligible = self._quality(
            remote_state,
            batch_size,
            num_remote,
            prompts.device,
            prompts.dtype,
        )

        fallback = torch.zeros(batch_size, device=prompts.device, dtype=torch.bool)
        if self.mode == "mean":
            # Keep this exact operation for checkpoint and bbox compatibility.
            merged = prompts.mean(dim=1)
            weights = torch.full(
                (batch_size, num_remote),
                1.0 / num_remote,
                device=prompts.device,
                dtype=prompts.dtype,
            )
        else:
            has_eligible = eligible.any(dim=1)
            fallback = ~has_eligible
            safe_quality = quality.clamp_min(self.eps)
            if self.mode == "confidence_softmax":
                logits = safe_quality.log() / self.temperature
                logits = logits.masked_fill(~eligible, torch.finfo(logits.dtype).min)
                logits = torch.where(
                    has_eligible.view(-1, 1), logits, torch.zeros_like(logits)
                )
                weights = torch.softmax(logits, dim=1)
                weights = weights * eligible.to(dtype=weights.dtype)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
            else:
                center = torch.as_tensor(
                    0.5, device=prompts.device, dtype=prompts.dtype
                ).log()
                logits = (safe_quality.log() - center) / self.temperature
                weights = torch.sigmoid(logits) * eligible.to(dtype=prompts.dtype)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)

            fallback_weights = self._fallback_weights(valid, prompts.dtype)
            weights = torch.where(fallback.view(-1, 1), fallback_weights, weights)
            merged = (prompts * weights[:, :, None, None]).sum(dim=1)

        quality_for_stats = torch.where(valid, quality, torch.zeros_like(quality))
        valid_count = valid.sum(dim=1)
        quality_count = (valid & (quality > 0)).sum(dim=1)
        quality_sum = quality_for_stats.sum(dim=1)
        quality_mean = quality_sum / quality_count.clamp_min(1).to(prompts.dtype)
        quality_min = torch.where(
            valid & (quality > 0),
            quality,
            torch.full_like(quality, float("inf")),
        ).min(dim=1).values
        quality_min = torch.where(
            torch.isfinite(quality_min), quality_min, torch.zeros_like(quality_min)
        )
        quality_max = quality_for_stats.max(dim=1).values
        entropy = -(weights * weights.clamp_min(self.eps).log()).sum(dim=1)
        confidence_remote = (weights * quality).sum(dim=1, keepdim=True)

        return {
            "prompt": merged,
            "weights": weights.detach(),
            "quality": quality.detach(),
            "confidence": confidence_remote.detach(),
            "fallback": fallback.detach(),
            "diagnostics": {
                "remote_weight_entropy": entropy.detach(),
                "remote_weight_max": weights.max(dim=1).values.detach(),
                "remote_weight_mean": weights.mean(dim=1).detach(),
                "selected_remote_index": weights.argmax(dim=1).detach(),
                "valid_remote_count": valid_count.detach(),
                "remote_quality_mean": quality_mean.detach(),
                "remote_quality_min": quality_min.detach(),
                "remote_quality_max": quality_max.detach(),
                "fallback_to_uniform": fallback.detach(),
            },
        }


class PromptAligner(nn.Module):
    """Normalize local prompts and optionally align them with remote prompts."""

    def __init__(self, prompt_dim, gate="cosine", remote_aggregation="mean",
                 remote_weight_temperature=0.25, remote_weight_eps=1e-6,
                 remote_weight_min_quality=0.0,
                 remote_weight_diagnostics=True):
        super().__init__()
        if gate not in ("cosine", "confidence", "cosine_confidence"):
            raise ValueError("Unsupported align gate: %s" % gate)
        self.gate = gate
        self.local_norm = nn.LayerNorm(prompt_dim)
        self.remote_norm = nn.LayerNorm(prompt_dim)
        self.align_proj = nn.Sequential(
            nn.Linear(prompt_dim * 2, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.remote_aggregator = RemotePromptAggregator(
            mode=remote_aggregation,
            temperature=remote_weight_temperature,
            eps=remote_weight_eps,
            min_quality=remote_weight_min_quality,
            diagnostics=remote_weight_diagnostics,
        )

    def _merge_remote(self, remote_prompt, remote_state=None):
        return self.remote_aggregator(remote_prompt, remote_state)

    def forward(self, local_prompt, remote_prompt=None, local_state=None, remote_state=None):
        if local_prompt.dim() != 3:
            raise ValueError("local_prompt must have shape [B, M, C]")

        local = self.local_norm(local_prompt)
        if remote_prompt is None:
            return {
                "prompt": F.normalize(local, dim=-1),
                "gate": None,
            }

        aggregation = self._merge_remote(remote_prompt, remote_state)
        remote = self.remote_norm(aggregation["prompt"])
        if remote.shape[1] != local.shape[1]:
            remote = F.interpolate(
                remote.transpose(1, 2),
                size=local.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        cosine_gate = None
        confidence_gate = None
        if self.gate in ("cosine", "cosine_confidence"):
            cosine = F.cosine_similarity(local, remote, dim=-1, eps=1e-6).mean(dim=1, keepdim=True)
            cosine_gate = ((cosine + 1.0) * 0.5).unsqueeze(-1)

        if self.gate in ("confidence", "cosine_confidence"):
            B = local.shape[0]
            local_conf = _state_confidence(local_state, B, local.device, local.dtype)
            remote_conf = _state_confidence(
                remote_state, B, local.device, local.dtype
            )
            if self.remote_aggregator.mode != "mean":
                weighted_conf = aggregation["confidence"].to(
                    device=local.device, dtype=local.dtype
                )
                remote_conf = torch.where(
                    aggregation["fallback"].view(B, 1),
                    remote_conf,
                    weighted_conf,
                )
            confidence_gate = (remote_conf / (local_conf + remote_conf + 1e-6)).view(B, 1, 1)

        if self.gate == "cosine_confidence":
            gate = cosine_gate * confidence_gate
        elif self.gate == "cosine":
            gate = cosine_gate
        else:
            gate = confidence_gate

        gate = gate.clamp(0.0, 1.0)
        aligned_delta = self.align_proj(torch.cat([local, remote], dim=-1))
        aligned = local + gate * aligned_delta
        return {
            "prompt": F.normalize(aligned, dim=-1),
            "gate": gate,
            "remote_weights": aggregation["weights"],
            "remote_quality": aggregation["quality"],
            "remote_aggregation_diagnostics": aggregation["diagnostics"],
        }


class PromptFusion(nn.Module):
    """Fuse aligned prompt tokens into search tokens."""

    def __init__(self, token_dim, prompt_dim=None, mode="gated_add", init_scale=0.0,
                 max_scale=0.0):
        super().__init__()
        if mode not in ("gated_add", "film"):
            raise ValueError("Unsupported fusion mode: %s" % mode)
        prompt_dim = int(prompt_dim or token_dim)
        self.mode = mode
        self.max_scale = float(max_scale)
        self.prompt_proj = nn.Linear(prompt_dim, token_dim)
        self.gate = nn.Linear(token_dim * 2, token_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))
        if mode == "film":
            self.film = nn.Linear(prompt_dim, token_dim * 2)

    def _residual_scale(self):
        if self.max_scale <= 0:
            return self.residual_scale
        max_scale = torch.as_tensor(
            self.max_scale,
            device=self.residual_scale.device,
            dtype=self.residual_scale.dtype,
        )
        return max_scale * torch.tanh(self.residual_scale / max_scale)

    def forward(self, search_tokens, aligned_prompt):
        if search_tokens.dim() != 3:
            raise ValueError("search_tokens must have shape [B, N, C]")
        if aligned_prompt.dim() != 3:
            raise ValueError("aligned_prompt must have shape [B, M, C]")

        prompt_pool = aligned_prompt.mean(dim=1)
        prompt = self.prompt_proj(prompt_pool).unsqueeze(1).expand(-1, search_tokens.shape[1], -1)
        residual_scale = self._residual_scale()

        if self.mode == "gated_add":
            gate = torch.sigmoid(self.gate(torch.cat([search_tokens, prompt], dim=-1)))
            return search_tokens + residual_scale * gate * prompt

        gamma_beta = self.film(prompt_pool).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return search_tokens + residual_scale * (search_tokens * torch.tanh(gamma) + beta)


class RemoteSuppressionGate(nn.Module):
    """Prediction-only gate for suppressing a frozen remote residual."""

    def __init__(self, token_dim, init_bias=-4.0):
        super().__init__()
        self.proj = nn.Linear(int(token_dim) * 3 + 3, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, local_feature, remote_context, remote_delta,
                diagnostics=None, override=None):
        if local_feature.dim() != 3:
            raise ValueError("local_feature must have shape [B, N, C]")
        if remote_context.dim() != 3:
            raise ValueError("remote_context must have shape [B, N, C]")
        if remote_delta.dim() != 3:
            raise ValueError("remote_delta must have shape [B, N, C]")

        batch_size = local_feature.shape[0]
        device = local_feature.device
        dtype = local_feature.dtype

        def _diag_value(key):
            if not isinstance(diagnostics, dict) or key not in diagnostics:
                return torch.zeros(batch_size, 1, device=device, dtype=dtype)
            value = diagnostics[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, device=device, dtype=dtype)
            value = value.detach().to(device=device, dtype=dtype).reshape(batch_size, -1)
            return value[:, :1]

        gate_input = torch.cat([
            local_feature.detach().mean(dim=1),
            remote_context.detach().mean(dim=1),
            remote_delta.detach().abs().mean(dim=1),
            _diag_value("remote_quality_mean"),
            _diag_value("remote_weight_entropy"),
            _diag_value("remote_weight_max"),
        ], dim=1)

        if override is None:
            suppress = torch.sigmoid(self.proj(gate_input) + self.bias)
        else:
            suppress = torch.full(
                (batch_size, 1),
                float(override),
                device=device,
                dtype=dtype,
            )
        return suppress.view(batch_size, 1, 1), gate_input


class PCUM(nn.Module):
    """Prompt-based cross-UAV feature fusion and consistency modeling."""

    def __init__(self, token_dim, prompt_dim=None, num_prompts=4, topk=16,
                 saliency_source="feature_norm", fusion_mode="gated_add",
                 align_gate="cosine", enabled=False, fusion_init_scale=0.0,
                 fusion_scale_max=0.0, remote_aggregation="mean",
                 remote_weight_temperature=0.25, remote_weight_eps=1e-6,
                 remote_weight_min_quality=0.0,
                 remote_weight_diagnostics=True,
                 remote_suppression_enabled=False,
                 remote_suppression_gate_init_bias=-4.0,
                 remote_suppression_active_threshold=0.5):
        super().__init__()
        prompt_dim = int(prompt_dim or token_dim)
        self.enabled = bool(enabled)
        self.selector = SaliencyTokenSelector(topk=topk, source=saliency_source)
        self.encoder = MultiLayerPromptEncoder(
            input_dim=token_dim,
            prompt_dim=prompt_dim,
            num_prompts=num_prompts,
        )
        self.aligner = PromptAligner(
            prompt_dim=prompt_dim,
            gate=align_gate,
            remote_aggregation=remote_aggregation,
            remote_weight_temperature=remote_weight_temperature,
            remote_weight_eps=remote_weight_eps,
            remote_weight_min_quality=remote_weight_min_quality,
            remote_weight_diagnostics=remote_weight_diagnostics,
        )
        self.fusion = PromptFusion(
            token_dim=token_dim,
            prompt_dim=prompt_dim,
            mode=fusion_mode,
            init_scale=fusion_init_scale,
            max_scale=fusion_scale_max,
        )
        self.remote_suppression_enabled = bool(remote_suppression_enabled)
        self.remote_suppression_active_threshold = float(
            remote_suppression_active_threshold)
        self.remote_suppression_gate = None
        if self.remote_suppression_enabled:
            self.remote_suppression_gate = RemoteSuppressionGate(
                token_dim=token_dim,
                init_bias=remote_suppression_gate_init_bias,
            )

    def _unpack_features(self, features):
        if torch.is_tensor(features):
            return features, None, None, None, None

        search = features.get("search", features.get("search_tokens"))
        template = features.get("template", features.get("template_tokens"))
        layers = features.get("layers", None)
        attention = features.get("attention_score", None)
        confidence = features.get("confidence_score", None)
        if search is None and layers is not None:
            search = _as_layer_list(layers)[-1]
        if search is None:
            raise ValueError("features must provide search/search_tokens or layers")
        return search, template, layers, attention, confidence

    def _select_and_encode_prompt(self, search_tokens, template_tokens, layers,
                                  attention_score, confidence_score):
        selected = self.selector(
            search_tokens,
            template_feature=template_tokens,
            attention_score=attention_score,
            confidence_score=confidence_score,
        )

        prompt_input = selected["tokens"]
        if layers is not None:
            prompt_input = []
            for layer_tokens in _as_layer_list(layers):
                layer_selected = self.selector(
                    layer_tokens,
                    template_feature=None,
                    attention_score=attention_score,
                    confidence_score=confidence_score,
                )
                prompt_input.append(layer_selected["tokens"])

        return selected, self.encoder(prompt_input)

    def _forward_standard(self, search_tokens, template_tokens, layers,
                          attention_score, confidence_score, local_state,
                          remote_prompts, remote_states):
        selected, local_prompt = self._select_and_encode_prompt(
            search_tokens,
            template_tokens,
            layers,
            attention_score,
            confidence_score,
        )
        aligned = self.aligner(
            local_prompt,
            remote_prompt=remote_prompts,
            local_state=local_state,
            remote_state=remote_states,
        )
        fused_search = self.fusion(search_tokens, aligned["prompt"])

        return {
            "search_tokens": fused_search,
            "local_prompt": local_prompt,
            "aligned_prompt": aligned["prompt"],
            "selected_indices": selected["indices"],
            "selected_scores": selected["scores"],
            "align_gate": aligned["gate"],
            "remote_weights": aligned.get("remote_weights", None),
            "remote_quality": aligned.get("remote_quality", None),
            "remote_aggregation_diagnostics": aligned.get(
                "remote_aggregation_diagnostics", None
            ),
        }

    def _forward_remote_suppression(self, search_tokens, template_tokens, layers,
                                    attention_score, confidence_score, local_state,
                                    remote_prompts, remote_states,
                                    remote_suppression_override=None):
        selected, local_prompt = self._select_and_encode_prompt(
            search_tokens,
            template_tokens,
            layers,
            attention_score,
            confidence_score,
        )
        local_aligned = self.aligner(local_prompt, remote_prompt=None)
        local_feature = self.fusion(search_tokens, local_aligned["prompt"])

        if remote_prompts is None:
            aligned = local_aligned
            a0_feature = local_feature
        else:
            aligned = self.aligner(
                local_prompt,
                remote_prompt=remote_prompts,
                local_state=local_state,
                remote_state=remote_states,
            )
            a0_feature = self.fusion(search_tokens, aligned["prompt"])

        remote_delta = (a0_feature - local_feature).detach()
        suppress, gate_input = self.remote_suppression_gate(
            local_feature.detach(),
            a0_feature.detach(),
            remote_delta,
            diagnostics=aligned.get("remote_aggregation_diagnostics", None),
            override=remote_suppression_override,
        )
        suppressed_delta = suppress * remote_delta
        fused_search = a0_feature.detach() - suppressed_delta
        active = suppress.detach() > self.remote_suppression_active_threshold

        align_gate = aligned["gate"]
        if torch.is_tensor(align_gate):
            align_gate = align_gate.detach()
        return {
            "search_tokens": fused_search,
            "local_prompt": local_prompt.detach(),
            "aligned_prompt": aligned["prompt"].detach(),
            "selected_indices": selected["indices"],
            "selected_scores": selected["scores"],
            "align_gate": align_gate,
            "remote_weights": aligned.get("remote_weights", None),
            "remote_quality": aligned.get("remote_quality", None),
            "remote_aggregation_diagnostics": aligned.get(
                "remote_aggregation_diagnostics", None
            ),
            "remote_suppression": suppress,
            "remote_suppression_gate_input": gate_input.detach(),
            "remote_delta_norm": remote_delta.detach().float().norm(dim=-1).mean().detach(),
            "suppressed_delta_norm": suppressed_delta.detach().float().norm(dim=-1).mean().detach(),
            "remote_suppression_active_ratio": active.float().mean().detach(),
            "remote_suppression_enabled": True,
        }

    def forward(self, features, local_state=None, remote_prompts=None, remote_states=None,
                remote_suppression_override=None):
        search_tokens, template_tokens, layers, attention_score, confidence_score = self._unpack_features(features)
        if not self.enabled:
            return {
                "search_tokens": search_tokens,
                "local_prompt": None,
                "aligned_prompt": None,
                "selected_indices": None,
                "align_gate": None,
            }

        if self.remote_suppression_enabled:
            return self._forward_remote_suppression(
                search_tokens,
                template_tokens,
                layers,
                attention_score,
                confidence_score,
                local_state,
                remote_prompts,
                remote_states,
                remote_suppression_override=remote_suppression_override,
            )

        return self._forward_standard(
            search_tokens,
            template_tokens,
            layers,
            attention_score,
            confidence_score,
            local_state,
            remote_prompts,
            remote_states,
        )


def build_pcum(cfg, token_dim=None):
    pcum_cfg = cfg.MODEL.PCUM
    token_dim = int(token_dim or pcum_cfg.TOKEN_DIM)
    return PCUM(
        token_dim=token_dim,
        prompt_dim=pcum_cfg.PROMPT_DIM,
        num_prompts=pcum_cfg.NUM_PROMPTS,
        topk=pcum_cfg.TOPK,
        saliency_source=pcum_cfg.SALIENCY_SOURCE,
        fusion_mode=pcum_cfg.FUSION,
        align_gate=pcum_cfg.ALIGN_GATE,
        enabled=pcum_cfg.ENABLED,
        fusion_init_scale=getattr(pcum_cfg, "FUSION_INIT_SCALE", 0.0),
        fusion_scale_max=getattr(pcum_cfg, "FUSION_SCALE_MAX", 0.0),
        remote_aggregation=getattr(pcum_cfg, "REMOTE_AGGREGATION", "mean"),
        remote_weight_temperature=getattr(
            pcum_cfg, "REMOTE_WEIGHT_TEMPERATURE", 0.25),
        remote_weight_eps=getattr(pcum_cfg, "REMOTE_WEIGHT_EPS", 1e-6),
        remote_weight_min_quality=getattr(
            pcum_cfg, "REMOTE_WEIGHT_MIN_QUALITY", 0.0),
        remote_weight_diagnostics=getattr(
            pcum_cfg, "REMOTE_WEIGHT_DIAGNOSTICS", True),
        remote_suppression_enabled=getattr(
            pcum_cfg, "REMOTE_SUPPRESSION_ENABLED", False),
        remote_suppression_gate_init_bias=getattr(
            pcum_cfg, "REMOTE_SUPPRESSION_GATE_INIT_BIAS", -4.0),
        remote_suppression_active_threshold=getattr(
            pcum_cfg, "REMOTE_SUPPRESSION_ACTIVE_THRESHOLD", 0.5),
    )
