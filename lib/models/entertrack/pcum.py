import torch
from torch import nn
import torch.nn.functional as F


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


class PromptAligner(nn.Module):
    """Normalize local prompts and optionally align them with remote prompts."""

    def __init__(self, prompt_dim, gate="cosine"):
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

    def _merge_remote(self, remote_prompt):
        if isinstance(remote_prompt, (list, tuple)):
            remote_prompt = torch.stack(remote_prompt, dim=1)
        if remote_prompt.dim() == 4:
            remote_prompt = remote_prompt.mean(dim=1)
        if remote_prompt.dim() != 3:
            raise ValueError("remote_prompt must have shape [B, M, C] or [B, R, M, C]")
        return remote_prompt

    def forward(self, local_prompt, remote_prompt=None, local_state=None, remote_state=None):
        if local_prompt.dim() != 3:
            raise ValueError("local_prompt must have shape [B, M, C]")

        local = self.local_norm(local_prompt)
        if remote_prompt is None:
            return {
                "prompt": F.normalize(local, dim=-1),
                "gate": None,
            }

        remote = self.remote_norm(self._merge_remote(remote_prompt))
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
            remote_conf = _state_confidence(remote_state, B, local.device, local.dtype)
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


class PCUM(nn.Module):
    """Prompt-based cross-UAV feature fusion and consistency modeling."""

    def __init__(self, token_dim, prompt_dim=None, num_prompts=4, topk=16,
                 saliency_source="feature_norm", fusion_mode="gated_add",
                 align_gate="cosine", enabled=False, fusion_init_scale=0.0,
                 fusion_scale_max=0.0):
        super().__init__()
        prompt_dim = int(prompt_dim or token_dim)
        self.enabled = bool(enabled)
        self.selector = SaliencyTokenSelector(topk=topk, source=saliency_source)
        self.encoder = MultiLayerPromptEncoder(
            input_dim=token_dim,
            prompt_dim=prompt_dim,
            num_prompts=num_prompts,
        )
        self.aligner = PromptAligner(prompt_dim=prompt_dim, gate=align_gate)
        self.fusion = PromptFusion(
            token_dim=token_dim,
            prompt_dim=prompt_dim,
            mode=fusion_mode,
            init_scale=fusion_init_scale,
            max_scale=fusion_scale_max,
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

    def forward(self, features, local_state=None, remote_prompts=None, remote_states=None):
        search_tokens, template_tokens, layers, attention_score, confidence_score = self._unpack_features(features)
        if not self.enabled:
            return {
                "search_tokens": search_tokens,
                "local_prompt": None,
                "aligned_prompt": None,
                "selected_indices": None,
                "align_gate": None,
            }

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

        local_prompt = self.encoder(prompt_input)
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
        }


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
    )
