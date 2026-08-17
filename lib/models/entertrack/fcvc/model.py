import torch
import torch.nn as nn

from .config import FCVCConfig
from .deformable_cross_view import DeformableCrossViewBlock
from .global_matcher import GlobalSemanticMatcher
from .local_query_builder import ReceiverLocalQueryBuilder
from .residual_writer import ResidualWriter
from .sender_bundle import validate_sender_pair
from .teacher import FCVCTeacher


class FCVCModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or FCVCConfig()
        self.query_builder = ReceiverLocalQueryBuilder(
            self.cfg.token_dim, self.cfg.embed_dim, self.cfg.num_queries,
            self.cfg.grid_size)
        self.matcher = GlobalSemanticMatcher(
            self.cfg.token_dim, self.cfg.embed_dim, self.cfg.num_heads,
            self.cfg.num_senders, self.cfg.grid_size, self.cfg.null_bias)
        self.mid_block = DeformableCrossViewBlock(
            self.cfg.token_dim, self.cfg.embed_dim, self.cfg.num_heads,
            self.cfg.num_senders, self.cfg.samples_per_sender,
            self.cfg.grid_size)
        self.high_block = DeformableCrossViewBlock(
            self.cfg.token_dim, self.cfg.embed_dim, self.cfg.num_heads,
            self.cfg.num_senders, self.cfg.samples_per_sender,
            self.cfg.grid_size)
        self.mid_writer = ResidualWriter(
            self.cfg.token_dim, self.cfg.embed_dim, self.cfg.num_heads,
            self.cfg.residual_norm_bound)
        self.high_writer = ResidualWriter(
            self.cfg.token_dim, self.cfg.embed_dim, self.cfg.num_heads,
            self.cfg.residual_norm_bound)
        self.teacher = FCVCTeacher(
            self.cfg.token_dim, self.cfg.embed_dim, self.cfg.num_queries)

    def trainable_parameter_names(self, include_teacher=True):
        allowed = (
            "query_builder.", "matcher.", "mid_block.", "high_block.",
            "mid_writer.", "high_writer.",
        )
        teacher = ("teacher.",) if include_teacher else ()
        prefixes = allowed + teacher
        return [name for name, p in self.named_parameters()
                if p.requires_grad and name.startswith(prefixes)]

    def optimizer_parameters(self, include_teacher=True):
        names = set(self.trainable_parameter_names(include_teacher))
        return [p for name, p in self.named_parameters() if name in names]

    def fcvc_state_dict(self, include_teacher=False):
        prefixes = tuple("fcvc." + n for n in self.trainable_parameter_names(include_teacher))
        raw = self.state_dict()
        return {k: v for k, v in raw.items()
                if k in self.trainable_parameter_names(include_teacher)
                or k.startswith(prefixes)}

    def forward(self, local, sender_bundles=(), replay_fn=None, forward_head=None,
                force_zero_residual=False, force_null=False):
        local_output = local["local_output"]
        if not self.cfg.enabled:
            return {"state_output": local_output, "reported_output": local_output,
                    "used_remote": False, "reason": "disabled"}
        sender_bundles = tuple(sorted(
            sender_bundles,
            key=lambda bundle: tuple(int(v) for v in bundle.view_id.detach().cpu().reshape(-1).tolist())))
        valid, reason = validate_sender_pair(tuple(sender_bundles),
                                             local["mid_search"].shape[0],
                                             self.cfg.token_dim,
                                             self.cfg.search_tokens)
        if not valid:
            return {"state_output": local_output, "reported_output": local_output,
                    "used_remote": False, "reason": reason}
        q = self.query_builder(
            local["mid_search"], local["high_search"],
            local["response_map"], local["confidence_uncertainty"],
            local["target_prototype"])
        match = self.matcher(q["queries"], tuple(sender_bundles))
        mid = self.mid_block(
            q["queries"], [b.mid_features for b in sender_bundles],
            match["sender_reference_points"], match["matched"],
            force_null=force_null)
        mid_written = self.mid_writer(
            local["template_mid"], local["mid_search"], mid["queries"],
            force_zero_residual=force_zero_residual or force_null)
        if replay_fn is None:
            high_tokens = torch.cat((local["template_high"].clone(),
                                     mid_written["search_tokens"]), dim=1)
        else:
            high_tokens = replay_fn(mid_written["tokens"])
        template_high = high_tokens[:, :-self.cfg.search_tokens]
        search_high = high_tokens[:, -self.cfg.search_tokens:]
        high = self.high_block(
            mid["queries"], [b.high_features for b in sender_bundles],
            match["sender_reference_points"], match["matched"],
            force_null=force_null)
        high_written = self.high_writer(
            template_high, search_high, high["queries"],
            force_zero_residual=force_zero_residual or force_null)
        if force_zero_residual or force_null:
            reported = local_output
        elif forward_head is None:
            reported = dict(local_output)
            reported["fcvc_search_tokens"] = high_written["search_tokens"]
        else:
            reported = forward_head(high_written["tokens"])
        return {
            "state_output": local_output,
            "reported_output": reported,
            "local_output": local_output,
            "used_remote": True,
            "queries": high["queries"],
            "mid_writer": mid_written,
            "high_writer": high_written,
            "global_match": match,
            "mid_block": mid,
            "high_block": high,
        }
