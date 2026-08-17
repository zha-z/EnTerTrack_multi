import torch

from lib.models.entertrack.utils import combine_tokens
from .structures import TapReplayOutput


def split_template_search(tokens: torch.Tensor, search_len: int = 256):
    return tokens[:, :-search_len], tokens[:, -search_len:]


def replay_blocks(backbone, tokens: torch.Tensor, start_block: int = 4,
                  end_block: int = 6) -> torch.Tensor:
    if start_block < 1 or end_block > len(backbone.blocks):
        raise ValueError("replay block range is outside backbone")
    out = tokens
    lens_t = backbone.pos_embed_z.shape[1]
    lens_s = backbone.pos_embed_x.shape[1]
    b = out.shape[0]
    global_t = torch.arange(lens_t, device=out.device).unsqueeze(0).repeat(b, 1)
    global_s = torch.arange(lens_s, device=out.device).unsqueeze(0).repeat(b, 1)
    attn_mask = None
    frozen = None
    for blk in list(backbone.blocks)[start_block - 1:end_block]:
        out, global_s, _, _, _, attn_mask, frozen = blk(
            out, global_t, global_s, attn_mask, None, False, 1.0, frozen_token=frozen)
    return backbone.norm(out)


def capture_taps(backbone, template: torch.Tensor, search: torch.Tensor,
                 mid_after_block: int = 3) -> TapReplayOutput:
    b = search.shape[0]
    x = backbone.patch_embed(search)
    z = backbone.patch_embed(template)
    z = z + backbone.pos_embed_z
    x = x + backbone.pos_embed_x
    if backbone.add_sep_seg:
        x = x + backbone.search_segment_pos_embed
        z = z + backbone.template_segment_pos_embed
    tokens = combine_tokens(z, x, mode=backbone.cat_mode)
    if backbone.add_cls_token:
        cls_tokens = backbone.cls_token.expand(b, -1, -1) + backbone.cls_pos_embed
        tokens = torch.cat([cls_tokens, tokens], dim=1)
    tokens = backbone.pos_drop(tokens)
    lens_t = backbone.pos_embed_z.shape[1]
    lens_s = backbone.pos_embed_x.shape[1]
    global_t = torch.arange(lens_t, device=tokens.device).unsqueeze(0).repeat(b, 1)
    global_s = torch.arange(lens_s, device=tokens.device).unsqueeze(0).repeat(b, 1)
    attn_mask = None
    frozen = None
    mid = None
    for i, blk in enumerate(backbone.blocks):
        tokens, global_s, _, _, _, attn_mask, frozen = blk(
            tokens, global_t, global_s, attn_mask, None, False, 1.0,
            frozen_token=frozen)
        if i + 1 == mid_after_block:
            mid = tokens.clone()
    final_tokens = backbone.norm(tokens)
    replay_tokens = replay_blocks(backbone, mid.clone(), mid_after_block + 1,
                                  len(backbone.blocks))
    return TapReplayOutput(
        mid_tokens=mid,
        high_tokens=final_tokens,
        replay_tokens=replay_tokens,
        final_tokens=final_tokens,
    )
