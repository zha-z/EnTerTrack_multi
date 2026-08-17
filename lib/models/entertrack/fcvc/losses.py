import torch
import torch.nn.functional as F


LOSS_WEIGHTS = {
    "align": 0.50,
    "recon": 1.00,
    "safe": 0.50,
    "cycle": 0.10,
    "teacher_track": 0.25,
    "giou": 2.0,
    "l1": 5.0,
    "cls": 1.0,
}


def reconstruction_loss(student_slots, teacher_slots):
    target = teacher_slots.detach()
    cos = 1.0 - F.cosine_similarity(student_slots, target, dim=-1)
    mse = (student_slots - target).square().mean(dim=-1)
    denom = target.detach().square().mean(dim=-1).clamp_min(1e-6)
    return 0.5 * cos.mean() + 0.5 * (mse / denom).mean()


def safe_loss(collab_loss, local_loss):
    local = local_loss.detach()
    return F.relu((collab_loss - local) / (local + 0.1)).mean()


def cycle_loss(returned_queries, original_queries):
    return (1.0 - F.cosine_similarity(returned_queries, original_queries.detach(), dim=-1)).mean()


def align_loss(attention, target_distribution):
    target = target_distribution.detach().clamp_min(1e-6)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    pred = attention.clamp_min(1e-6)
    return -(target * pred.log()).sum(dim=-1).mean()


def fcvc_total_loss(track_student, align, recon, safe, cycle,
                    track_teacher=None):
    track = track_student
    if track_teacher is not None:
        track = track + LOSS_WEIGHTS["teacher_track"] * track_teacher
    return (track
            + LOSS_WEIGHTS["align"] * align
            + LOSS_WEIGHTS["recon"] * recon
            + LOSS_WEIGHTS["safe"] * safe
            + LOSS_WEIGHTS["cycle"] * cycle)
