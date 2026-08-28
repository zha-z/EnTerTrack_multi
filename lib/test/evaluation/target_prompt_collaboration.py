"""Three-view orchestration for one E3 frame."""

import time


def run_target_prompt_frame(trackers, images, frame_id, target_id):
    """Run exactly one local forward/view then head-only receiver collaboration."""
    trackers = tuple(trackers)
    images = tuple(images)
    views = ("A", "B", "C")
    if len(trackers) != 3 or len(images) != 3:
        raise ValueError("E3 frame runner requires exactly three views")
    before = tuple(int(getattr(
        tracker, "_target_prompt_local_forward_count", 0))
        for tracker in trackers)
    local_candidates = []
    local_times = []
    for tracker, image in zip(trackers, images):
        started = time.time()
        local_candidates.append(tracker.target_prompt_local_candidate(image))
        local_times.append(time.time() - started)
    after = tuple(int(getattr(
        tracker, "_target_prompt_local_forward_count", 0))
        for tracker in trackers)
    if tuple(end - start for start, end in zip(before, after)) != (1, 1, 1):
        raise RuntimeError("E3 must run one local backbone forward per view/frame")

    results = []
    for receiver, (tracker, local, local_time) in enumerate(zip(
            trackers, local_candidates, local_times)):
        sender_indices = tuple(index for index in range(3) if index != receiver)
        remote = tuple(local_candidates[index] for index in sender_indices)
        sender_views = tuple(views[index] for index in sender_indices)
        started = time.time()
        collaborative = tracker.target_prompt_candidate(
            local, remote, views[receiver], sender_views,
            frame_id, target_id=target_id)
        # Deliberately pass no frame info: runtime E3 does not consume GT fields.
        output, max_score, apce = tracker.target_prompt_finalize_frame(
            local, collaborative, info=None,
            debug_name="target-prompt-e3-{}".format(views[receiver].lower()))
        results.append({
            "output": output,
            "max_score": max_score,
            "apce": apce,
            "time": local_time + (time.time() - started),
            "local_candidate": local,
            "collaborative_candidate": collaborative,
        })
    return tuple(results)
