from __future__ import annotations

from dataclasses import dataclass


def packet_iou(a: list[int] | set[int], b: list[int] | set[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def match_packet_sets(
    predicted: list[tuple[str, list[int]]],
    truth: list[tuple[str, list[int]]],
    minimum_iou: float = 0.9,
) -> dict[str, float]:
    matched_truth: set[int] = set()
    tp = 0
    ious: list[float] = []
    for _, pred_packets in predicted:
        best_idx = -1
        best_iou = 0.0
        for i, (_, true_packets) in enumerate(truth):
            if i in matched_truth:
                continue
            iou = packet_iou(pred_packets, true_packets)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0 and best_iou >= minimum_iou:
            matched_truth.add(best_idx)
            tp += 1
            ious.append(best_iou)
    fp = len(predicted) - tp
    fn = len(truth) - tp
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "predicted_positive_count": len(predicted),
        "true_keyframe_count": len(truth),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_packet_iou": sum(ious) / len(ious) if ious else 0.0,
    }

