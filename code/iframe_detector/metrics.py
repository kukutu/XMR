from __future__ import annotations

from collections import defaultdict
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
    truth_sets = [set(packets) for _, packets in truth]
    truth_by_packet: dict[int, set[int]] = defaultdict(set)
    for idx, packets in enumerate(truth_sets):
        for packet in packets:
            truth_by_packet[int(packet)].add(idx)
    matched_truth: set[int] = set()
    tp = 0
    ious: list[float] = []
    for _, pred_packets in predicted:
        pred_set = set(pred_packets)
        candidate_truth: set[int] = set()
        for packet in pred_set:
            candidate_truth.update(truth_by_packet.get(int(packet), set()))
        best_idx = -1
        best_iou = 0.0
        for i in candidate_truth:
            if i in matched_truth:
                continue
            true_set = truth_sets[i]
            if not pred_set and not true_set:
                iou = 1.0
            elif not pred_set or not true_set:
                iou = 0.0
            else:
                intersection = len(pred_set & true_set)
                union = len(pred_set) + len(true_set) - intersection
                iou = intersection / union if union else 0.0
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


def _empty_counts() -> dict[str, float]:
    return {
        "true_positive_count": 0,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "predicted_positive_count": 0,
        "true_keyframe_count": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mean_packet_iou": 0.0,
    }


def combine_match_counts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return _empty_counts()
    tp = int(sum(r["true_positive_count"] for r in rows))
    fp = int(sum(r["false_positive_count"] for r in rows))
    fn = int(sum(r["false_negative_count"] for r in rows))
    pred = int(sum(r["predicted_positive_count"] for r in rows))
    truth = int(sum(r["true_keyframe_count"] for r in rows))
    precision = tp / pred if pred else 0.0
    recall = tp / truth if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou_weight = sum(r["mean_packet_iou"] * r["true_positive_count"] for r in rows)
    return {
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "predicted_positive_count": pred,
        "true_keyframe_count": truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_packet_iou": iou_weight / tp if tp else 0.0,
    }


def match_packet_sets_grouped(
    predicted: list[tuple[str, str, list[int]]],
    truth: list[tuple[str, str, list[int]]],
    minimum_iou: float = 0.9,
) -> dict[str, float]:
    pred_by_group: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    truth_by_group: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for group_key, item_key, packets in predicted:
        pred_by_group[str(group_key)].append((str(item_key), packets))
    for group_key, item_key, packets in truth:
        truth_by_group[str(group_key)].append((str(item_key), packets))
    groups = sorted(set(pred_by_group) | set(truth_by_group))
    return combine_match_counts(
        [
            match_packet_sets(pred_by_group.get(group, []), truth_by_group.get(group, []), minimum_iou)
            for group in groups
        ]
    )
