# XMR Latest Scheme

Date: 2026-08-22

This folder records the current tracked scheme. Generated data, packet exports,
models, and predictions remain under `artifacts/` and are intentionally ignored
by git.

## Label Policy

The current supervised pipeline does not use weak Annex-B labels as training
labels.

The label table is split into two stage-specific datasets:

- `frame_packet_labels.csv.gz`: stage-1 packet-to-frame labels. These supervise
  packet grouping / frame reconstruction.
- `frame_type_labels.csv.gz`: stage-2 frame type labels. These supervise
  frame-level I-frame classification through `is_keyframe`.

Strong labels currently come from parseable FLV video tags. In the recovered
full split, strong labels are available only for `wechat_primary`:

- frames: 30,282
- keyframes: 531
- label source: `flv`

Because the recovered `final_app_ood` captures have no strong labels, the
strong-only result below is a same-source temporal validation result, not an
OOD generalization result.

## Main Code

- `code/build_labels.py`: derives parseable strong labels. Weak Annex-B remains
  available only behind `--allow-weak-annexb`.
- `code/prepare_two_stage_labels.py`: splits parsed frame labels into stage-1
  and stage-2 label files.
- `code/train_two_stage.py`: trains the boundary model from frame-packet labels
  and the I-frame classifier from frame-type labels.
- `code/predict_two_stage.py`: reconstructs candidate frames, scores I-frame
  probability, and applies optional post-processing.
- `code/evaluate_frame_reconstruction.py`: evaluates stage-1 frame packet sets.
- `code/evaluate_iframe_oracle.py`: evaluates stage-2 classification on true
  frames.
- `code/evaluate_two_stage_iou.py`: evaluates end-to-end I-frame packet sets.

## Rebuild Commands

```powershell
$py = 'F:\anaconda\envs\pcap\python.exe'
$env:PYTHONPATH = 'D:\XMR\code'

& $py code\build_labels.py `
  --packet-root artifacts\recovered_split_full_packets `
  --output-root artifacts\recovered_split_full_labels_strong

& $py code\prepare_two_stage_labels.py `
  --input-label-root artifacts\recovered_split_full_labels_strong `
  --output-root artifacts\recovered_split_full_two_stage_labels_strong `
  --allow-label-source flv

& $py code\train_two_stage.py `
  --packet-root artifacts\recovered_split_full_packets `
  --label-root artifacts\recovered_split_full_two_stage_labels_strong `
  --output-root artifacts\recovered_split_full_models_strong_temporal `
  --label-split-fallback temporal `
  --fallback-test-fraction 0.3
```

## Current Strong-Only Validation Results

All metrics use packet IoU > 0.9.

| Stage | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Stage 1 frame reconstruction | 74.73% | 56.05% | 64.06% |
| Stage 2 oracle I-frame classification | 98.32% | 99.25% | 98.78% |
| End-to-end raw | 68.41% | 93.41% | 78.98% |
| End-to-end + GOP/NMS | 88.06% | 93.03% | 90.48% |

Primary current end-to-end artifact:

`artifacts/recovered_split_full_predictions_strong_temporal_wechat_gop/iframe_iou90_summary_full.json`

## Caveat

The recovered dataset contains encrypted or otherwise unparseable media traffic.
Those captures are useful as target-domain traffic for inference, but without
strong labels they are not used as supervised training/evaluation labels in the
current strict pipeline.
