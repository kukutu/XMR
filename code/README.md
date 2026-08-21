# XMR I-frame Detector Rebuild

This directory contains a rebuilt, git-maintained experiment pipeline after the
previous workspace cleanup loss.

The pipeline is intentionally conservative:

1. Export packet metadata and optional transport payloads with `tshark`.
2. Group packets into flows and select likely media flows by downlink bytes.
3. Reassemble parseable TCP payload streams and derive video frame labels from
   FLV video tags when visible.
4. Train a pairwise packet-gap boundary model for frame reconstruction.
5. Train a frame-level I-frame classifier.
6. Evaluate final predicted I-frame packet sets with packet IoU.

Raw pcaps and generated artifacts are intentionally ignored by git.

## Typical Commands

```powershell
$py = 'F:\anaconda\envs\pcap\python.exe'
$env:PYTHONPATH = 'D:\XMR\code'

& $py code\export_packets.py `
  --data-root data `
  --output-root artifacts\rebuild_packets `
  --packet-limit 200000 `
  --keep-payload

& $py code\build_labels.py `
  --packet-root artifacts\rebuild_packets `
  --output-root artifacts\rebuild_labels
```

`build_labels.py` defaults to strong FLV labels only. Raw Annex-B start-code
search is available with `--allow-weak-annexb`, but it should be treated as
weak-label/smoke-test mode because encrypted or random payloads can contain
accidental start-code byte patterns.

```powershell
& $py code\train_two_stage.py `
  --packet-root artifacts\rebuild_packets `
  --label-root artifacts\rebuild_labels `
  --output-root artifacts\rebuild_models

& $py code\predict_two_stage.py `
  --packet-root artifacts\rebuild_packets `
  --model-root artifacts\rebuild_models `
  --output-root artifacts\rebuild_predictions

& $py code\evaluate_two_stage_iou.py `
  --prediction-root artifacts\rebuild_predictions `
  --label-root artifacts\rebuild_labels `
  --minimum-packet-iou 0.9 `
  --output-path artifacts\rebuild_predictions\iframe_iou90_summary.json
```

By default these commands use
`code/config/splits_recovered_20260727_bidir.json`, which restores the previous
primary split:

- `development`: `douyin_primary`, `pinduoduo_primary`, `wechat_primary`,
  `wechat_secondary`
- `final_app_ood`: `xiaohongshu_primary`, `xiaohongshu_secondary`

Run this before training to verify the restored split against the current pcaps:

```powershell
& $py code\verify_recovered_split.py --data-root data --fail-on-missing
```

For quick validation:

```powershell
& $py -m unittest discover -s code\tests -v
```
