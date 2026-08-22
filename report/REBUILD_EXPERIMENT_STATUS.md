# XMR Rebuild Experiment Status

Date: 2026-08-21

## Recovery Outcome

The deleted code/model/report files could not be reliably recovered from the
Recycle Bin or NTFS MFT. MFT recovery found filenames and nominal sizes, but
integrity checks failed for the important files:

- Python source files mostly contained null bytes or binary data.
- `metadata.json` and other JSON files did not parse.
- `model.pkl` files did not unpickle.
- The recovered `.docx` report was not a valid Word ZIP package.

The pipeline has therefore been reimplemented under `code/` and is now tracked
with git. Raw pcaps and generated artifacts remain ignored.

## Restored Data

Twelve raw pcapng files were restored from `D:\pcap` with exact-size matches.
The only known missing original capture is:

- `data\livstreaming_jd_1.pcapng`

## Rebuilt Pipeline

Current rebuilt scripts:

- `code/export_packets.py`: export packet metadata and payload hex through
  `tshark`.
- `code/build_labels.py`: derive labels from parseable TCP streams.
- `code/train_two_stage.py`: train pairwise gap boundary and frame-level
  I-frame LightGBM models.
- `code/predict_two_stage.py`: reconstruct frames and score I-frame candidates.
- `code/evaluate_two_stage_iou.py`: evaluate predicted I-frame packet sets with
  packet IoU.

Default label policy is intentionally conservative:

- Strong labels: FLV video tags.
- Weak labels: Annex-B start-code search, only enabled by
  `--allow-weak-annexb`.

Annex-B weak labels can false-trigger on encrypted or random payloads and should
not be treated as a reliable final metric.

## Latest Strong-Only Two-Stage Scheme

Current strict supervised training uses only strong parseable labels. Weak
Annex-B labels are not used as training labels.

The parsed frame table is split into two stage-specific label files:

- `frame_packet_labels.csv.gz`: packet-to-frame membership labels for stage 1.
- `frame_type_labels.csv.gz`: frame-level `is_keyframe` labels for stage 2.

On the recovered full split, strong FLV labels are available only for
`wechat_primary`:

- 30,282 labeled frames.
- 531 I-frame/keyframe labels.
- No strong labels are available for the current `final_app_ood`
  xiaohongshu captures.

Therefore the strict strong-label pipeline cannot currently produce a valid
final-app OOD metric. A temporal holdout on `wechat_primary` is used only to
verify that the two-stage implementation is correct.

Strong-only temporal validation at packet IoU >= 0.9:

- Stage 1 frame reconstruction: precision 74.73%, recall 56.05%, F1 64.06%.
- Stage 2 oracle I-frame classification: precision 98.32%, recall 99.25%,
  F1 98.78%.
- End-to-end raw: precision 68.41%, recall 93.41%, F1 78.98%.
- End-to-end with GOP/NMS: precision 88.06%, recall 93.03%, F1 90.48%.

The current scheme summary is tracked in `latest/README.md`.

## Smoke Test

Smoke setup:

- Exported first 20,000 packets from each restored pcap.
- Environment: `F:\anaconda\envs\pcap\python.exe`, `F:\wireshark\tshark.exe`.
- Unit tests: 4 passed.

Strong FLV labels on the smoke slice:

- 0 labeled frames.

Weak Annex-B labels on the smoke slice:

- 3,616 labeled frames.
- 2,757 keyframe labels.
- This keyframe ratio is suspiciously high and confirms the weak-label caveat.

Weak-label two-stage smoke result at packet IoU >= 0.9:

- Default threshold 0.4: precision 3.13%, recall 50.82%, F1 5.90%.
- Explicit threshold 0.7: precision 1.31%, recall 1.05%, F1 1.17%.

Interpretation:

- The rebuilt code path is executable end to end.
- The current weak-label smoke result is not a valid replacement for the lost
  previous experiment result.
- Reliable model training requires either full-capture strong labels from
  parseable streams, recovered old derived labels, or a new labeling source for
  RTP/SRTP/UDP traffic.
