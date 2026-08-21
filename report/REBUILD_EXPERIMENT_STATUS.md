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
