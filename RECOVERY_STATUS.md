# XMR Recovery Status

Date: 2026-08-21

The previous workspace files were accidentally removed during cleanup. A direct
Recycle Bin restore is not available because PowerShell `Remove-Item` bypassed
the Recycle Bin.

Recovery attempts completed:

- Scanned the D: NTFS MFT and restored candidate files to
  `F:\XMR_recovery_20260821`.
- Most recovered code/model/report/artifact files failed integrity validation:
  Python files contain null bytes or invalid binary content, JSON files do not
  parse, pickle models do not load, and gzipped CSV files mostly fail gzip
  validation.
- The restored `.docx` report file is not a valid Word ZIP package.
- Original pcap files were restored from `D:\pcap` where exact-size copies were
  available.

Current known missing raw capture:

- `data\livstreaming_jd_1.pcapng`

Current rebuild direction:

- Recreate the experiment code from scratch under `code/`.
- Keep raw data and generated artifacts out of git.
- Use git commits for each stable rebuild step.
