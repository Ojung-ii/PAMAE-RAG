# Environment Check

## Pytest

See `00_pytest.log`.

## Git Diff Check

See `00_git_diff_check.log`.

## Git Status

See `00_git_status_short.log`.

## GPU

See `00_nvidia_smi.log` and `00_nvidia_smi_retry.log`.

`nvidia-smi` did not communicate with the driver in this session at 02:40 KST. This does not block the current retrieval experiments because PAMAE-RAG preprocessing and retrieval are currently CPU-based `numpy`/`scikit-learn` code unless a GPU model is explicitly introduced.
