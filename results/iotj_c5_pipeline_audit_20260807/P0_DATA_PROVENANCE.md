# P0 data provenance

- OLD upstream is `D:\A Python learning\Federated Learning\TRAE SOLO\dataset\processed\unit_5` (legacy `preprocessor.py`, SHA-bound below).
- NEW upstream is `D:\A Python learning\Federated Learning\TRAE SOLO\results\time_aware_pipeline_probe_window_fullgrid\time_aware_60_170_window_fullgrid\processed\unit_5` (time-aware `preprocessor_time_aware.py`; the generating scripts are present in the parent workspace but not tracked by the audited branch).
- Both upstreams contain the same 80 Unit-5 raw filenames and 1,680 windows. The resolved `dataset/data1` raw files are individually SHA256-bound in the CSV, but their processed `features.npy` hashes and values differ.
- OLD: remove first 20 s, row-based downsample, relative conductance baseline over the post-removal first 30 s, then crop 40--150 s (physical 60--170 s), float64.
- NEW: preserve/clean the time column, merge duplicate timestamps, interpolate on real seconds at 10 Hz, relative conductance, directly crop 60--170 s, float32.
- Therefore the change is **not float precision only**. It is a preprocessing implementation/version difference over the same named raw experiments.
- Dataset-level normalization is not applied to saved windows; `norm_stats.npz` is a separately computed source-train statistic used only by loaders that request normalization.

| Field | OLD processed Unit5 | NEW processed Unit5 |
|---|---|---|
| Path | `D:\A Python learning\Federated Learning\TRAE SOLO\dataset\processed\unit_5` | `D:\A Python learning\Federated Learning\TRAE SOLO\results\time_aware_pipeline_probe_window_fullgrid\time_aware_60_170_window_fullgrid\processed\unit_5` |
| Shape / dtype | (1680, 100, 8) / float64 | (1680, 100, 8) / float32 |
| Min / max | -0.139712692358 / 4.70936607064 | -0.0260862261057 / 4.70926952362 |
| Mean / std | 0.654436223811 / 0.887650620094 | 0.665605306625 / 0.892043411732 |
| features SHA256 | `15881cc9cd9f25c173c0e844d99a0c96b35890d95ef661dd5c2647c31aaf0db4` | `91b307c85daffb1bbe257f156a8b40939d8f03f1926bfa61088a947dbb6f1133` |
| classification labels SHA256 | `1c8d58314f41adf41b6e89e3d7ef170e6dd34ec993dbb774c9ee3db2b03f994c` | `d288dbc5868fd573e965f104f8a89df07c4329e60bb5553c72d253c545e40705` |
| regression labels SHA256 | `dd653ec40195d843fe5e459cb3fdaf5ece002a21ba89534921fe9b39621ff53a` | `dd653ec40195d843fe5e459cb3fdaf5ece002a21ba89534921fe9b39621ff53a` |
| phase labels SHA256 | `73d34c67c876f4770fb01a8c942a6bffa8e09c73d4edbcbbcb08f62cb982225c` | `19c8e34af91e2accff1ab9aae6923811d43e3e0db144c331e983d8d837937b7c` |

Classification/regression/phase label **values** are identical; classification and phase byte hashes differ because OLD uses int32/int8 and NEW uses int64.

| Parameter | OLD | NEW |
|---|---|---|
| original_fs / target_fs | 100 / 10 Hz | 100 / 10 Hz |
| unstable / baseline | remove first 20 s; 30 s baseline afterward | baseline on raw time [20,50) s |
| physical response crop | 60--170 s (implemented as 40--150 s after removal) | 60--170 s directly |
| window / stride | 100 / 50 samples | 100 / 50 samples |
| relative conductance | `(1/R - mean(1/R_baseline))/mean(1/R_baseline)` | same formula |
| time handling | row decimation; timestamp column discarded | stable timestamp sort, duplicate merge, real-time interpolation |
| saved feature dtype | float64 | float32 |
| saved-window z-score | none | none |
| generator Git provenance | tracked: `preprocessor.py` at initial `a0ce0b5`; splitter later touched by `396e304` | generator files present but untracked in the audited branch; SHA256 recorded |

See `p0_dataset_manifest.csv` for shapes, dtypes, moments, byte hashes, labels, metadata, split manifests and norm statistics.
