# Changelog

All notable changes to this project are documented here.
Format: `[YYYY-MM-DD] <type>: <short description>` followed by bullet details.
Types: `feat` (new feature), `arch` (architecture change), `train` (training/RL change), `data` (dataset change), `eval` (evaluation change), `fix` (bug fix), `docs` (documentation), `infra` (tooling/env/config).

---

## Upcoming / In-progress

- [ ] Architecture v1 → v2 proposal (Roo Code)
- [ ] Curriculum learning for arithmetic difficulty scaling

---

## History

### [2026-07-13] eval/fix/infra: v4 first successful full Modal run — results + runtime fixes

- **Results (500 random test samples/model, full log in `docs/run_details.md`):**
  - Pretrained: **95.00%** (475/500) overall — **100%** on 1–4 digit, **99.30%** on 5-digit.
  - RL Fine-Tuned: **94.60%** (473/500) overall — **100%** on 1–4 digit, **97.90%** on 5-digit.
  - By operation (pretrained): − 100%, + 95.7%, × 89.9%. Fixes the v3 collapse (4.00% → 95.00%) and beats v2 (67.6%).
- **RL finding:** RL fine-tuning did **not** improve over supervised pretraining (95.00% → 94.60%; × regressed 89.9% → 88.6%). Deferred to v5. Eval has ~±2% sampling variance (unseeded, regenerated per run).
- **infra:** Switched Modal GPU **A100-40GB → H100** (supervised is compute-bound; H100's 80 GB also allows disabling gradient checkpointing). Added parallel `DataLoader` workers, single shared MLflow SQLite DB on the volume, `PYTHONUNBUFFERED`/`GIT_PYTHON_REFRESH`/`PYTORCH_CUDA_ALLOC_CONF` env, whitelist-only file upload, and `run_rl` timeout 3h→6h.
- **fix (runtime bugs that blocked the first run — see `docs/architecture_v3_vs_v4.md` §12):** `modal.Mount` removed in Modal 1.0 (use `image.add_local_dir`); `select_dataset` called `.get()` on `CurriculumPhase` dataclasses; `evaluate.py` used relative checkpoint paths that don't resolve on the volume; MLflow metric names rejected `+`/`*` operation symbols; removed version-fragile `mlflow.pytorch.log_model`; wired the ignored `use_gradient_checkpointing` flag through to the model.
- **Models saved** to the `arithmetic-llm-checkpoints` volume and downloaded locally to `checkpoints/` (`pretrained_v4.pth`, `rl_finetuned_v4.pth`).

### [2026-07-12] arch/eval/infra: v4 Modal Cloud + Scaled Scratchpad CoT

- **Phase 1 - Evaluation fix:** `src/evaluate.py` now evaluates WITH scratchpad (`use_scratchpad=True`), uses greedy decoding (`temperature=0.0`), stops at EOS, and extracts the answer after the last `|`. Added per-operation (+, -, *) accuracy breakdown.
- **Phase 2 - Model scaling:** Scaled ArithTransformer v4 to ~310M params (`embed_dim=896`, `num_heads=14`, `num_layers=24`, `dim_feedforward=3584`, `max_len=256`). Updated `configs/hyperparams.yaml` and `src/config.py` defaults.
- **Phase 3 - Training config:** Supervised config now uses 300k samples, 5-digit curriculum, 20 epochs, batch_size=128, warmup=800. RL config uses 15k episodes, 7-phase 2→3→4→5 curriculum, replay buffer size=4000, `max_new_tokens=60`.
- **Phase 4 - Modal integration:** Added `modal_train.py` with A100-40GB functions and persistent checkpoint volume. `run_pipeline.py` now supports `--stage {supervised,rl,eval,all}` and is the single entry point used by Modal.
- **Phase 5 - Dataset & RL:** `src/dataset.py` `_get_multiplicand` now supports 5-digit distributions. `src/train_rl.py` adds `ds_very_hard` and 7-phase curriculum selection.
- **Phase 6 - Validation:** Smoke tests passed (model creation, 5-digit dataset, greedy generation, training step loss decrease, answer extraction).
- **Phase 7 - Documentation:** Added Modal instructions to README, GETTING_STARTED, and `docs/run_details.md`. Added `modal` to `requirements.txt` and `.modal/` to `.gitignore`.

### [2026-05-17] docs: Initial agent rules and project setup

- Added `docs/PROJECT_RULES.md` with global agent rules, git policy, venv execution rules, and design-first workflow.
- Added `docs/AGENT_RooCode.md`, `docs/AGENT_Cline.md`, `docs/AGENT_aider.md`, `docs/AGENT_gemcli.md` with per-agent instructions.
- Added `docs/README_agents.md` (this file's companion) explaining the agent workflow.
- Added `docs/architecture_design_template.md` as template for all architecture comparison docs.
- Added `CHANGELOG.md` to track project evolution.

---

## How to add an entry

Copy and fill in:

```
### [YYYY-MM-DD] <type>: <short description>

- Bullet: what changed and why.
- Bullet: which files were modified.
- Bullet: any known issues or follow-ups.
```
