"""
Modal cloud training wrapper for Arithmetic LLM v4.

This script launches the project's single entry point (`run_pipeline.py`)
on Modal H100 GPUs. All training and evaluation is orchestrated through
`run_pipeline.py`, consistent with local execution.

Usage:
    # Run the full pipeline on Modal
    modal run modal_train.py

    # Run only a specific stage
    modal run modal_train.py --stage supervised
    modal run modal_train.py --stage rl
    modal run modal_train.py --stage eval

Requirements:
    - modal package installed and authenticated
    - Modal account with GPU access
"""

import os
import subprocess

import modal


# ---------------------------------------------------------------------------
# Modal App Definition
# ---------------------------------------------------------------------------

# Persistent volume for checkpoints, logs, and MLflow artifacts.
# Mounts at /checkpoints inside the container.
checkpoints_volume = modal.Volume.from_name(
    "arithmetic-llm-checkpoints",
    create_if_missing=True,
)

# Container image with project dependencies. Only the files required at
# runtime are added to the image (a whitelist), keeping the upload tiny — the
# rest of the repo (docs, PNGs, mlflow.db, session logs) is never sent.
#
# NOTE: modal.Mount was removed in Modal 1.0. Local files are now attached to
# the image via .add_local_dir()/.add_local_file(). By default this happens at
# container startup (copy=False), which is fine since nothing in the build
# depends on them. The pipeline needs: run_pipeline.py, src/, configs/, and
# requirements.txt (the latter only logged to MLflow for reproducibility).
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "mlflow>=2.0.0",
    )
    # PYTHONUNBUFFERED: stream prints/tqdm from the nested training subprocesses
    #   to the Modal logs in real time (otherwise stdout is block-buffered since
    #   the container has no TTY, and epoch logs only appear at the end).
    # GIT_PYTHON_REFRESH: silence MLflow's noisy "Bad git executable" warnings —
    #   there is no git binary in the container and we don't need git provenance.
    # MLFLOW_TRACKING_URI: point MLflow at a single SQLite DB on the persistent
    #   volume, so EVERY Modal run appends to the same tracking store and runs
    #   can be compared across sessions. Read automatically by the mlflow client,
    #   so no changes to src/ are needed.
    # PYTORCH_CUDA_ALLOC_CONF: reduce CUDA memory fragmentation (the allocator
    #   otherwise strands memory as "reserved but unallocated"). Recommended by
    #   PyTorch's own OOM message; cheap insurance near the VRAM ceiling.
    .env({
        "PYTHONUNBUFFERED": "1",
        "GIT_PYTHON_REFRESH": "quiet",
        "MLFLOW_TRACKING_URI": "sqlite:////checkpoints/mlflow.db",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir(
        "src", remote_path="/root/llm/src",
        ignore=["__pycache__", "*.pyc"],
    )
    .add_local_dir("configs", remote_path="/root/llm/configs")
    .add_local_file("run_pipeline.py", remote_path="/root/llm/run_pipeline.py")
    .add_local_file("requirements.txt", remote_path="/root/llm/requirements.txt")
)

app = modal.App("arithmetic-llm-v4", image=image)

# The single experiment all runs log into, and where its artifacts (logged
# models, config files) are stored. Both live on the persistent volume so the
# tracking DB and its artifacts survive across runs and can be compared.
MLFLOW_EXPERIMENT = "Arithmetic_LLM_Scaling_v4"
MLFLOW_ARTIFACT_LOCATION = "file:///checkpoints/mlartifacts"


# ---------------------------------------------------------------------------
# Helper: detect repo root and run pipeline stage
# ---------------------------------------------------------------------------

def _ensure_shared_mlflow_experiment():
    """
    Make MLflow write to the single shared store on the volume.

    MLFLOW_TRACKING_URI (set on the image) already points the DB at
    sqlite:////checkpoints/mlflow.db. Here we additionally guarantee the
    experiment exists with its artifact_location pinned to the volume, so
    logged models/artifacts persist too (otherwise a new experiment would
    default its artifacts to the ephemeral container disk). Idempotent: the
    first run creates it; later runs find it already in the shared DB and the
    src/ code's mlflow.set_experiment(...) simply reuses it.
    """
    import mlflow

    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=MLFLOW_ARTIFACT_LOCATION,
        )
        print(f"Created shared MLflow experiment '{MLFLOW_EXPERIMENT}' "
              f"(artifacts -> {MLFLOW_ARTIFACT_LOCATION}).")
    else:
        print(f"Reusing shared MLflow experiment '{MLFLOW_EXPERIMENT}' "
              f"at {mlflow.get_tracking_uri()}.")


def _run_pipeline_stage(stage: str = "all"):
    """Run run_pipeline.py inside the Modal container."""
    repo_root = "/root/llm"

    if not os.path.isdir(repo_root):
        raise FileNotFoundError(
            f"Repo mount not found at {repo_root}. "
            "Ensure modal_train.py is run from the repo root."
        )

    # Ensure the shared MLflow store/experiment exist before the pipeline runs.
    _ensure_shared_mlflow_experiment()

    cmd = ["python", "run_pipeline.py"]
    if stage != "all":
        cmd.extend(["--stage", stage])

    print(f"Running in {repo_root}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=repo_root, check=True)
    finally:
        # Persist the updated tracking DB + artifacts to the volume so the next
        # run (and `modal volume get`) sees them, even if the pipeline errors.
        checkpoints_volume.commit()


# ---------------------------------------------------------------------------
# Modal Functions
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",               # ~989 TFLOPS, 80 GB VRAM — compute-bound, so ~2-3x faster than A100
    cpu=8,                    # More workers to keep the faster H100 fed
    memory=16384,             # 16 GB RAM
    timeout=3600 * 6,         # 6h timeout (supervised + RL comfortably fit)
    volumes={"/checkpoints": checkpoints_volume},
)
def run_full_pipeline():
    """Run the complete v4 pipeline on Modal (supervised → RL → eval)."""
    # Ensure checkpoints are written to the mounted volume.
    os.environ["ARITH_LLM_CHECKPOINT_DIR"] = "/checkpoints"
    _run_pipeline_stage("all")


@app.function(
    gpu="H100",
    cpu=8,
    memory=16384,
    timeout=3600 * 3,
    volumes={"/checkpoints": checkpoints_volume},
)
def run_supervised():
    """Run only supervised pretraining on Modal."""
    os.environ["ARITH_LLM_CHECKPOINT_DIR"] = "/checkpoints"
    _run_pipeline_stage("supervised")


@app.function(
    gpu="H100",
    cpu=8,
    memory=16384,
    # 6h: RL runs 15k episodes of batch-1, no-KV-cache autoregressive generation
    # (O(n^2), slower in the late 5-digit phases). This is latency-bound, so the
    # H100 doesn't speed it up much — give it headroom so it isn't killed before
    # the model saves. Raise further if the 5-digit phases push past this.
    timeout=3600 * 6,
    volumes={"/checkpoints": checkpoints_volume},
)
def run_rl():
    """Run only RL fine-tuning on Modal."""
    os.environ["ARITH_LLM_CHECKPOINT_DIR"] = "/checkpoints"
    _run_pipeline_stage("rl")


@app.function(
    gpu="H100",
    cpu=2,
    memory=8192,
    timeout=3600,
    volumes={"/checkpoints": checkpoints_volume},
)
def run_eval():
    """Run only evaluation on Modal."""
    os.environ["ARITH_LLM_CHECKPOINT_DIR"] = "/checkpoints"
    _run_pipeline_stage("eval")


# ---------------------------------------------------------------------------
# Local Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(stage: str = "all"):
    """
    Orchestrate v4 training on Modal.

    Args:
        stage: "all" | "supervised" | "rl" | "eval"
    """
    print(f"Starting v4 pipeline on Modal (stage={stage})...")

    if stage == "all":
        run_full_pipeline.remote()
    elif stage == "supervised":
        run_supervised.remote()
    elif stage == "rl":
        run_rl.remote()
    elif stage == "eval":
        run_eval.remote()
    else:
        raise ValueError(f"Unknown stage: {stage}")

    print("✅ Modal run complete.")
