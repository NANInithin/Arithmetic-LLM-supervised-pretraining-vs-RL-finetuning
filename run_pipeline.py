import argparse
import subprocess
import os
import mlflow
from src.config import get_config

# Configuration: Load from YAML (v4 architecture)
CONFIG = get_config()
METADATA_RUN_NAME = "Pipeline_Metadata_v4"
EXPERIMENT_NAME = "Arithmetic_LLM_Scaling_v4"


def ensure_dirs():
    """Ensure required directories exist."""
    for dir_name in [CONFIG.paths.checkpoint_dir, CONFIG.paths.results_dir,
                     CONFIG.paths.plots_dir, CONFIG.paths.logs_dir,
                     CONFIG.paths.mlruns_dir]:
        os.makedirs(dir_name, exist_ok=True)


def log_pipeline_metadata():
    """
    Logs configuration and environment files to a dedicated MLflow run
    to ensure the entire pipeline state is reproducible.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=METADATA_RUN_NAME):
        # Log hyperparameters if the file exists
        if os.path.exists("configs/hyperparams.yaml"):
            mlflow.log_artifact("configs/hyperparams.yaml")
            print("✅ Logged configs/hyperparams.yaml to MLflow.")

        # Log requirements for environment reproducibility
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt")
            print("✅ Logged requirements.txt to MLflow.")

        # Log key config parameters
        mlflow.log_params({
            "model/embed_dim": CONFIG.model.embed_dim,
            "model/num_heads": CONFIG.model.num_heads,
            "model/num_layers": CONFIG.model.num_layers,
            "model/dim_feedforward": CONFIG.model.dim_feedforward,
            "model/max_len": CONFIG.model.max_len,
            "training/batch_size": CONFIG.training.batch_size,
            "training/learning_rate": CONFIG.training.learning_rate,
            "training/num_samples": CONFIG.training.num_samples,
            "training/max_digits_supervised": CONFIG.training.max_digits_supervised,
            "training/epochs": CONFIG.training.epochs,
            "rl/total_episodes": CONFIG.rl.total_episodes,
            "rl/batch_size": CONFIG.rl.batch_size,
            "rl/max_new_tokens": CONFIG.rl.max_new_tokens,
        })


def run_step(command, description):
    """
    Executes a shell command and monitors for errors.
    """
    print(f"\n>>> STARTING: {description}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"!!! ERROR in {description}")
        exit(1)


def run_full_pipeline():
    """Run the complete training pipeline: supervised → RL → eval."""
    # 1. Ensure directories exist
    print("\nEnsuring directories exist...")
    ensure_dirs()

    # 2. Clean old local log files (since we now rely on MLflow)
    print("Cleaning local workspace...")
    for f in ["supervised_loss.npy", "rl_rewards_replay.npy"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed old log: {f}")

    # 3. Log Pipeline Metadata (Configs and Requirements)
    log_pipeline_metadata()

    # 4. Sequential Execution of the ML Lifecycle
    # Step A: Supervised Learning (The foundation)
    run_step("python -m src.train_supervised", "Supervised Pretraining (v4)")

    # Step B: RL Fine-Tuning (Polishing with Denser Rewards)
    run_step("python -m src.train_rl", "RL Fine-Tuning (v4)")

    # Step C: Final Evaluation
    run_step("python -m src.evaluate", "Final Model Evaluation")


def main():
    parser = argparse.ArgumentParser(
        description="Arithmetic LLM v4 Pipeline - Supervised Pretraining + RL Fine-Tuning + Evaluation"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["supervised", "rl", "eval", "all"],
        default="all",
        help="Pipeline stage to run. Default: all"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Arithmetic LLM Pipeline - v4 Architecture (~310M params)")
    print("=" * 60)

    if args.stage == "all":
        run_full_pipeline()
    elif args.stage == "supervised":
        ensure_dirs()
        run_step("python -m src.train_supervised", "Supervised Pretraining (v4)")
    elif args.stage == "rl":
        ensure_dirs()
        run_step("python -m src.train_rl", "RL Fine-Tuning (v4)")
    elif args.stage == "eval":
        ensure_dirs()
        run_step("python -m src.evaluate", "Final Model Evaluation")

    # Final Success Summary
    print("\n" + "=" * 50)
    print("[PIPELINE COMPLETE SUCCESS]")
    print("=" * 50)
    print("1. All metrics are now live in MLflow.")
    print("2. Models are saved to:")
    print(f"   - {CONFIG.paths.pretrained_model}")
    print(f"   - {CONFIG.paths.rl_model}")
    print("3. To view graphs, start the MLflow UI with:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("   Then open: http://127.0.0.1:5000")
    print("=" * 50)


if __name__ == "__main__":
    main()
