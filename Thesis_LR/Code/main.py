from pathlib import Path

try:
	from orchestrator import apply_pipeline_defaults, load_config, run_simulation
except ModuleNotFoundError:
	from Code.orchestrator import apply_pipeline_defaults, load_config, run_simulation


def main() -> None:
	default_config_path = Path(__file__).resolve().parent / "config" / "default_experiment.yaml"
	config = load_config(str(default_config_path))
	config = apply_pipeline_defaults(config)
	run_simulation(config)


if __name__ == "__main__":
	main()
