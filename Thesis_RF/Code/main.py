try:
	from orchestrator import simulation_main
except ModuleNotFoundError:
	from Code.orchestrator import simulation_main


def main(argv: list[str] | None = None) -> int:
	return simulation_main(argv)


if __name__ == "__main__":
	raise SystemExit(main())
