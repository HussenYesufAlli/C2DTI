.PHONY: test smoke verify check-data-example check-data-all scaffold-data-layout fill-demo-data run-once-all real-all

test:
	python -m unittest discover -s tests -p 'test_*.py'

smoke:
	python scripts/run.py --config configs/minimal.yaml --dry-run
	python scripts/run.py --config configs/minimal.yaml --run-once

verify: test smoke

check-data-example:
	python scripts/run.py --config configs/davis_real_pipeline_strict.yaml --check-data

check-data-all:
	python scripts/check_all_data.py

scaffold-data-layout:
	python scripts/scaffold_data_layout.py

fill-demo-data:
	python scripts/fill_demo_data.py

run-once-all:
	python scripts/run_all_once.py

real-all: check-data-all run-once-all
