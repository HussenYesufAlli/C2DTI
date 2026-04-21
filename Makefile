.PHONY: test smoke verify check-data-example check-data-all

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
