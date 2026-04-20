.PHONY: test smoke

test:
	python -m unittest discover -s tests -p 'test_*.py'

smoke:
	python scripts/run.py --config configs/minimal.yaml --dry-run
	python scripts/run.py --config configs/minimal.yaml --run-once
