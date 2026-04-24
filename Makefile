SHELL := /bin/bash

PYTHON ?= python
RUN_REAL := $(PYTHON) scripts/run.py
RUN_BINARY := $(PYTHON) scripts/run_binary.py

REAL_CHECK_EXAMPLE_CONFIG := configs/davis_real_pipeline_strict.yaml
BINARY_EXAMPLE_CONFIG := configs/davis_binary_true_end_to_end.yaml
GATE_CONFIGS := configs/davis_gate.yaml configs/kiba_gate.yaml configs/bindingdb_gate.yaml
PARITY_SEEDS ?= 10 34 42
MAX_RUNS ?= 0

.PHONY: test verify help \
	quality quality-gate \
	check-data-example check-data-all check-data-gate data-check \
	run-once-all real-all real-all-gate validate-outputs \
	run-config dry-run-config run-configs check-configs \
	binary-dry-run binary-example binary-run-config binary-dry-run-config \
	eval-matrix eval-matrix-exec binary-eval-matrix binary-eval-matrix-exec graph-parity graph-parity-exec \
	compile-results compile-binary-results compile-supervisor-report reports \
	gate-summary gate-bundle gate-all supervisor-workflow

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

verify: test

help:
	@printf '%s\n' \
		'C2DTI Make Targets' \
		'' \
		'Quality and gates:' \
		'  make test                    Run unit tests' \
		'  make verify                  Run unit-test verification' \
		'  make gate-all                Run the full project quality gate' \
		'  make gate-summary            Fail if the latest gate summary contains non-pass status' \
		'  make gate-bundle             Package the latest gate evidence bundle' \
		'' \
		'Data readiness:' \
		'  make check-data-example      Run one strict dataset check example' \
		'  make check-data-all          Check all strict real-data configs' \
		'  make check-data-gate         Check gate configs only' \
		'' \
		'Continuous / regression runs:' \
		'  make run-once-all            Run all strict continuous configs once' \
		'  make real-all                Check strict data, then run all strict continuous configs' \
		'  make run-config CONFIG=...   Run one continuous config once' \
		'  make dry-run-config CONFIG=...   Dry-run one continuous config' \
		'  make check-configs CONFIGS="cfg1 cfg2"   Check selected continuous configs' \
		'  make run-configs CONFIGS="cfg1 cfg2"     Run selected continuous configs once' \
		'' \
		'Binary runs:' \
		'  make binary-dry-run          Dry-run the default binary example config' \
		'  make binary-example          Run the default binary example config once' \
		'  make binary-run-config CONFIG=...      Run one binary config once' \
		'  make binary-dry-run-config CONFIG=...  Dry-run one binary config' \
		'' \
		'Phase-6 and parity matrices:' \
		'  make eval-matrix             Generate the Phase-6 continuous matrix command sheet' \
		'  make eval-matrix-exec        Execute the Phase-6 continuous matrix' \
		'  make binary-eval-matrix      Generate the binary matrix command sheet' \
		'  make binary-eval-matrix-exec Execute the binary matrix' \
		'  make graph-parity            Generate the graph parity matrix command sheet' \
		'  make graph-parity-exec       Execute the graph parity matrix' \
		'' \
		'Reports and supervisor outputs:' \
		'  make validate-outputs        Validate latest continuous run artifacts' \
		'  make compile-results         Compile continuous matrix results' \
		'  make compile-binary-results  Compile binary matrix results' \
		'  make compile-supervisor-report  Build supervisor comparison artifacts' \
		'  make reports                 Compile all report artifacts' \
		'  make supervisor-workflow     Run the supervisor workflow shell script' \
		'' \
		'Optional variables:' \
		'  CONFIG=path/to/config.yaml' \
		'  CONFIGS="cfg1 cfg2 cfg3"' \
		'  MAX_RUNS=10' \
		'  PARITY_SEEDS="10 34 42"'

quality: verify

quality-gate: gate-all

check-data-example:
	$(RUN_REAL) --config $(REAL_CHECK_EXAMPLE_CONFIG) --check-data

check-data-all:
	$(PYTHON) scripts/check_all_data.py

check-data-gate:
	$(PYTHON) scripts/check_all_data.py --configs $(GATE_CONFIGS)

data-check: check-data-all

run-once-all:
	$(PYTHON) scripts/run_all_once.py

real-all: check-data-all run-once-all

real-all-gate: check-data-gate run-once-all

run-config:
	@test -n "$(CONFIG)" || (echo 'Use CONFIG=path/to/config.yaml'; exit 2)
	$(RUN_REAL) --config $(CONFIG) --run-once

dry-run-config:
	@test -n "$(CONFIG)" || (echo 'Use CONFIG=path/to/config.yaml'; exit 2)
	$(RUN_REAL) --config $(CONFIG) --dry-run

run-configs:
	@test -n "$(CONFIGS)" || (echo 'Use CONFIGS="cfg1 cfg2 ..."'; exit 2)
	$(PYTHON) scripts/run_all_once.py --configs $(CONFIGS)

check-configs:
	@test -n "$(CONFIGS)" || (echo 'Use CONFIGS="cfg1 cfg2 ..."'; exit 2)
	$(PYTHON) scripts/check_all_data.py --configs $(CONFIGS)

binary-dry-run:
	$(RUN_BINARY) --config $(BINARY_EXAMPLE_CONFIG) --dry-run

binary-example:
	$(RUN_BINARY) --config $(BINARY_EXAMPLE_CONFIG) --run-once

binary-run-config:
	@test -n "$(CONFIG)" || (echo 'Use CONFIG=path/to/binary_config.yaml'; exit 2)
	$(RUN_BINARY) --config $(CONFIG) --run-once

binary-dry-run-config:
	@test -n "$(CONFIG)" || (echo 'Use CONFIG=path/to/binary_config.yaml'; exit 2)
	$(RUN_BINARY) --config $(CONFIG) --dry-run

validate-outputs:
	$(PYTHON) scripts/validate_run_outputs.py

eval-matrix:
	$(PYTHON) scripts/run_eval_matrix.py --mode dry-run

eval-matrix-exec:
	$(PYTHON) scripts/run_eval_matrix.py --mode run-once --execute --max-runs $(MAX_RUNS)

binary-eval-matrix:
	$(PYTHON) scripts/run_binary_eval_matrix.py --mode dry-run

binary-eval-matrix-exec:
	$(PYTHON) scripts/run_binary_eval_matrix.py --mode run-once --execute --max-runs $(MAX_RUNS)

graph-parity:
	$(PYTHON) scripts/run_graph_parity_matrix.py --mode dry-run --seeds $(PARITY_SEEDS)

graph-parity-exec:
	$(PYTHON) scripts/run_graph_parity_matrix.py --mode run-once --execute --seeds $(PARITY_SEEDS)

compile-results:
	$(PYTHON) scripts/compile_results.py

compile-binary-results:
	$(PYTHON) scripts/compile_binary_results.py

compile-supervisor-report:
	$(PYTHON) scripts/compile_supervisor_report.py

reports: compile-results compile-binary-results compile-supervisor-report

gate-summary:
	$(PYTHON) scripts/gate_summary.py --fail-on-nonpass

gate-bundle:
	$(PYTHON) scripts/gate_bundle.py

gate-all:
	$(PYTHON) scripts/gate_all.py --real-cmd "make real-all-gate"

supervisor-workflow:
	bash scripts/run_supervisor_workflow.sh
