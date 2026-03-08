# --- Configuration ---
PYTHON = python3
VENV = venv
BIN = $(VENV)/bin

# Check for OS to handle activation differences (Windows vs Unix)
ifeq ($(OS), Windows_NT)
    ACTIVATE = $(VENV)\Scripts\activate
    VENV_PYTHON = $(VENV)\Scripts\python
    VENV_PIP = $(VENV)\Scripts\pip
else
    ACTIVATE = . $(VENV)/bin/activate
    VENV_PYTHON = $(VENV)/bin/python
    VENV_PIP = $(VENV)/bin/pip
endif

# --- Default Goal ---
.PHONY: help
help:
	@echo "Quantum GA & VQE Project Management"
	@echo "------------------------------------"
	@echo "Usage:"
	@echo "  make install         - Create venv and install dependencies"
	@echo "  make run-ga          - Run Genetic Algorithm (Unitary Decomposition)"
	@echo "  make run-vqe-pure    - Run Pure VQE (Strongly Entangling Layers)"
	@echo "  make run-vqe-warm    - Run Warm Start VQE (GA + Optimization)"
	@echo "  make visualize       - Generate Matplotlib graphs from VQE results"
	@echo "  make clean           - Remove venv and temporary files"

# --- Installation ---
install: $(VENV)/bin/activate

$(VENV)/bin/activate:
	@echo "Creating Virtual Environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing Dependencies..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install numpy pennylane matplotlib

# --- Execution Targets ---

run-ga: install
	@echo "Executing Genetic Algorithm..."
	$(VENV_PYTHON) genetic_algorithm.py

run-vqe-pure: install
	@echo "Executing Pure VQE Script..."
	$(VENV_PYTHON) pure_vqe.py

run-vqe-warm: install
	@echo "Executing Warm Start VQE (GA-initialized)..."
	$(VENV_PYTHON) warm_start_vqe.py

visualize: install
	@if [ ! -f vqe_results.json ]; then \
		echo "Error: vqe_results.json not found. Run 'make run-vqe-pure' first."; \
		exit 1; \
	fi
	@echo "Generating Visualization Plots..."
	$(VENV_PYTHON) visualise_vqe.py

# --- Cleanup ---
clean:
	rm -rf $(VENV)
	rm -f vqe_results.json
	find . -type d -name "__pycache__" -exec rm -rf {} +