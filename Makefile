# SPDX-License-Identifier: Apache-2.0
# 
install-linters: ## install the linters
	pip install uv mypy==1.9.0 ruff==0.6.9

lint:
	python3 ci/check_licenses.py
	python -m ruff format --check
	python -m ruff check

format:
	python3 ci/check_licenses.py --fix
	python -m ruff format

docs:
	uvx --with mkdocs-material mkdocs build

docs-serve:
	uvx --with mkdocs-material mkdocs serve

pytest:
	torchrun --nproc_per_node 1 -m pytest -vv -rs

.PHONY: docs
