# VULCA Framework Makefile

.PHONY: install run clean test help

help:
	@echo "VULCA Framework - Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make run        - Run full benchmark pipeline"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean output files"
	@echo "  make server     - Start vLLM server"

install:
	pip install -r requirements.txt

run:
	bash run_experiments.sh

server:
	python -m vllm.entrypoints.openai.api_server \
		--model Qwen/Qwen2.5-VL-7B-Instruct \
		--port 8000 \
		--trust-remote-code \
		--max-model-len 16384

test:
	python src/evaluate.py --test
	python src/pipeline.py --dry_run

clean:
	rm -rf outputs/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

format:
	black src/
	isort src/