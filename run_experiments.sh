#!/bin/bash
# VULCA Framework - Main Experiment Runner

echo "========================================="
echo "VULCA Framework Benchmark Evaluation"
echo "========================================="

# Check Python environment
echo "Checking Python environment..."
python --version

# Install dependencies if needed
if [ "$1" == "--install" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start vLLM server if needed
if [ "$1" == "--start-server" ]; then
    echo "Starting vLLM server..."
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --port 8000 \
        --trust-remote-code \
        --max-model-len 16384 &
    sleep 10
fi

# Run main pipeline
echo "Running benchmark pipeline..."
python src/pipeline.py \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    $@

echo "========================================="
echo "Experiment complete!"
echo "Results saved to: outputs/"
echo "========================================="