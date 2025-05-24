#!/bin/bash

# List of models to evaluate
models=('gpt-4o' 'gpt-4o-mini' 'o3-mini' 'deepseek-v3' 'deepseek-r1')

# Create results directory if it doesn't exist
mkdir -p ../results

# Run evaluation for each model
for model in "${models[@]}"
do
    echo "Running evaluation with model: $model"
    python test_model.py "$model"
    echo "-----------------------------------"
done

# Generate summary of results
echo "Generating summary of results..."
python summarize_results.py 