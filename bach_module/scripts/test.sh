#!/bin/bash

# Data preprocessing script
# Sử dụng: bash scripts/test.sh

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo Python path: $PYTHONPATH

set -e

echo Testing
# python -m spacy download en_core_web_sm
# python -m src.KGSum.kgsum_agent

python -m src.language_models.call_llm_openai

# python -m baselines.GraphCare.graphcare_.graph_generation.ChatGPT