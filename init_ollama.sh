#!/usr/bin/env bash

module load cs/ollama
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_CONTEXT_LENGTH=32768
export OLLAMA_NUM_PARALLEL=5
ollama serve &
