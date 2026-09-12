#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================================="
echo "  Starting Full Data Generation Pipeline"
echo "======================================================="
echo "This will take some time (especially step 3) depending"
echo "on your hardware (CPU vs Apple Silicon vs NVIDIA GPU)."
echo "======================================================="

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo ""
echo "[1/7] Reconstructing threads from raw Twitter dataset..."
python3 EDA/00_thread_reconstruction.py

echo ""
echo "[2/7] Extracting relevant Apple support messages..."
python3 EDA/01_extract_messages.py

echo ""
echo "[3/7] Generating embeddings (Heavy computation)..."
python3 EDA/02_embed.py

echo ""
echo "[4/7] Clustering messages to discover intents..."
python3 EDA/03_cluster.py

echo ""
echo "[5/7] Generating cluster inspection reports..."
python3 EDA/04_inspect.py

echo ""
echo "[6/7] Finalizing taxonomy and preparing few-shot examples..."
python3 EDA/05_build_taxonomy.py

echo ""
echo "[7/7] Building the semantic retrieval index..."
python3 EDA/06_build_semantic_index.py

echo ""
echo "======================================================="
echo "  Data Generation Pipeline Complete!"
echo "======================================================="
echo "The agent's knowledge base and taxonomy are now ready."
echo "You can now start the server: uv run uvicorn main:app --reload"
