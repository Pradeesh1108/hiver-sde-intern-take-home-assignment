#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "======================================================="
echo "  Setting up Hiver AI Agent Project"
echo "======================================================="

# 1. Backend Setup
echo ""
echo "[1/3] Setting up Python backend environment..."
if ! command -v uv &> /dev/null
then
    echo "  'uv' could not be found. Installing 'uv' via curl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "  Please restart your terminal or source your profile to use uv, then re-run this script."
    exit 1
fi

echo "  Creating virtual environment and syncing dependencies..."
uv sync
echo "  Backend dependencies installed successfully."

# 2. Frontend Setup
echo ""
echo "[2/3] Setting up React frontend environment..."
if ! command -v npm &> /dev/null
then
    echo "  'npm' could not be found. Please install Node.js and try again."
    exit 1
fi

cd client
echo "  Installing Node modules..."
npm install
echo "  Building production frontend..."
npm run build
cd ..
echo "  Frontend built successfully."

# 3. Environment Variables
echo ""
echo "[3/3] Checking environment variables..."
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "  Copying .env.example to .env..."
        cp .env.example .env
    else
        echo "  Creating a starter .env file..."
        echo "GROQ_API_KEY=your_api_key_here" > .env
        echo "LLM_PROVIDER=groq" >> .env
    fi
    echo "  [ACTION REQUIRED] Please edit the '.env' file and add your actual API keys!"
else
    echo "  .env file already exists."
fi

echo ""
echo "======================================================="
echo "  Setup Complete! 🎉"
echo "======================================================="
echo "Next Steps:"
echo "1. Activate your virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Add your Groq API key to the .env file."
echo ""
echo "3. Generate the embeddings and semantic index (Required for first run):"
echo "   ./scripts/generate_data.sh"
echo ""
echo "4. Start the server:"
echo "   uv run uvicorn main:app --reload"
echo "======================================================="
