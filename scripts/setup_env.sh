#!/bin/bash

# Exit on error
set -e

# Define brew path
if [[ $(uname) == "Darwin" ]]; then
    BREW_PATH="/opt/homebrew/bin/brew"
    if [ ! -f "$BREW_PATH" ]; then
        BREW_PATH="/usr/local/bin/brew"
    fi
fi

echo "--- [1/4] Checking system dependencies ---"
if [[ $(uname) == "Darwin" ]]; then
    if [ -f "$BREW_PATH" ]; then
        echo "Updating brew and installing espeak-ng..."
        $BREW_PATH install espeak-ng
    else
        echo "WARNING: Homebrew not found at common locations. Please install espeak-ng manually: brew install espeak-ng"
    fi
else
    echo "Please ensure espeak-ng is installed on your Linux system (e.g., sudo apt-get install espeak-ng)"
fi

echo "--- [2/4] Setting up virtual environment ---"
python3 -m venv venv-diagnostic
source venv-diagnostic/bin/activate

echo "--- [3/4] Installing requirements ---"
pip install --upgrade pip
pip install -r requirements.txt

echo "--- [4/4] Installing german-phoneme-validator ---"
pip install git+https://github.com/SergejKurtasch/german-phoneme-validator.git

echo "------------------------------------------------"
echo "Setup complete! Use scripts/run_app.sh to start the application."
