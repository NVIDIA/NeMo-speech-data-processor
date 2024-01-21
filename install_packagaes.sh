#!/bin/bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/main.txt

for arg in "$@"; do
    if [ "$arg" == "contribute" ]; then
        echo "Installing packages for contribution..."
        pip install pytest
        pip install pre-commit
        pre-commit install
        pip install -r requirements/tests.txt
        echo "Contribution packages installation completed."
    fi

    if [ "$arg" == "asr_models" ]; then
        echo "Installing ASR model-related packages..."
        pip install -U openai-whisper
        pip install transformers
        pip install accelerate
        echo "ASR model packages installation completed."
    fi

    if [ "$arg" != "contribute" ] && [ "$arg" != "asr_models" ]; then
        echo "Invalid option: $arg. Please provide 'contribute' or 'asr_models'."
        exit 1
    fi
done

echo "The Environment is set up successfully"
