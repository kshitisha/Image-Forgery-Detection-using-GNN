#!/bin/bash

echo "Setting up Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo " Installing Libraries (optimized for M1)..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install torch-geometric
pip install scikit-learn scikit-image opencv-python matplotlib networkx

echo "Setup complete! To start:"
echo "source venv/bin/activate"
echo "python main.py"
