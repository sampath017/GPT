#!/bin/bash

# Variables
REPO_NAME="GPT"
GITHUB_REPO_URL="https://github.com/sampath017/GPT.git"
CONDA_ENV_FILE="environment.yml"

# Clone the GitHub repository
echo "Cloning the GitHub repository..."
git clone $GITHUB_REPO_URL

# Navigate to the repository directory
cd $REPO_NAME

# Install libs
echo "Updating the base conda environment from $CONDA_ENV_FILE..."
conda env update -n base -f $CONDA_ENV_FILE

echo "Setup complete."
