#!/bin/bash

# Set variable to where your source resides
export PROJECT_DIR="${HOME}/ufm"
echo "Entering $PROJECT_DIR"
cd $PROJECT_DIR

# Create a Python virtual environment in the home directory if it doesn't already exist
if [ ! -d "${PROJECT_DIR}/venv" ]; then
	echo "Creating new venv in ${PROJECT_DIR}/venv"
	python3 -m venv "${PROJECT_DIR}/venv"
fi

# Activate the virtual environment
echo "Entering venv"
source "${PROJECT_DIR}/venv/bin/activate"

# Update pip from their ancient version 9
pip install --upgrade pip

echo "Installing python requirements"
pip3 install -r requirements.txt

# Update PYTHONPATH to include the project directory
# This change is only effective within this script and the activated venv session
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
echo "Updating PYTHONPATH with project directory..."
echo "PYTHONPATH=${PYTHONPATH}"

