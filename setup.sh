#!/bin/zsh

# Create the environment prefix
conda create --prefix ./env python=3.10 -y

# Define the path to the local pip binary
LOCAL_PIP="./env/bin/pip"

# Install Requirements
$LOCAL_PIP install --prefix ./env -r requirements.txt

