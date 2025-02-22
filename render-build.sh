#!/bin/bash

pip install --upgrade --force-reinstall numpy

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
