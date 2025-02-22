#!/bin/bash
pip install torch==2.4.1 --no-cache-dir

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
