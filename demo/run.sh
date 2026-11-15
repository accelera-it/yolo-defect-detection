#!/usr/bin/env sh
# ui.py loads models/ relative to the repository root, so start from there
cd "$(dirname "$0")/.." && streamlit run ui.py