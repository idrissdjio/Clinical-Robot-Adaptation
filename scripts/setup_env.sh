#!/bin/bash
echo 'Setting up ClinAdapt environment...'
python3 -m venv clinadapt_env
source clinadapt_env/bin/activate
pip install numpy mujoco gymnasium torch pytest matplotlib
echo 'Done. Activate with: source clinadapt_env/bin/activate'
