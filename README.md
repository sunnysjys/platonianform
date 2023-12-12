# README for platonianform Codebase

## Overview

This codebase accompanies the VAE Disentanglement Study, focusing on exploring the intricate neuron-level representations within Variational Autoencoders (VAEs). It includes a range of experiments designed to dissect and understand the relationships between latent dimensions, decoder features, and image properties in VAEs.

## Prerequisites

- Python 3.x
- PyTorch
- Numpy
- Matplotlib
- Seaborn
- SciPy

## Installation

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/sunnysjys/platonianform)https://github.com/sunnysjys/platonianform/

2. **Setup a Virtual Enviroment**:
   ```bash
   python -m venv vae-env
   source vae-env/bin/activate

4. **Install Dependencies**:
   ```bash
   pip install requirements.txt

## Structure
- `main.py`: Defines the entry point for running training and eval 
- `plot_latent_space.py`: Lays out all experiments mentioned in the paper
- `requirements.txt`: Lists all the necessary Python packages.

## Usage
1. Training the Model
   `python3 disentangling-vae-master/main.py VAE_dsprites -d dsprites`
3. Running Experiments
   `python3 disentangling-vae-master/main.py VAE_dsprites --is-eval-only --is-metrics`
5. Testing and Evaluation
   `python3 disentangling-vae-master/main.py VAE_dsprites --is-eval-only`

   
