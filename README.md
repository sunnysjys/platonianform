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
   ```bash
   python3 disentangling-vae-master/main.py VAE_dsprites -d dsprites
3. Running Experiments
   ```bash
   python3 disentangling-vae-master/main.py VAE_dsprites --is-eval-only --is-metrics
5. Testing and Evaluation
   ```bash
   python3 disentangling-vae-master/main.py VAE_dsprites --is-eval-only



## Experiments Description

### Experiment 1: Varying Top and Bottom Y Coordinates Across All X
This experiment varies the y-coordinate at its extreme values (0 and 31) while iterating over all x-coordinates. It aims to observe the differences in latent space representations for these extreme y-values across all x-values.

### Experiment 2: Holding X Constant and Incrementally Increasing Y
This experiment keeps the x-coordinate constant and incrementally increases the y-coordinate from 0 to 31. It helps in visualizing how changes in the y-coordinate alone can affect the latent space representation.

### Experiment 3: Customizable Experiment 2
A variation of Experiment 2, where the shape can be controlled and multiple x-coordinates can be set as constant. This allows for a more detailed analysis of how specific shapes and x-coordinates influence the latent space.

### Experiment 4: Manipulation of Multiple Latent Dimensions
In this experiment, the user can manipulate multiple latent dimensions to observe changes in the generated output. This provides insights into the effects of varying specific latent dimensions.

### Experiment 5: Generating Delta Vector Between Two Shapes
This experiment generates a delta vector between two shapes by holding rotation constant. It is useful for understanding the relationship and differences in latent space between different shapes.

### Experiment A: Heatmap Generation for Linear Layer Neurons
Holding shape, scale, and rotation constant, this experiment varies (x, y) positions and measures the mean and standard deviation of neurons in the linear layers. It generates heatmaps for each neuron, providing a visual representation of neuron activations.

### Experiment B: Linear Layer Outputs for All Feature Combinations
This experiment generates the first linear layer output for all possible feature combinations, providing comprehensive data on how different features affect the linear layer's response.

### Experiment C: Neuron Firing Analysis for Given X Values
Given specific x-values, this experiment identifies neurons that fire across all feature combinations. It helps in understanding which neurons are most responsive to certain x-coordinates.

### Experiment D: Firing Percentage of Neurons Across All X Values
Similar to Experiment C, this experiment plots the firing percentage of specified neurons across all x-values, offering a broader view of neuron behavior over the entire range of x-coordinates.

### Experiment E: Neuron Firing Z-Score Analysis for X Values
This experiment calculates the z-score of neuron firing frequency for given x-values, offering a statistical perspective on neuron activations.

### Experiment F: Z-Score Analysis Across All X Values for Specific Neurons
It extends Experiment E to cover all x-values for specified neurons, plotting the average z-score to provide a comprehensive view of neuron behavior.

### Experiment G: Bayesian Probability Calculations for Neuron Firing
For given neurons and x-values, this experiment calculates various probabilities, including the chance of a neuron firing given an x-value, providing insight into neuron behavior from a probabilistic standpoint.

### Experiment H: Bayesian Analysis Across All X Values
This experiment conducts a Bayesian analysis similar to Experiment G but extends it across all x-values for given neurons.

### Experiment I: Bayesian Analysis for All Neurons
An extensive version of Experiment H, it repeats the Bayesian analysis for all neurons in the model.

### Experiment J: Z-Score Analysis for All Neurons
This experiment conducts a z-score analysis (similar to Experiment F) for all neurons, offering a detailed statistical perspective on neuron activations.

### Experiment K: Output Value Analysis for Specific Neurons and X Values
Focusing on specific neurons and x-values, this experiment plots the output values of neurons, providing a direct view of neuron responses to particular feature combinations.
