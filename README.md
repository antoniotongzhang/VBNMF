# Variational Bayesian Non-negative Matrix Factorization (VBNMF)

This repository contains the implementation of Variational Bayesian Non-negative Matrix Factorization (VBNMF) using PyTorch.

## Dependencies
- Python
- NumPy
- PyTorch
- scikit-learn

## Description
This is a course final project for COSC 5P77. The code contains a novel implementation of VBNMF in Python and PyTorch, along with an evaluation on the Breast Cancer dataset.

## Parameters
- `V`: Input data matrix.
- `K`: Number of components.
- `alpha_W`, `beta_W`, `alpha_H`, `beta_H`: Hyperparameters for the Gamma distribution priors.
- `max_iters`: Maximum number of iterations for updating the decomposed matrices.

## Output
- The output displays the original matrix V, along with the updated matrices W and H, as well as the reconstruction error between the original and reconstructed matrices.

