# Variational Bayesian Non-negative Matrix Factorization (VBNMF)

This repository contains the implementation of Variational Bayesian Non-negative Matrix Factorization (VBNMF) using PyTorch.

## Dependencies
- Python
- Scikit-learn (used to input the Breast Cancer Wisconsin dataset and apply standard scaler for data standardization)
- NumPy (used to find minimum values for scaled features and apply the second norm)
- PyTorch (used to create tensors of matrices and perform matrix multiplication)


## Description
This is a course final project for COSC 5P77. The code contains a novel implementation of VBNMF in Python and PyTorch, along with an evaluation on breast cancer wisconsin dataset.

## Parameters
- `V`: Input data matrix.
- `K`: Number of components.
- `alpha_W`, `beta_W`, `alpha_H`, `beta_H`: Hyperparameters for the Gamma distribution priors.
- `max_iters`: Maximum number of iterations for updating the decomposed matrices.

## Output
- The output displays the original matrix V, along with the updated matrices W and H, as well as the reconstruction error between the original and reconstructed matrices.

