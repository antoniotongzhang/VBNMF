import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

#load breast cancer dataset
dataset = load_breast_cancer()
X = dataset.data  

#use standard scaler from sklearn library
scaler = StandardScaler()
#standardize the features
X_scaled = scaler.fit_transform(X)

#find the minimum value for all features
x_min = np.min(X_scaled)
#apply a shift to make values non-negative
X_shifted = X_scaled - x_min

#convert to PyTorch tensor
V = torch.tensor(X_shifted, dtype=torch.float32)

#print the matrix V
print("Matrix V:")
print(V)

#method to implement VBNFM
def vbnmf(V, K, alpha_W, beta_W, alpha_H, beta_H, max_iters):
    #NxD matrix
    N, D = V.shape

    #initialize W and H randomly, generate random numbers from a uniform distribution on the interval between 0 and 1(including 0)
    W = torch.rand(N, K)
    H = torch.rand(K, D)

    #print the matrix W
    print("Matrix W:")
    print(W)
    
    #print the matrix H
    print("Matrix H:")
    print(H)
    
    #update W and H iteratively
    for _ in range(max_iters):
        alpha_W_new = alpha_W + (H**2).sum(dim=1)
        beta_W_new = beta_W + torch.matmul(V,H.t())
        W = alpha_W_new / beta_W_new
        
        alpha_H_new = alpha_H + (H**2).sum(dim=0)
        beta_H_new = beta_H + torch.matmul(W.t(),V)
        H = alpha_H_new / beta_H_new
       
    return W, H

#K is the number of components
#change this number to try with different number of components. I used 5, 10, 15,
#20, 25, 30, 60, and 90 in my experiments
K = 5  
#K = 10
#K = 15
#K = 20
#K = 25
#K = 30
#K = 60
#K = 90

#initialize parameters, I used 100 for all of them here
para = 100
alpha_W_init = para
beta_W_init = para
alpha_H_init = para
beta_H_init = para

#maximum iterations
ite = 100

#run VBNMF
#W_res and H_res will be the final updated matrices, which will be used later to recompose the matrix.
W_res, H_res = vbnmf(V, K, alpha_W_init, beta_W_init, alpha_H_init, beta_H_init, ite)

#perform matrix multiplication
recon = torch.matmul(W_res, H_res)

#print the reconstructed matrix
print("Reconstructed Matrix:")
print(recon)

#reconstruction error for VBNMF, calculated using second norm
reconstruction_error = np.linalg.norm(V - recon)
#print the reconstruction error
print("Reconstruction Error:", reconstruction_error)
