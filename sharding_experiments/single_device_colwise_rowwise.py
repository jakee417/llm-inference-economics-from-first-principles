import torch
import torch.nn.functional as F

def split_columnwise(A, num_splits):
    return torch.split(A, A.size(1) // num_splits, dim=1)

def split_rowwise(A, num_splits):
    return torch.split(A, A.size(0) // num_splits, dim=0)

def normal_forward_pass(X, A, B, f):
    Y = f(torch.mm(X, A))
    Z = torch.mm(Y, B)
    return Z

def tensor_parallel_forward_pass(X, A, B, f):
    A1, A2 = split_columnwise(A, 2)
    B1, B2 = split_rowwise(B, 2)
    Y1 = f(torch.mm(X, A1))
    Y2 = f(torch.mm(X, A2))
    Z1 = torch.mm(Y1, B1)
    Z2 = torch.mm(Y2, B2)
    Z = Z1 + Z2
    return Z

# Set up random seed for reproducibility
torch.manual_seed(0)

X = torch.randn(2, 2)
A = torch.randn(2, 2)
B = torch.randn(2, 2)
Z = tensor_parallel_forward_pass(X, A, B, torch.tanh)
Z_normal = normal_forward_pass(X, A, B, torch.tanh)
print(torch.allclose(Z, Z_normal))  # outputs: True

if __name__ =="__main__":
    # Assuming W is of shape (out_features, in_features)
    # and X is of shape (batch_size, in_features)

    # W^T * X
    output1 = torch.mm(W.t(), X.t()).t()

    # X * W
    output2 = torch.mm(X, W.t())

    # Check if they're equal
    print(torch.allclose(output1, output2))  # Should print True