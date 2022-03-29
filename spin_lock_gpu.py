import torch

GB = 1024 ** 3

def create_large_tensor(tensor_size, device):
    return torch.zeros(tensor_size, device=device)

devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

for device in devices:
    create_large_tensor(5 * GB, device)

# Pause 
input()
