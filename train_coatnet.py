import torch
import torch.nn as nn
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100
import os.path as osp
import torchvision.transforms as transforms
import pretty_errors
from icecream import ic
from coatnet import coatnet_0, coatnet_1, coatnet_2, coatnet_3, coatnet_4, count_parameters

epochs = 100

# ImageNet standard transform
# transform = torch.nn.Sequential(
#     T.ToTensor(),
#     T.Resize([224, 224]),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# )
# transform = torch.jit.script(transform)
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Define datasets and dataloaders
# ImageNet1K Dataset
# dataset_root = osp.expanduser("~/data/imagenet1k")
# datasets = {
#     split: ImageNet(root=dataset_root, split=split, transform=transform)
#     for split in ["train", "val"]
# }
# CIFAR100 Dataset
dataset_root = osp.expanduser("~/data/cifar100")
datasets = {
    split: CIFAR100(root=dataset_root, train=(split == "train"), transform=transform, download=True)
    for split in ["train", "val"]
}
dataloaders = {
    split: torch.utils.data.DataLoader(
        datasets[split], batch_size=128, shuffle=True, num_workers=4
    )
    for split in ["train", "val"]
}


# Init CoAtNet model
model = coatnet_4().cuda()
loss_fn = nn.CrossEntropyLoss()
# Define Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5,
    amsgrad=False,
)
# Print model info
print(model)
print(count_parameters(model))

# Print total batches
total_batches = len(dataloaders["train"])
print(f"Total batches: {total_batches}")


# Train CoAtNet
for eid in range(epochs):
    # Logging
    print("Epoch {}".format(eid))
    # Set batch time counters
    batch_time = 0
    for i, (inputs, targets) in enumerate(dataloaders["train"]):
        # Train model
        model.train()
        outputs = model(inputs.cuda())
        loss = loss_fn(outputs, targets.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()   
        # Logging
        if i % 50 == 0:
            print("Training loss: {}".format(loss.item()))
            # print elapsed time
            elapsed = batch_time / 50
            

    # Validate model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(dataloaders["val"]):
            outputs = model(inputs.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print("Validation accuracy: {}".format(correct / total))

    