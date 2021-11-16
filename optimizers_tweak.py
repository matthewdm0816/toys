import torch
from torch import tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import torch_optimizer as toptim
import torchvision
import torchvision.transforms as T
from icecream import ic
import types
from tqdm import tqdm
from tensorboardX import SummaryWriter


epochs = 20
batch_size = 256
otype = "AdaBelief"

writer = SummaryWriter(comment=otype)


transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
dataset = torchvision.datasets.FashionMNIST(
    root="FashionMNIST", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="FashionMNIST", train=False, download=True, transform=transform
)
loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
)
test_loader = DataLoader(
    dataset, batch_size=batch_size * 4, shuffle=True, num_workers=8, pin_memory=True
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone: nn.Module = torchvision.models.resnext101_32x8d()
        self.backbone: nn.Module = torchvision.models.resnet50()

        def _forward_impl(obj, x):
            # See note [TorchScript super()]
            x = obj.conv1(x)
            x = obj.bn1(x)
            x = obj.relu(x)
            x = obj.maxpool(x)

            x = obj.layer1(x)
            x = obj.layer2(x)
            x = obj.layer3(x)
            x = obj.layer4(x)

            x = obj.avgpool(x)
            x = torch.flatten(x, 1)
            # x = self.fc(x)b
            return x

        self.backbone._forward_impl = types.MethodType(_forward_impl, self.backbone)
        # self.backbone._forward_impl = _forward_impl.__get__(self.backbone, torchvision.models.ResNet)

        self.cls = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.backbone(x)
        # ic(x, x.shape)
        return self.cls(x)


model = SimpleModel().cuda()


def get_opimizer_from_type(model: nn.Module, otype: str):
    if otype == "RAdam":
        return toptim.RAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif otype == "Yogi":
        return toptim.Yogi(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif otype == "AdaBound":
        return toptim.AdaBound(model.parameters(), le=1e-3, weight_decay=1e-4)
    elif otype == "AdaBelief":
        return toptim.AdaBelief(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif otype == "AdaHessian":
        raise NotImplementedError
        return toptim.AdaHessian(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )  # 1e-2 as purposed in Apollo
    elif otype == "Apollo":
        return toptim.Apollo(
            model.parameters(), lr=1e-2, beta=0.9, weight_decay=1e-4
        )  # 1e-2 as purposed in Apollo
    else:
        raise NotImplementedError


# optimizer = toptim.Yogi(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = get_opimizer_from_type(model, otype)
loss_fct = nn.CrossEntropyLoss()
for eid in range(epochs):
    model.train()
    for bid, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = [tensor.cuda() for tensor in batch]
        inputs, labels = batch
        inputs = inputs.repeat(1, 3, 1, 1)
        # ic(inputs.shape)
        outputs = model(inputs)
        loss = loss_fct(outputs, labels)
        acc = torch.sum(outputs.detach().argmax(dim=-1).eq(labels)) / batch_size
        if (bid + 1) % 10 == 0:
            print(
                "[{eid}/{bid}]: Loss: {loss:.04f}, Accuracy: {acc:.2f}%".format(
                    eid=eid, bid=bid, loss=loss.item(), acc=100 * acc
                )
            )
        writer.add_scalar(
            tag="train/loss",
            scalar_value=loss.item(),
            global_step=bid + len(loader) * eid,
        )
        writer.add_scalar(
            tag="train/acc", scalar_value=100 * acc, global_step=bid + len(loader) * eid
        )
        loss.backward()
        optimizer.step()
        model.zero_grad()

    model.eval()
    print("Testing...")
    with torch.no_grad():
        correct = 0.0
        total_loss = 0.0
        total_elem = 0.0
        for bid, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch = [tensor.cuda() for tensor in batch]
            inputs, labels = batch
            inputs = inputs.repeat(1, 3, 1, 1)
            # ic(inputs.shape)
            outputs = model(inputs)
            loss = loss_fct(outputs, labels)

            total_elem += labels.shape[0]
            correct += torch.sum(outputs.detach().argmax(dim=-1).eq(labels)).item()
            total_loss += loss.item() * labels.shape[0]
        writer.add_scalar(
            tag="eval/loss", scalar_value=total_loss / total_elem, global_step=bid
        )
        writer.add_scalar(
            tag="eval/acc", scalar_value=100 * correct / total_elem, global_step=bid
        )
        print(
            "[{eid}]: Loss: {loss:.04f}, Accuracy: {acc:.2f}%".format(
                eid=eid, loss=total_loss / total_elem, acc=100 * correct / total_elem
            )
        )
