import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

class ToyDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data["data"])

    def __getitem__(self, index) -> dict:
        return {
            "data": self.data["data"][index],
            "label": self.data["label"][index],
        }


def concat_dict(d1: dict, d2: dict) -> dict:
    d3 = dict()
    for k in d1.keys():
        d3[k] = torch.cat([d1[k], d2[k]], dim=0)  # concat tensors

    return d3


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(2, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 2),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x) -> torch.Tensor:
        return self.nn(x)


def compose_arc_data(
    n_samples: int,
    origin: torch.Tensor,
    radius_min: float = 1.0,
    radius_max: float = 2.0,
    angle_min: float = 0,
    angle_max: float = np.pi,
):
    rs = torch.rand(n_samples) * (radius_max - radius_min) + radius_min
    thetas = torch.rand(n_samples) * (angle_max - angle_min) + angle_min
    ic(thetas.max(), thetas.min())
    sins = thetas.sin()
    coss = thetas.cos()
    xs = rs * coss
    ys = rs * sins
    return torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=-1) + origin


def find_decision_boundary(
    model: nn.Module,
    x_max: float,
    x_min: float,
    y_max: float,
    y_min: float,
    n_samples: int,
):
    xseeds = torch.rand(n_samples) * (x_max - x_min) + x_min
    yseeds = torch.rand(n_samples) * (y_max - y_min) + y_min
    seeds = torch.cat((xseeds.view(-1, 1), yseeds.view(-1, 1)), dim=-1)
    for seed in seeds:
        seed = nn.Parameter(seed)
        optimizer = AdamW([seed], lr=1e-2)
        for _ in range(100):
            output = model(seed)


def find_decision_boundary_search(
    model: nn.Module,
    x_max: float,
    x_min: float,
    y_max: float,
    y_min: float,
    n_samples: int,
):
    x = np.linspace(x_min, x_max, num=n_samples)
    y = np.linspace(y_min, y_max, num=n_samples)
    seeds = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])  # [N^2, 2]
    seeds = torch.from_numpy(seeds).float()
    ic(seeds.shape)
    with torch.no_grad():
        values = model(seeds)
    scores = -torch.abs(values[:, 0] - values[:, 1])  # [N^2]
    ic(scores.shape)
    top_indices = torch.topk(scores, k=n_samples).indices  # => [N^2] indices
    return seeds[top_indices]


def plot_prediction(
    model: nn.Module,
    data: dict,
    x_max: float,
    x_min: float,
    y_max: float,
    y_min: float,
    save_name: str = "prediction.png",
):
    boundary_points = find_decision_boundary_search(
        model, x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min, n_samples=300
    )
    boundary_data = {
        "data": boundary_points,
        "label": torch.ones(boundary_points.shape[0]) * 2,
    }
    full_data = concat_dict(data, boundary_data)
    full_data["x"] = full_data["data"][:, 0]
    full_data["y"] = full_data["data"][:, 1]
    label_dict = {0: "Postive", 1: "Negative", 2: "Boundary"}
    size_dict = {0: 1, 1: 1, 2: 0.02}
    full_data["label_name"] = [
        label_dict[label.item()] for label in full_data["label"].long()
    ]
    full_data["size"] = [size_dict[label.item()] for label in full_data["label"].long()]

    fig = plt.figure(figsize=(8, 8))
    sns.scatterplot(x="x", y="y", data=full_data, hue="label_name", size="size")
    plt.xlim(-2.5, 4)
    plt.ylim(-2.5, 2.5)
    plt.savefig(fname=save_name, dpi=300)
    plt.clf()
    


def make_movie(path: str):
    from pathlib import Path
    import imageio
    image_path = Path('predictions')
    images = sorted(list(image_path.glob('*.png')))
    ic(images)
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))
    imageio.mimwrite('animated_predictions.gif', image_list, fps=1)


if __name__ == "__main__":
    n_half_samples = 1000
    composed_dataset_pos = {
        "data": compose_arc_data(n_half_samples, torch.tensor([0, 0])),
        "label": torch.bernoulli(torch.ones(n_half_samples) * 0.95),
    }
    composed_dataset_neg = {
        "data": compose_arc_data(
            n_half_samples,
            torch.tensor([1.5, 0.5]),
            angle_min=np.pi,
            angle_max=np.pi * 2,
        ),
        "label": torch.bernoulli(torch.ones(n_half_samples) * 0.05),
    }
    composed = concat_dict(composed_dataset_pos, composed_dataset_neg)
    toy_dataset = ToyDataset(composed)
    loader = DataLoader(toy_dataset, batch_size=100, shuffle=True)
    model = ToyModel()
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    for i in tqdm(range(200)):
        for batch in loader:
            # ic(batch)
            output = model(batch["data"])
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(output, batch["label"].long())
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if (i + 1) % 10 == 0:
            ic(i, loss)
            plot_prediction(
                model,
                composed,
                x_max=4,
                x_min=-2.5,
                y_max=2.5,
                y_min=-2.5,
                save_name="predictions/prediction-{:04d}.png".format(i),
            )
    make_movie("predictions")
