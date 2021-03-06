from argparse import ArgumentParser
import pickle
import datasets
import torch 
from icecream import ic
import numpy as np
from matplotlib import pyplot as plt
import math
parser = ArgumentParser()
parser.add_argument(
    "--lbl2idx_path",
    type=str,
    default="/home/yangliu/data/vqav2/lbl2idx_separate_top3k_full.pkl",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/home/yangliu/data/vqav2/processed_tokenized_separate_top3k_full",
)
parser.add_argument(
    "--histogram_path", type=str, default="histogram.png"
)

args = parser.parse_args()

dataset = datasets.load_from_disk(args.dataset_path)
dataset = datasets.concatenate_datasets([dataset["train"], dataset["val"]])
with open(args.lbl2idx_path, "rb") as f:
    lbl2idx, idx2lbl = pickle.load(f)


# Get most frequent answer
def get_mode(samples, unk_id: int):
    lbls: torch.Tensor = torch.tensor(samples["labels"], requires_grad=False)
    mode = lbls.mode(dim=-1).values.numpy()
    mode[mode == -1] = unk_id  # replace all -1 to unk_id
    return {"answer_id": mode}


dataset = dataset.map(
    lambda samples: get_mode(samples, unk_id=-1),
    batched=True,
    num_proc=None,
)


# Histogram
answer_count = {}

def stat_answer_count(samples):
    for answer_id in samples["answer_id"]:
        answer_count[answer_id] = answer_count.get(answer_id, 0) + 1

dataset.map(stat_answer_count, batched=True, num_proc=None)

answer_count = {
    idx2lbl[answer_id]: count for answer_id, count in answer_count.items()
}

ic(answer_count)

nans = len(answer_count)

answer_prob = sorted(list(answer_count.items()), key=lambda t: t[-1], reverse=True)
ic(answer_prob)

def get_percentage(ns, k: int):
    return sum(ns[:k]) / sum(ns)

ns = [t[1] for t in answer_prob]
for k in [10, 50, 100, 200, 500, 1000, 1500, 2000]:
    ic(get_percentage(ns, k=k))    

xs = sum([[i] * t[-1] for i, t in enumerate(answer_prob)], start=[])
xs = np.array(xs)
# draw histogram
plt.hist(xs, bins=nans)
# save histogram image
plt.savefig(args.histogram_path, dpi=300)

xs = xs.max() - xs + 1
ic(xs[:100], len(xs))

def hill_estimator(xs, k: int):
    return (np.log(xs[:k]).sum() / k - np.log(xs[k])) ** (-1)

ic(hill_estimator(xs, k=10))
ic(hill_estimator(xs, k=100))
ic(hill_estimator(xs, k=1000))
ic(hill_estimator(xs, k=len(xs) // 10))
ic(hill_estimator(xs, k=len(xs) // 6))
ic(hill_estimator(xs, k=len(xs) // 5))
ic(hill_estimator(xs, k=len(xs) // 4))
ic(hill_estimator(xs, k=len(xs) // 3))
ic(hill_estimator(xs, k=len(xs) // 2))
ic(hill_estimator(xs, k=math.floor(len(xs) * 0.75)))



