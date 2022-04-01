from argparse import ArgumentParser
import pickle
import datasets
import torch 
from icecream import ic
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
