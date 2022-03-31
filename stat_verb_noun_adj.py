import datasets
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
import torch
from argparse import ArgumentParser
import pickle
from icecream import ic

POS_NAMES = {
    "noun": ["NN", "NNS", "NNP", "NNPS"],
    "verb": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "adj": ["JJ", "JJR", "JJS"],
}


def tag_label(label):
    return pos_tag([label])[0][1]


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

# Answer Counting
lbl2pos = {lbl: tag_label(lbl) for lbl in lbl2idx}

ans_type2lbl = {
    ans_type: {lbl for lbl, pos in lbl2pos.items() if pos in POS_NAMES[ans_type]}
    for ans_type in POS_NAMES
}

count_ans_type = {ans_type: len(ans_type2lbl[ans_type]) for ans_type in POS_NAMES}
ic(count_ans_type)

# Question Counting

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


def filter_by_type(sample, type: str):
    # print(sample)
    return idx2lbl[sample["answer_id"]] in ans_type2lbl[type]


typed_dataset = {
    ans_type: dataset.filter(
        lambda sample: filter_by_type(sample, ans_type),
        # input_columns=["answer_id"],
    )
    for ans_type in POS_NAMES
}
count_ans_type_Q = {
    ans_type: typed_dataset[ans_type].num_rows for ans_type in POS_NAMES
}

ic(count_ans_type_Q)
