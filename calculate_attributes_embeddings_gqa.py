#!/usr/bin/env python
# coding: utf-8

# In[31]:


import datasets, torch, transformers
import numpy as np
from icecream import ic
import logging, colorama
from tqdm import tqdm
import math
import os
from argparse import ArgumentParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_format = (
    colorama.Fore.MAGENTA
    + "[%(asctime)s %(name)s %(levelname)s] "
    + colorama.Fore.WHITE
    + "%(message)s"
)
import clip

ic(clip.available_models())


# In[32]:


vinvl_path = "/home/yangliu/data/vqav2/processed_tokenized_separate_top3k_new"
vinvl_datasets = datasets.load_from_disk(vinvl_path)
ic(vinvl_datasets)
ic(vinvl_datasets["train"]["tags"][0])


# In[33]:


# Find all tags words
tags_words = []
for split in vinvl_datasets.keys():
    for tag in vinvl_datasets[split]["tags"]:
        tags_words.extend(tag.split(" "))
# for tag in vinvl_datasets["train"]["tags"]:
#     tags_words.extend(tag.split(" "))

tags_words = list(set(tags_words))
# Show some of the tags
ic(tags_words[:10])
ic(len(tags_words))


# In[34]:


# Find 'urn' in all tags
for tag in vinvl_datasets["train"]["tags"]:
    if "urn" in tag.split(" "):
        print(tag)
        break


# In[35]:


class ObjectDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, key):
        return self.__getitem__(key)

    def purge(self):
        r"""converts all sub-elements into ObjectDict"""
        for k, v in self.items():
            if type(v) == dict:
                res = ObjectDict()
                res.update(v)
                res.purge()
                self[k] = res


parser = ArgumentParser()
parser.add_argument("--lbl2idx_path", type=str, default=)


opt = ObjectDict()
# opt.model_type = "clip"
opt.answer_bs = 16
opt.save_path = "../bert-vqa/a2v_rn50_union.pkl"  # dummy
opt.lbl2idx_path = "~/data/vqav2/lbl2idx_separate_top3k.pkl"
tags = sorted(tags_words)
device = "cuda:3"
opt.model_type = "RN50x16"


# In[36]:


with torch.no_grad():
    logging.info("Generating Tags Embeddings...")

    model, preprocess = clip.load(opt.model_type, device=device)
    embedded_tags = None
    for idx in tqdm(range(math.ceil(len(tags) / opt.answer_bs))):
        answers_chunk = tags[idx * opt.answer_bs : (idx + 1) * opt.answer_bs]
        answers_tok = clip.tokenize(
            answers_chunk,
        ).to(device)
        text_feature = model.encode_text(answers_tok)
        if embedded_tags is None:
            embedded_tags = text_feature.detach().cpu()  # take first token outputs
        else:
            embedded_tags = torch.cat(
                (embedded_tags, text_feature.detach().cpu()), dim=0
            )
embedded_tags = {tag: embedding.numpy() for tag, embedding in zip(tags, embedded_tags)}


# In[37]:


import pickle

with open(os.path.expanduser(opt.lbl2idx_path), "rb") as f:
    lbl2idx, idx2lbl = pickle.load(f)
answers = [idx2lbl[idx] for idx in sorted(idx2lbl.keys())]
with torch.no_grad():
    logging.info("Generating Words Embeddings...")

    model, preprocess = clip.load(opt.model_type, device=device)
    embedded_answers = None
    for idx in tqdm(range(math.ceil(len(answers) / opt.answer_bs))):
        answers_chunk = answers[idx * opt.answer_bs : (idx + 1) * opt.answer_bs]
        answers_tok = clip.tokenize(
            answers_chunk,
        ).to(device)
        text_feature = model.encode_text(answers_tok)
        if embedded_answers is None:
            embedded_answers = text_feature.detach().cpu()  # take first token outputs
        else:
            embedded_answers = torch.cat(
                (embedded_answers, text_feature.detach().cpu()), dim=0
            )
embedded_answers = {
    answer: embedding.numpy() for answer, embedding in zip(answers, embedded_answers)
}


# In[38]:


# Show common and diffenrent words
ic(len(answers))
common_keys = list(set(embedded_tags.keys()) & set(embedded_answers.keys()))
uncommon_keys = list(set(embedded_tags.keys()) - set(embedded_answers.keys()))
uncommon_keys = sorted(uncommon_keys)
ic(uncommon_keys[:10], common_keys[:10])
ic(len(uncommon_keys), len(common_keys))
ic(torch.norm(torch.from_numpy(embedded_tags["apple"]) - embedded_answers["apple"]))


# In[39]:


# Merge into one embedding
# embedded_a2v = torch.from_numpy(embedded_a2v)
embedded_diff = torch.stack(
    [torch.from_numpy(embedded_tags[uncommon_key]) for uncommon_key in uncommon_keys]
)
embedded_answers_base = torch.stack(
    [torch.from_numpy(embedded_answers[idx2lbl[idx]]) for idx in sorted(idx2lbl.keys())]
)
# unk_embedding = embedded_answers_base[-1:]
ic(embedded_diff.shape)
ic(embedded_answers_base.shape)
# Keep UNK the last one
# embedded_union = torch.cat([embedded_answers_base[:-1], embedded_diff, unk_embedding], dim=0)
embedded_union = torch.cat([embedded_answers_base, embedded_diff], dim=0)
ic(embedded_union.shape)


# In[40]:


# Merge pointers (lbl2idx and idx2lbl)
lbl2idx_union = {
    **lbl2idx,
    **{
        uncommon_key: len(lbl2idx) + i
        for i, uncommon_key in enumerate(sorted(uncommon_keys))
    },
}
# Keep UNK the last one
# lbl2idx_union["UNK"] = len(lbl2idx_union) - 1
idx2lbl_union = {idx: lbl for lbl, idx in lbl2idx_union.items()}
ic(idx2lbl_union[3333])
ic(idx2lbl_union[3577], lbl2idx_union[idx2lbl_union[3577]])
ic(lbl2idx_union["UNK"], idx2lbl_union[lbl2idx_union["UNK"]])
ic(lbl2idx_union["baseline"])


# In[41]:


# save merged embeddings and pointers
midfix = opt.model_type.replace("/", "_")
with open(os.path.expanduser(f"~/data/bert-vqa/a2v_{midfix}.pt"), "wb") as f:
    torch.save(embedded_answers_base, f)
with open(os.path.expanduser(f"~/data/bert-vqa/a2v_{midfix}_union.pt"), "wb") as f:
    torch.save(embedded_union, f)
with open(os.path.expanduser(f"~/data/bert-vqa/lbl2idx_{midfix}_union.pkl"), "wb") as f:
    pickle.dump((lbl2idx_union, idx2lbl_union), f)


# In[42]:


# Compare embeddings
embeddings_old = torch.load(os.path.expanduser("~/data/bert-vqa/a2v_clip_top3k.pt"))
embeddings_new = torch.load(os.path.expanduser("~/data/bert-vqa/a2v_rn50_union.pt"))


# In[43]:


ic(embeddings_old.shape, embeddings_old.dtype)
ic(embeddings_new.shape, embeddings_new.dtype)
N = 100
ic(torch.norm(embeddings_old[:N] - embeddings_new[:N], dim=-1).mean())
ic(torch.norm(embeddings_old[:N] - embeddings_new.to(torch.float32)[:N], dim=-1).mean())


# In[44]:


sorted(idx2lbl.keys()) == list(range(len(idx2lbl)))
with open(os.path.expanduser("~/data/vqav2/lbl2idx_separate_top3k.pkl"), "rb") as f:
    lbl2idx_old, idx2lbl_old = pickle.load(f)

for k, v in lbl2idx_old.items():
    assert lbl2idx_old[k] == lbl2idx_union[k]

for k, v in idx2lbl_old.items():
    assert (
        idx2lbl_old[k] == idx2lbl_union[k]
    ), f"idx2lbl_old[{k}] = {idx2lbl_old[k]}, idx2lbl_union[{k}] = {idx2lbl_union[k]}"


# In[45]:


ic(idx2lbl_union[3129], lbl2idx_union["UNK"])


# In[ ]:
