import torch
import numpy as np
import os.path as osp
from skhubness.analysis import Hubness
a2v_path = osp.expanduser('~/data/bert-vqa/a2v_clip_top20k.pt')

embedded: np.ndarray = torch.load(a2v_path).numpy()
print(embedded.shape, embedded)

# Take 100 samples randomly
sample = embedded.copy()
np.random.shuffle(sample)
sample = sample[:300]

# Compute precomputed L1 


for metric in ["euclidean", "cosine"]:
    print(f"Hubness under {metric}")
    hub = Hubness(k=1, return_value='robinhood', algorithm='hnsw', metric=metric)
    hub.fit(embedded)
    print(hub.score())
    print(hub.score(X=sample))