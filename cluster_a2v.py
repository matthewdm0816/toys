import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
import os.path as osp
import torch

a2v_path = osp.expanduser('~/data/bert-vqa/a2v_clip_top3k.pt')
embedded: np.ndarray = torch.load(a2v_path).numpy()
with open(lbl2idx_path, "rb") as f:
    lbl2idx, idx2lbl = pickle.load(f)
# embedded = embedded[2:]
print(embedded.shape, embedded)


kmeans = KMeans(n_clusters=20)
kmeans.fit(embedded)
print(kmeans.labels_)