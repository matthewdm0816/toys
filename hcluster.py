from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import wordnet as wn
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

sample_words = [
    "cat",
    "dog",
    "tiger",
    "lion",
    "elephant",
    "giraffe",
    "zebra",
    "bear",
    "cow",
    "pig",
    "sheep",
    "monkey",
    "squirrel",
    "rabbit",
    "deer",
    "camel",
    "crocodile",
    "dolphin",
    "whale",
    "shark",
    "turtle",
    "snake",
    "fish",
    "spider",
    "beetle",
    "ant",
    "bee",
    "butterfly",
    "caterpillar",
    "cockroach",
    "fly",
    "ladybug",
    "mosquito",
    "pillbug",
    "snail",
    "spider",
    "termite",
    "wasp",
    "day",
    "hit",
    "skateboard",
    "light",
    "dark",
    "bright",
]


def agglocluster_words(words: List[str], clusters: int = 5):
    # Compute wup_similarity
    words = sorted(words)
    wup_similarity_generator = lambda w1, w2: wn.wup_similarity(
        wn.synsets(w1)[0], wn.synsets(w2)[0]
    )
    similarity = [[wup_similarity_generator(w1, w2) for w2 in words] for w1 in words]
    # Compute agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=clusters, affinity="precomputed", linkage="average"
    )
    clustering.fit(similarity)
    return clustering, clustering.labels_, clustering.children_


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    precomputed_distances = [
        
    ]

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

if __name__ == "__main__":
    # print all synsets of sample words
    # for word in sample_words:
    #     print(word, wn.synsets(word))

    # filter words without synsets
    sample_words = [word for word in sample_words if len(wn.synsets(word)) > 0]

    model, labels, children = agglocluster_words(sample_words, clusters=3)
    print(labels)
    print(children)

    # Plot dendrogram
    plt.title("Hierarchical Clustering Dendrogram")
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.savefig("cluster_wn.png", dpi=500)

