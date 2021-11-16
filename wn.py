from nltk.corpus import wordnet as wn

word = "dog.n.01"
# Calculate all ancestors of the word
print(wn.synset(word).hypernyms())


def calc_min_depth_wn(word, verbose=False):
    word_sn = wn.synsets(word)[0]
    trees = word_sn.tree(lambda s: s.hypernyms())
    if verbose:
        print(trees)
    def calc_tree_depth(tree):
        # Calulate recursively nested list min depth
        if len(tree) == 1:
            return 1
        else:
            return 1 + min([calc_tree_depth(subtree) for subtree in tree[1:]])

    return calc_tree_depth(trees)
