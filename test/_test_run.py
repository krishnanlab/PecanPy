import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_label(fp, min_size=10):
    """Load label from file and construct label matrix.

    Args:
        fp(str): path to the label ('.tsv') file, which contains two columns,
            node ID and the corresponding label ID.
        min_size(int): minimm size of labelset below which are discarded

    Returns:
        A label matrix (n_nodes by n_classes) and a dictionary mapping from
            node id to node index.

    """
    print(f"Loading label file from: {fp}")

    labelsets = {}
    with open(fp, "r") as f:
        for line in f:
            ID, label = line.strip().split("\t")
            label = int(label)
            ID = int(ID)

            if label not in labelsets:
                labelsets[label] = []

            labelsets[label].append(ID)

    pop_lst = []
    for label, labelset in labelsets.items():
        if len(labelset) < 10:
            pop_lst.append(label)

    for label in pop_lst:
        labelsets.pop(label)

    IDs = set()
    for labelset in labelsets.values():
        IDs.update(labelset)

    IDmap = {j: i for i, j in enumerate(IDs)}

    n_nodes = len(IDmap)
    n_classes = len(labelsets)
    label_mat = np.zeros((n_nodes, n_classes), dtype=bool)

    for i, labelset in enumerate(labelsets.values()):
        for ID in labelset:
            label_mat[IDmap[ID], i] = True

    return label_mat, IDmap


def load_emb(fp):
    """Load embedding from `.emb` file
    Args:
        fp(str): path to `.emb` file
    Returns:
        emb: embedding vectors as numpy matrix
        IDmap: mapping from ID to corresponding index
    """
    print(f"Loading embedding file from: {fp}")
    emb = np.loadtxt(fp, delimiter=" ", skiprows=1)
    IDmap = {j: i for i, j in enumerate(emb[:, 0].astype(int).tolist())}

    return emb[:, 1:], IDmap


def test(
    emb,
    label_mat,
    emb_IDmap,
    label_IDmap,
    n_splits,
    random_state,
    shuffle,
    verbose,
):
    """Test embedding performance
    Perform node classification using L2 regularized Logistic Regression
    with 5-Fold Cross Validation
    """
    n_classes = label_mat.shape[1]
    label_IDs = list(label_IDmap)
    emb_idx = [emb_IDmap[ID] for ID in label_IDs]
    x = emb[emb_idx]

    splitter = StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=shuffle
    )
    mdl = LogisticRegression(
        penalty="l2", solver="lbfgs", warm_start=False, max_iter=1000
    )

    y_true_all = []
    y_pred_all = []

    for i in range(n_classes):
        y = label_mat[:, i]
        label = i + 1

        y_true = np.array([], dtype=bool)
        y_pred = np.array([])

        for j, (train, test) in enumerate(splitter.split(y, y)):
            if verbose:
                print(
                    f"Class #{label:>4d},\tfold {j+1:>2d}/{n_splits:<2d}",
                    flush=True,
                    end="\r",
                )
            mdl.fit(x[train], y[train])

            y_true = np.append(y_true, y[test])
            y_pred = np.append(y_pred, mdl.decision_function(x[test]))

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    if verbose:
        print("")

    return y_true_all, y_pred_all


def eval_emb(
    emb,
    label_mat,
    emb_IDmap,
    label_IDmap,
    n_splits=5,
    random_state=None,
    shuffle=False,
    verbose=True,
):
    """Evaluate predictions using auROC
    Args:
        emb(:obj:`np.ndarray`): embedding matrix
        label_mat(:obj:`np.ndarray`): label matrix
        emb_IDmap(dict of `str`:`int`): IDmap for embedding matrix
        label_IDmap(dict of `str`:`int`): IDmap fro label matrix
        n_splits(int): number folds in stratified k-fold cross validation
        random_state(int): random state used to generate split
        shuffle (bool): whether or not to shuffle splits
        verbose (bool): whether or not to show evaluation progress
    """
    y_true_all, y_pred_all = test(
        emb,
        label_mat,
        emb_IDmap,
        label_IDmap,
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle,
        verbose=verbose,
    )

    auroc_all = [
        roc_auc_score(y_true, y_pred)
        for y_true, y_pred in zip(y_true_all, y_pred_all)
    ]

    return auroc_all
