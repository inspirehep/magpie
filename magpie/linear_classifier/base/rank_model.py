"""
Implementation of pairwise ranking using scikit-learn LinearSVC
Link: https://gist.github.com/agramfort/2071994

Reference: "Large Margin Rank Boundaries for Ordinal Regression",
           R. Herbrich, T. Graepel, K. Obermayer.

Original authors: Fabian Pedregosa <fabian@fseoane.net>
                  Alexandre Gramfort <alexandre.gramfort@inria.fr>

Edited by: Jan Stypka <jan.stypka@cern.ch>
"""

import itertools
import numpy as np

from sklearn.linear_model import SGDClassifier


def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are chosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


def fast_transform_pairwise(x, y):
    """ The same as transform_pairwise(), but a little optimised for our usecase
    :param x: the same as in transform_pairwise
    :param y: the same as in transform_pairwise

    :return x, y: the same as in transform_pairwise
    """
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]

    hits = np.where(y[:, 0])[0]
    comparisons = len(hits) * (len(y) - len(hits))
    row_to_write = 0
    print comparisons, x.shape
    x_new = np.zeros((comparisons, len(x[0])))
    y_new = np.zeros(comparisons, dtype=np.int16)

    for h in hits:
        for i in xrange(len(x)):
            try:
                if i == h:
                    continue
            except Exception:
                import ipdb; ipdb.set_trace()
                raise

            x_new[row_to_write] = x[h] - x[i]
            y_new[row_to_write] = np.sign(y[h, 0] - y[i, 0])

            # Balanced output classes
            if y_new[row_to_write] != (-1) ** row_to_write:
                y_new[row_to_write] = - y_new[row_to_write]
                x_new[row_to_write] = - x_new[row_to_write]

            row_to_write += 1

    assert row_to_write == comparisons

    return x_new, y_new


class RankSVM(SGDClassifier):
    """Performs pairwise ranking with an underlying SGDClassifier model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`SGDClassifier` for a full description of parameters.
    """

    def __init__(self, **kwargs):
        super(RankSVM, self).__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        kwargs: dictionary
        Returns
        -------
        self
        """
        X_trans, y_trans = fast_transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans, **kwargs)
        return self

    def partial_fit(self, X, y, **kwargs):
        """
        Partially fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        kwargs: dictionary
        Returns
        -------
        self
        """
        X_trans, y_trans = fast_transform_pairwise(X, y)
        super(RankSVM, self).partial_fit(X_trans, y_trans, **kwargs)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            product = np.dot(X, self.coef_[0].T)
            return np.argsort(product)
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)
