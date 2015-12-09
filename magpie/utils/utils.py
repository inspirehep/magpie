import cPickle as pickle
import itertools
import os

MAX_PERMUTATION_LENGTH = 3


def get_all_permutations(phrase):
    """ Generate all word permutations in a string. If a string contains more
     than 3 words, only the original phrase is returned. Otherwise all
     permutations of the phrase are generated and returned as a list. """
    words = phrase.split()

    if len(words) > MAX_PERMUTATION_LENGTH:
        return [phrase]

    permutations = []
    for p in itertools.permutations(words):
        permutations.append(' '.join(p))

    return permutations


def save_to_disk(path_to_disk, obj, overwrite=False):
    """ Pickle an object to disk """
    dirname = os.path.dirname(path_to_disk)
    if not os.path.exists(dirname):
        raise ValueError("Path " + dirname + " does not exist")

    if not overwrite and os.path.exists(path_to_disk):
        raise ValueError("File " + path_to_disk + "already exists")

    try:
        pickle.dump(obj, open(path_to_disk, 'wb'))
    except pickle.PickleError:
        raise ValueError("Failed to save model to " + path_to_disk)


def load_from_disk(path_to_disk):
    """ Load a pickle from disk to memory """
    if not os.path.exists(path_to_disk):
        raise ValueError("File " + path_to_disk + " does not exist")

    try:
        obj = pickle.load(open(path_to_disk, 'rb'))
    except pickle.PickleError:
        raise ValueError("Failed to load the model from " + path_to_disk)
    return obj
