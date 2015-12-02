import itertools

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
