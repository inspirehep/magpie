from magpie.candidates.keyword_token import add_token
from magpie.utils.stemmer import stem


def get_anchors(words, ontology):
    """ Match single words in the document over the topology to find `anchors`
    i.e. matches that later on can be used for ngram generation or
    subgraph extraction """
    trie = ontology.get_trie()
    anchors = dict()

    for position, word in enumerate(words):
        if word in trie:
            add_token(word, anchors, position, ontology)
        elif stem(word) in trie:
            add_token(stem(word), anchors, position, ontology, form=word)

    return anchors.values()
