from magpie.candidates.keyword_token import add_token
from magpie.utils.stemmer import stem


def get_anchors(words, ontology):
    """ Match single words in the document over the topology to find `anchors`
    i.e. matches that later on can be used for ngram generation or
    subgraph extraction """
    trie = ontology.get_trie()
    anchors = dict()

    for position, word in enumerate(words):
        for form in [word, stem(word)]:
            if form in trie:
                uri = ontology.get_uri_from_label(form)
                add_token(uri, anchors, position, ontology, form=form)

    return anchors.values()
