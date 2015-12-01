from keyword_token import add_token
from magpie.utils.stemmer import stem


def generate_anchor_candidates(document, ontology):
    document_words = document.get_meaningful_words()
    return get_anchors(document_words, ontology)


def get_anchors(words, ontology):
    trie = ontology.get_trie()
    anchors = dict()

    for position, word in enumerate(words):
        if word in trie:
            add_token(word, anchors, position, ontology)
        elif stem(word) in trie:
            add_token(stem(word), anchors, position, ontology, form=word)

    return anchors.values()
