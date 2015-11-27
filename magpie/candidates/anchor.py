from keyword_token import add_token
from magpie.utils.stemmer import stem


def generate_anchor_candidates(document, ontology):
    ontology_dict = ontology.get_literal_uri_mapping()
    document_words = document.get_meaningful_words()
    return get_anchors(document_words, ontology_dict)


def get_anchors(words, ontology_dict):
    anchors = dict()

    for position, word in enumerate(words):
        if word in ontology_dict:
            add_token(word, anchors, position, ontology_dict)
        elif stem(word) in ontology_dict:
            add_token(stem(word), anchors, position, ontology_dict, form=word)

    return anchors.values()
