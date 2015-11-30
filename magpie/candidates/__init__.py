from anchor import generate_anchor_candidates
from ngram import generate_ngram_candidates

STRATEGY = 'NGRAMS'


def generate_keyword_candidates(document, ontology):
    """
    :param document: Document object containing the text that we generate
    generate keywords from
    :param ontology: Ontology object on which we match the keywords
    :return:
    """
    return {
        'NGRAMS': generate_ngram_candidates(document, ontology),
        'ANCHOR': generate_anchor_candidates(document, ontology),
    }[STRATEGY]
