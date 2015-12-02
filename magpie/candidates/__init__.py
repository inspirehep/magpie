from ngram import generate_ngram_candidates
from subgraph import generate_subgraph_candidates

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
        'SUBGRAPH': generate_subgraph_candidates(document, ontology),
    }[STRATEGY]
