from magpie.linear_classifier.config import NO_OF_LABELS
from magpie.linear_classifier.candidates.keyword_token import KeywordToken
from magpie.linear_classifier.labels import get_keywords
from ngram import generate_ngram_candidates
from subgraph import generate_subgraph_candidates

STRATEGY = 'SUBGRAPH'


def generate_keyword_candidates(document, ontology):
    """
    :param document: Document object containing the text that we generate
    generate keywords from
    :param ontology: Ontology object on which we match the keywords
    :return:
    """
    if NO_OF_LABELS <= 500:
        return {KeywordToken(i, canonical_label=kw, parsed_label=ontology.parse_label(kw))
                for i, kw in enumerate(get_keywords())}

    if STRATEGY == 'NGRAMS':
        return generate_ngram_candidates(document, ontology)
    elif STRATEGY == 'SUBGRAPH':
        return generate_subgraph_candidates(document, ontology)
    elif STRATEGY == 'ENSEMBLE':
        return generate_subgraph_candidates(document, ontology)\
            | generate_ngram_candidates(document, ontology)
    else:
        raise ValueError("Unknown STRATEGY = " + STRATEGY)
