from anchor import generate_anchor_candidates
from fuzzywuzzy import process as fuzz_process
from nltk.util import ngrams
from keyword_token import add_token

CUTOFF = 91


def generate_ngram_candidates(document, ontology):
    all_words = document.get_meaningful_words()
    all_concepts = ontology.get_all_concept_values()
    ontology_dict = ontology.get_literal_uri_mapping()
    tokens = set()

    # 1-grams
    anchors = generate_anchor_candidates(document, ontology)
    # TODO remove the standalone ones
    tokens |= set(anchors)

    # 2-grams
    n = 2
    concepts = ontology.get_nlength_concept_values(range(n + 1))
    ngram_tokens = dict()
    for position, ngram in get_ngrams_around_anchors(n, all_words, anchors):
        # TODO potential filtering of ngrams e.g. for linguistic purposes
        form = " ".join(ngram)
        for hit in fuzzy_match(form, concepts, CUTOFF):
            add_token(hit, ngram_tokens, position, ontology_dict, form=form)

    # for i in tokens:
    #     print i.value
    # for i in set(ngram_tokens.values()):
    #     print i.value
    tokens |= set(ngram_tokens.values())

    # 3-grams
    # n = 3
    # concepts = ontology.get_nlength_concept_values(n)
    # for position, ngram in get_all_ngrams(n, all_words):
    #     ng_string = " ".join(ngram)
    #     best_hit = fuzz_process.extract(ng_string, concepts)
    #     if best_hit:
    #         print ng_string, best_hit

    # 4-grams
    # n = 4
    # concepts = ontology.get_nlength_concept_values(n)
    # for position, ngram in get_all_ngrams(n, all_words):
    #     ng_string = " ".join(ngram)
    #     best_hit = fuzz_process.extract(ng_string, concepts)
    #     if best_hit:
    #         print ng_string, best_hit

    return tokens


def fuzzy_match(token, collection, cutoff):
    hits = fuzz_process.extract(token, collection)
    return [h[0] for h in hits if h[1] > cutoff] if hits else []


def get_all_ngrams(n, words):
    return enumerate(ngrams(words, n))  # not necessarily a list


def get_ngrams_around_anchors(n, words, anchors):
    all_ngs = []
    for anchor in anchors:
        for i in anchor.get_all_occurrences():
            start_index = max(0, i - n + 1)
            piece = words[start_index: min(i + n, len(words))]
            ngs = enumerate(ngrams(piece, n), start=start_index)
            all_ngs.extend(ngs)

    return all_ngs
