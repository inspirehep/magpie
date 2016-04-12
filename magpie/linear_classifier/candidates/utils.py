from magpie.linear_classifier.config import NO_OF_LABELS
from magpie.linear_classifier.candidates.keyword_token import add_token, KeywordToken
from magpie.linear_classifier.labels import get_keywords
from magpie.misc.stemmer import stem


def get_anchors(words, ontology):
    """
    Match single words in the document over the topology to find `anchors`
    i.e. matches that later on can be used for ngram generation or
    subgraph extraction

    :param words: an iterable of all the words you want to get anchors from
    :param ontology: Ontology object

    :return a list of KeywordTokens with anchors
    """
    trie = ontology.get_trie()
    anchors = dict()

    for position, word in enumerate(words):
        for form in [word, stem(word)]:
            if form in trie:
                uri = ontology.get_uri_from_label(form)
                add_token(uri, anchors, position, ontology, form=form)

    return anchors.values()


def remove_nostandalone_candidates(kw_candidates, ontology):
    """
    Remove from the list of keyword candidates the ones that can not be
    a keyword on their own, but only form composite keywords.

    :param kw_candidates: an iterable of KeywordTokens with candidates
    :param ontology: Ontology object

    :return new set of KeywordTokens after filtration
    """
    return {k for k in kw_candidates if ontology.can_exist_alone(k.get_uri())}


def add_gt_answers_to_candidates_set(kw_candidates, gt_answers, ontology):
    """
    During training, sometimes the desirable answers are not generated as
    candidate keywords in the first place. Nevertheless, they should be added
    to the list, extracted features and learned on to improve performance.

    This function adds the ground truth answers to the candidate set if they're
    not there.
    :param kw_candidates: a list of KeywordTokens
    :param gt_answers: a set of unicodes with the canonical labels
    :param ontology: an Ontology object

    :return: None, the function operates on the candidate list passed as an arg
    """
    candidate_set = {kw.get_canonical_form() for kw in kw_candidates}
    for keyword in gt_answers:
        if keyword not in candidate_set:
            parsed_label = ontology.parse_label(keyword)
            uri = ontology.get_uri_from_label(parsed_label)

            # if we can get the URI of the ground truth keyword
            if uri:
                kw_candidates.append(
                    KeywordToken(
                        uri,
                        canonical_label=keyword,
                        parsed_label=parsed_label
                    )
                )


def remove_not_considered_keywords(candidates):
    """
    Filters the candidates in place, dropping the ones that we should not consider
    :param candidates: set of KeywordTokens

    :return: filtered set of KeywordTokens
    """
    if NO_OF_LABELS < 0:
        return candidates
    considered_kw = set(get_keywords())
    return {kt for kt in candidates if kt.get_canonical_form() in considered_kw}
