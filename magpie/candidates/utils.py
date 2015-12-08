from magpie.candidates.keyword_token import add_token
from magpie.utils.stemmer import stem


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
    nostandalones = ontology.nostandalones
    return {kw for kw in kw_candidates if kw.get_uri() not in nostandalones}
