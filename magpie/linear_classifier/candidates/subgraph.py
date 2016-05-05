import collections
import copy

from magpie.linear_classifier.candidates.keyword_token import KeywordToken
from rdflib.namespace import SKOS

from magpie.linear_classifier.candidates.utils import get_anchors, remove_nostandalone_candidates, \
    remove_not_considered_keywords


def generate_subgraph_candidates(document, ontology):
    all_words = document.get_meaningful_words()
    anchors = get_anchors(all_words, ontology)

    candidates = set(copy.copy(anchors))

    # Broader
    broader_nodes = get_related_concepts(
        anchors,
        SKOS.broader,
        ontology,
        depth=2
    )
    candidates.update(broader_nodes)

    # Narrower
    narrower_nodes = get_related_concepts(
        anchors,
        SKOS.narrower,
        ontology,
        depth=2
    )
    candidates.update(narrower_nodes)

    # Composite
    composite_nodes = get_related_concepts(
        anchors,
        SKOS.composite,
        ontology,
        depth=1
    )
    candidates.update(composite_nodes)

    # CompositeOf
    compositeOf_nodes = get_related_concepts(
        anchors,
        SKOS.compositeOf,
        ontology,
        # depth=1
    )
    candidates.update(compositeOf_nodes)

    # Related
    related_nodes = get_related_concepts(
        anchors,
        SKOS.related,
        ontology,
        depth=2
    )
    candidates.update(related_nodes)

    # Attach labels
    for kw in candidates:
        kw.canonical_label = ontology.get_canonical_label_from_uri(kw.uri)
        kw.parsed_label = ontology.get_parsed_label_from_uri(kw.uri)

    candidates = remove_not_considered_keywords(candidates)

    return remove_nostandalone_candidates(candidates, ontology)


def get_related_concepts(anchors, relation, ontology, depth=None):
    """ Uses BFS to walk the ontology graph through a certain relation edges
    and gets nodes up to a certain depth. Not specifying depth means unlimited
    depth. """
    nodes = set()
    queue = collections.deque([(0, a) for a in anchors])

    while queue:
        distance, node = queue.popleft()
        if not depth or distance < depth:
            for child_uri in ontology.get_children_of_node(node.uri, relation):
                token = KeywordToken(child_uri, hops_from_anchor=distance + 1)
                if token not in nodes:
                    queue.append((distance + 1, token))
                    nodes.add(token)

    return nodes
