import cPickle as pickle
import logging
import networkx as nx
import os
import rdflib
from rdflib.namespace import SKOS

from magpie.config import ONTOLOGY_DIR
from magpie.candidates.trie import Trie, MarisaTrie
from magpie.misc.utils import get_all_permutations

PUNCTUATION_TABLE = dict((ord(c), None) for c in u',:;')


_ontology = dict()


# TODO change into a decorator
def OntologyFactory(source, recreate=False):
    """ A wrapper for Ontology class. Speeds up its creation. """
    global _ontology

    # Cache L1 (memory)
    if source in _ontology and not recreate:
        return _ontology[source]

    # Cache L2 (pickle)
    filename = os.path.basename(source).split('.')[0]
    pickle_path = os.path.join(ONTOLOGY_DIR, filename + '.pickle')
    if os.path.exists(pickle_path) and not recreate:
        loaded = pickle.load(open(pickle_path, 'rb'))
        _ontology[source] = loaded
        return loaded

    # Sorry, have to recreate
    ontology = Ontology(source)
    logging.info("Ontology " + source + " successfully loaded!")
    _ontology[source] = ontology  # dump to L1
    pickle.dump(ontology, open(pickle_path, 'wb'))  # dump to L2
    return ontology


class Ontology(object):
    """ Holds the ontology. """
    def __init__(self, source):
        self.source = source
        self.graph = self._build_graph(source)

        self.id_mapping = None  # defined in _build_tree()
        self.trie = self._build_trie()

    def __contains__(self, item):
        return self.exact_match(item)

    def exact_match(self, word, already_parsed=False):
        """ Look for a word (its canonical label) in the ontology """
        if not already_parsed:
            word = self.parse_label(word)
        return self.trie.exact_match(word)

    def fuzzy_match(self, word):
        """ Fuzzy match a given word over the ontology. """
        return self.trie.fuzzy_match(word)

    def get_uri_from_label(self, label):
        """ Get the full URI of a token if it's in the trie """
        node_id = self.trie[label]

        if not node_id:  # if not in the trie
            return None

        return self.id_mapping[node_id]

    def get_children_of_node(self, node_uri, relation):
        """ Get node_uri adjacent nodes separated by a given relation. """
        try:
            return [v for v, e in self.graph[node_uri].iteritems()
                    if e['relation'] == relation]
        except KeyError:
            return []

    def get_canonical_label_from_uri(self, uri):
        """ Get a canonical label of a given URI. """
        return self.graph.node.get(uri, {}).get('canonical')

    def get_parsed_label_from_uri(self, uri):
        """ Get a parsed label of a given URI. """
        return self.graph.node.get(uri, {}).get('parsed')

    def get_trie(self):
        """ Get the trie. """
        return self.trie

    def get_number_of_nodes(self):
        """ Get the total number of nodes in the ontology. """
        return self.graph.number_of_nodes()

    def get_number_of_edges(self):
        """ Get the total number of edges in the ontology. """
        return self.graph.number_of_edges()

    @staticmethod
    def load_ontology_file(source):
        """ Load the RDF file into memory. """
        graph = rdflib.Graph()

        # TODO Add loading in different formats
        graph.parse(source=source)
        logging.info("File " + source + " successfully loaded!")
        return graph

    @staticmethod
    def parse_label(label):
        """ Remove punctuation and uppercase letters from a label. """
        if not label:
            return None
        else:
            return label.translate(PUNCTUATION_TABLE).lower()

    @staticmethod
    def parse_uri(uri):
        """ Parse the URI and return the concept name without the prefix """
        return u'#'.join(unicode(uri).split('#')[1:])

    def _build_graph(self, source):
        """ Build a NetworkX graph. """
        rdf_graph = self.load_ontology_file(source)

        g = nx.DiGraph()
        relations = {
            SKOS.composite,
            SKOS.broader,
            SKOS.narrower,
            SKOS.compositeOf,
            SKOS.related
        }

        labels = {
            SKOS.prefLabel,
            SKOS.altLabel,
        }

        flags = {
            # TODO might want to generalize it in the future
            rdflib.term.Literal(u'nostandalone', lang=u'en'),
        }

        for s, p, o in rdf_graph:
            if p in relations:
                g.add_edge(s, o, relation=p)
            elif p in labels:
                if p == SKOS.prefLabel:
                    g.add_node(s, {'canonical': o.value})
                elif p == SKOS.altLabel:
                    if g.node.get(s, {}).get('alternative') is None:
                        g.add_node(s, {'alternative': []})
                    g.node[s]['alternative'].append(o.value)
            elif p == SKOS.note and o in flags:
                g.add_node(s, {o.value: True})

        # Add canonical and parsed labels
        for uri in g:
            node_attr = g.node[uri]
            if 'canonical' not in node_attr:
                node_attr['canonical'] = self.parse_uri(uri)

            node_attr['parsed'] = self.parse_label(node_attr['canonical'])

        return g

    def _build_trie(self):
        """ Build the ontology trie. """
        uri_keys, tree_nodes = [], []

        for uri in self.graph:
            parsed_label = self.graph.node[uri]['parsed']
            added_elements = 0

            # Permutations
            permutations = get_all_permutations(parsed_label)
            tree_nodes.extend(permutations)
            added_elements += len(permutations)

            # altLabels
            for alt in self.graph.node[uri].get('alternative', []):
                tree_nodes.append(alt)
                added_elements += 1

            uri_keys.extend([uri] * added_elements)

        # Build the tree
        # trie = Trie(tree_nodes)
        trie = MarisaTrie(tree_nodes)
        node_mapping = dict()

        for i in xrange(len(tree_nodes)):
            node_id = trie[tree_nodes[i]]
            node_mapping[node_id] = uri_keys[i]

        self.id_mapping = node_mapping

        return trie

    def can_exist_alone(self, uri):
        """ Check if an URI can be a keyword by itself. """
        if not uri:
            return False

        if uri not in self.graph:
            raise ValueError("URI {} does not exist in the ontology.".format(uri))

        return not self.graph.node[uri].get('nostandalone', False)
