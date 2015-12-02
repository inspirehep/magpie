import cPickle as pickle
import logging
import os
import rdflib
from rdflib.namespace import SKOS
from magpie.config import ONTOLOGY_DIR
from magpie.candidates.trie import OntologyTrie
from magpie.utils.utils import get_all_permutations


# TODO change into a decorator
def OntologyFactory(source):
    """ A wrapper for Ontology class. Speeds up creation. """
    filename = os.path.basename(source).split('.')[0]
    pickle_path = os.path.join(ONTOLOGY_DIR, filename + '.pickle')
    if os.path.exists(pickle_path):
        return pickle.load(open(pickle_path, 'rb'))
    else:
        ontology = Ontology(source)

        pickle.dump(ontology, open(pickle_path, 'wb'))
        return ontology


def memoize(f):
    """
    A basic memoizer
    """
    def memoized(*args, **kwargs):
        source = args[0]
        filename = os.path.basename(source).split('.')[0]
        pickle_path = os.path.join(ONTOLOGY_DIR, filename + '.pickle')
        if os.path.exists(pickle_path):
            # return pickle.load(open(pickle_path, 'rb'))
            print "exists"
            return f(*args, **kwargs)
        else:
            ontology = f(*args, **kwargs)
            # pickle.dump(ontology, open(pickle_path, 'wb'))
            return ontology
    return memoized


# @memoize
class Ontology(object):
    """ Holds the ontology. """
    def __init__(self, source):
        self.source = source
        self.graph = self.load_ontology_file(source)
        self.uri2canonical = self._build_uri2canonical()
        self.parsed2uri = self._build_parsed2uri()
        self.id_mapping = None  # defined in _build_tree()
        self.trie = self._build_trie()

    def fuzzy_match(self, word):
        """ Fuzzy match a given word over the ontology. """
        return self.trie.fuzzy_match(word)

    def get_uri_from_label(self, label):
        """ Get the full URI of a token if it's in the trie """
        if label not in self.trie:
            return None

        node_id = self.trie[label]
        return self.id_mapping[node_id]

    def get_children_of_node(self, node_uri, relation):
        """ Get the children of a specific URI for a given relation type. """
        return self.graph.objects(subject=node_uri, predicate=relation)

    def get_all_uris(self):
        """ Get all concept URIs """
        return self.graph.subjects(predicate=SKOS.prefLabel)

    def get_all_parsed_labels(self):
        """ Get all ontology concepts in their preferred form (prefLabel). """
        return {self.parse_label(x[1].value) for x
                in self.graph.subject_objects(SKOS.prefLabel)}

    def get_canonical_label_from_uri(self, uri):
        """ Get a canonical label of a given URI. """
        try:
            return self.uri2canonical[uri]
        except KeyError:
            return '#'.join(str(uri).split('#')[1:])

    def get_parsed_label_from_uri(self, uri):
        """ Get a parsed label of a given URI. """
        return self.parse_label(self.get_canonical_label_from_uri(uri))

    def get_trie(self):
        return self.trie

    @staticmethod
    def load_ontology_file(source):
        graph = rdflib.Graph()

        # TODO Add loading in different formats
        graph.parse(source=source)
        logging.info("File " + source + " successfully loaded!")
        return graph

    @staticmethod
    def parse_label(label):
        if not label:
            return None
        else:
            return ''.join(c for c in label if c not in ',:;').lower()

    def _build_trie(self):
        """ Build the ontology trie """
        uri_keys, tree_nodes = [], []

        for parsed_label, uri in self.parsed2uri.iteritems():
            added_elements = 0

            # Permutations
            permutations = get_all_permutations(parsed_label)
            tree_nodes.extend(permutations)
            added_elements += len(permutations)

            # altLabels
            for obj in self.graph.objects(subject=uri, predicate=SKOS.altLabel):
                tree_nodes.append(obj.value)
                added_elements += 1

            uri_keys.extend([uri] * added_elements)

        # Build the tree
        trie = OntologyTrie(tree_nodes)
        node_mapping = dict()

        for i in xrange(len(tree_nodes)):
            node_id = trie[tree_nodes[i]]
            node_mapping[node_id] = uri_keys[i]

        self.id_mapping = node_mapping

        return trie

    def _build_uri2canonical(self):
        """ Build a dictionary mapping all URIs to their prefLabels. """
        mapping = dict()
        for uri, pref_label in self.graph.subject_objects(SKOS.prefLabel):
            mapping[uri] = pref_label.value
        return mapping

    def _build_parsed2uri(self):
        """ Build a dictionary mapping all parsed labels to their URIs. """
        mapping = dict()
        for uri, pref_label in self.graph.subject_objects(SKOS.prefLabel):
            mapping[self.parse_label(pref_label.value)] = uri
        return mapping
