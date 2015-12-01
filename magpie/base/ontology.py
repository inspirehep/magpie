import cPickle as pickle
import logging
import os
import rdflib
from rdflib.namespace import SKOS
from magpie.config import ONTOLOGY_DIR
from magpie.candidates.trie import OntologyTrie
from magpie.utils.misc import get_all_permutations


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


def parse_label(label):
    if not label:
        return None
    else:
        return ''.join(c for c in label if c not in ',:;').lower()


# @memoize
class Ontology(object):
    """ Holds the ontology. """
    def __init__(self, source):
        self.source = source
        self.graph = self.load_ontology_file(source)
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

    def _build_trie(self):
        """ Build the ontology trie """
        uri_dict = self.get_uri2literal_mapping()
        uri_keys, tree_nodes = [], []

        for uri, parsed_label in uri_dict.iteritems():
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

    @staticmethod
    def load_ontology_file(source):
        graph = rdflib.Graph()

        # TODO Add loading in different formats
        graph.parse(source=source)
        logging.info("File " + source + " successfully loaded!")
        return graph

    def get_all_uris(self):
        """ Get all concept URIs """
        return self.graph.subjects(predicate=SKOS.prefLabel)

    def get_all_parsed_labels(self):
        """ Get all ontology concepts in their preferred form (prefLabel). """
        return {parse_label(x[1].value) for x
                in self.graph.subject_objects(SKOS.prefLabel)}

    def get_canonical_label(self, uri):
        """ Get a canonical label of a given URI """
        for obj in self.graph.objects(subject=uri, predicate=SKOS.prefLabel):
            return obj.value  # Return the first value found

    def get_uri2literal_mapping(self):
        """ Get a dictionary mapping all URIs to their parsed prefLabels. """
        mapping = dict()
        for uri, label in self.graph.subject_objects(SKOS.prefLabel):
            mapping[uri] = parse_label(label.value)

        return mapping

    def get_literal2uri_mapping(self):
        """ Get a dictionary mapping parsed node values to their
        canonical labels and full URIs. """
        mapping = dict()
        for uri, label in self.graph.subject_objects(SKOS.prefLabel):
            mapping[parse_label(label.value)] = (label, uri)

        return mapping

    def get_trie(self):
        return self.trie
