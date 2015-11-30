import cPickle as pickle
import logging
import os
import rdflib
from rdflib.namespace import SKOS
from magpie.config import ONTOLOGY_DIR
from magpie.candidates.trie import OntologyTrie


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
    return ''.join(c for c in label if c not in ',:;').lower()


# @memoize
class Ontology(object):
    """ Holds the ontology. """
    def __init__(self, source):
        self.source = source
        self.graph = self.load_ontology_file(source)
        self.trie = OntologyTrie(self.get_all_concept_values())

    def fuzzy_match(self, word):
        """ Fuzzy match a given word over the ontology. """
        return self.trie.fuzzy_match(word)

    @staticmethod
    def load_ontology_file(source):
        graph = rdflib.Graph()

        # TODO Add loading in different formats
        graph.parse(source=source)
        logging.info("File " + source + " successfully loaded!")
        return graph

    def get_all_concept_values(self):
        """ Get all ontology concepts in their preferred form (prefLabel). """
        return {parse_label(x[1].value) for x
                in self.graph.subject_objects(SKOS.prefLabel)}

    def get_literal_uri_mapping(self):
        """ Get a dictionary mapping parsed node values to their
        canonical labels and full URIs. """
        mapping = dict()
        for uri, label in self.graph.subject_objects(SKOS.prefLabel):
            mapping[parse_label(label.value)] = (label, uri)

        return mapping

    def get_trie(self):
        return self.trie
