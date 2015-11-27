import logging
import rdflib

SKOS_NAMESPACE = "http://www.w3.org/2004/02/skos/core#"


class Ontology(object):
    def __init__(self, source):
        self.source = source
        self.skos_namespace = rdflib.Namespace(SKOS_NAMESPACE)
        self.graph = self.load_ontology_file(source)

    @staticmethod
    def load_ontology_file(source):
        graph = rdflib.Graph()

        # TODO Add loading in different formats
        graph.parse(source=source)
        logging.info("File " + source + " successfully loaded!")
        return graph

    def get_all_concept_values(self):
        return {uri_lab[1].value for uri_lab
                in self.graph.subject_objects(self.skos_namespace["prefLabel"])}

    def get_literal_uri_mapping(self):
        return {uri_lab[1].value.lower(): uri_lab[0] for uri_lab
                in self.graph.subject_objects(self.skos_namespace["prefLabel"])}

    def get_nlength_concept_values(self, lengths):
        # TODO if n == 1 serve the ones without nostandalone?
        # this is problematic because of photon production <-> photoproduction
        all_concepts = self.get_all_concept_values()
        return {c for c in all_concepts if len(c.split(' ')) in lengths}
