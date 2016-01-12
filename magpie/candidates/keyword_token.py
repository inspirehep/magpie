import bisect


class KeywordToken(object):
    """
    Represents a keyword candidate with all its occurring forms and metrics
    calculated e.g. no of occurrences
    """
    def __init__(self,
                 uri,
                 position=-1,
                 parsed_label=None,
                 canonical_label=None,
                 hops_from_anchor=0,
                 form=None):
        self.canonical_label = canonical_label
        self.parsed_label = parsed_label
        self.hops_from_anchor = hops_from_anchor
        self.uri = uri
        self.occurrences = [position]
        self._hash = hash(self.uri)
        self.forms = {form or parsed_label}  # form if it's not None, value otherwise

    def add_occurrence(self, position, form=None):
        assert position >= 0

        # Check if exists first
        index = bisect.bisect(self.occurrences, position)
        if index == 0 or self.occurrences[index - 1] != position:
            self.occurrences.insert(index, position)

            if form:
                self.forms.add(form)

    def __hash__(self):
        return self._hash

    def __cmp__(self, other):
        """Compare objects using _hash."""
        if self._hash < other.__hash__():
            return -1
        elif self._hash == other.__hash__():
            return 0
        else:
            return 1

    def __str__(self):
        return self.get_canonical_form()

    def get_parsed_form(self):
        return self.parsed_label or self.uri

    def get_canonical_form(self):
        return self.canonical_label or self.get_parsed_form()

    def get_uri(self):
        return self.uri

    def get_first_occurrence(self):
        return self.occurrences[0]

    def get_last_occurrence(self):
        return self.occurrences[-1]

    def get_all_occurrences(self):
        return self.occurrences


def add_token(uri, collection, position, ontology, form=None):
    """ Adds an additional occurrence to a collections of tokens or creates
     a new token if it doesn't yet exist.
      :param uri - SKOS URI
      :param collection - collection of KeywordTokens
      :param position - integer, token position in the document
      :param ontology - Ontology object
      :param form - string representing the form that a token can take

      :return None"""
    if uri in collection:
        collection[uri].add_occurrence(position, form=form)
    else:
        canonical_label = ontology.get_canonical_label_from_uri(uri)
        parsed_label = ontology.get_parsed_label_from_uri(uri)

        collection[uri] = KeywordToken(
            uri,
            position=position,
            canonical_label=canonical_label,
            parsed_label=parsed_label,
            form=form
        )
