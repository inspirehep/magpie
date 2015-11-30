import bisect


class KeywordToken(object):
    """
    Represents a keyword candidate with all its occurring forms and metrics
    calculated e.g. no of occurrences
    """
    def __init__(self,
                 parsed_form,
                 position=-1,
                 uri=None,
                 canonical_form=None,
                 form=None):
        self.parsed_form = parsed_form
        self.canonical_form = canonical_form
        self.uri = uri
        self.occurrences = [position]
        self._hash = hash(self.parsed_form)
        self.forms = {form or parsed_form}  # form if it's not None, value otherwise

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
        return self.canonical_form

    def get_parsed_form(self):
        return self.parsed_form

    def get_canonical_form(self):
        return self.canonical_form.value

    def get_uri(self):
        return self.uri

    def get_first_occurrence(self):
        return self.occurrences[0]

    def get_last_occurrence(self):
        return self.occurrences[-1]

    def get_all_occurrences(self):
        return self.occurrences


def add_token(token, collection, position, ontology_dict, form=None):
    """ Adds an additional occurrence to a collections of tokens or creates
     a new token if it doesn't yet exist.
      :param token - string
      :param collection - collection of KeywordTokens
      :param position - integer, token position in the document
      :param ontology_dict - dict with Literal->URI ontology mapping
      :param form - string representing the form that a token can take

      :return None"""
    if token in collection:
        collection[token].add_occurrence(position, form=form)
    else:
        canonical_label, uri = ontology_dict.get(token)
        collection[token] = KeywordToken(token,
                                         position=position,
                                         canonical_form=canonical_label,
                                         uri=uri,
                                         form=form)
