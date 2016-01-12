import marisa_trie


class MarisaTrie(object):
    """
    Holds a trie containing all the ontology terms and enables to perform
    fuzzy matching over it. Implemented with marisa-trie.
    """
    def __init__(self, literals):
        self.trie = marisa_trie.Trie(literals)
        self.cutoff = 2
        self._second_row = self.get_trie_row(2)
        self._hits = set()
        self.comparisons = 0

    def exact_match(self, word):
        """
        Look if a word is in the trie
        :param word: word to match on
        :return: Set of hits
        """
        if word in self.trie:
            return {word}
        else:
            return set()

    def fuzzy_match(self, word):
        """ Fuzzy match a word over the trie. The allowed fuzziness is
        automatically calculated based on the word length. """
        self.adjust_cutoff(len(word))
        self._hits = set()
        self.comparisons = 0

        # We assume for performance purposes
        # that there's no mistake in the first two characters
        prefix = word[:2]
        if prefix in self._second_row:
            self._recursive_match(prefix, word)

        # for letter in self._first_row:
        #     self._recursive_match(letter, word)

        # print("Computed " + str(self.comparisons) + " comparisons")
        return {hit[0] for hit in self._hits}

    def adjust_cutoff(self, word_length):
        """ Determine the allowed Levenshtein distance for fuzzy matching,
         depending on the pattern length. """
        if word_length < 3:
            self.cutoff = 0
        elif word_length < 6:  # maybe 5?
            self.cutoff = 1
        else:
            self.cutoff = 2

    def get_trie_row(self, row_number, prefix=u'', exclude=None):
        """ Get all nodes of a trie from a certain level.
        A prefix can be added for filtering the tree.

        :param row_number - integer
        :param prefix - unicode, results have to contain this prefix
        :param exclude - set of unicodes, nodes to exclude from the results. """
        exclude = exclude or set()
        return {key[:row_number] for key in self.trie.keys(prefix)} - exclude

    def __getitem__(self, item):
        return self.trie.get(item)

    def __len__(self):
        return len(self.trie)

    def __contains__(self, item):
        return item in self.trie

    def _recursive_match(self, prefix, word, prev_row=None):
        """ Walks the trie computing the Levenshtein distance """
        self.comparisons += 1
        if not prev_row:
            prev_row = range(len(word) + 1)

        distance_row = MarisaTrie.iter_levenshtein(prefix, word, prev_row)

        if min(distance_row) <= self.cutoff:
            if prefix in self.trie and distance_row[-1] <= self.cutoff:
                self._hits.add((prefix, distance_row[-1]))

            children = self.get_trie_row(
                len(prefix) + 1,
                prefix=prefix,
                exclude={prefix}
            )
            for child in children:
                self._recursive_match(child, word, distance_row)

    @staticmethod
    def iter_levenshtein(prefix, word, prev_row):
        """ Compute another iteration of the Levenshtein distance. """
        current_row = [prev_row[0] + 1] * len(prev_row)

        for i in xrange(1, len(word) + 1):
            indicator_fun = 0 if prefix[-1] == word[i - 1] else 1
            current_row[i] = min(
                current_row[i - 1] + 1,
                prev_row[i] + 1,
                prev_row[i - 1] + indicator_fun,
            )

        return current_row


class TrieNode(object):
    """ Class representing a node in the trie. If the node represents a word,
     self.word != None, otherwise it's None. """
    def __init__(self):
        self.phrase = None
        self.children = {}


class Trie(object):
    """ Class manually implementing a trie. Works faster than marisa-trie,
     because it enables walking it for fuzzy search. """

    def __init__(self, init_nodes=None):
        self.root = TrieNode()
        self.cutoff = 2
        self.node_count = 0
        self.phrase_dict = dict()

        init_nodes = init_nodes or []
        for node in init_nodes:
            self.insert(node)

    def insert(self, phrase):
        """ Insert a phrase into the trie """
        node = self.root
        for letter in phrase:
            if letter not in node.children:
                node.children[letter] = TrieNode()

            node = node.children[letter]

        node.phrase = phrase
        self.node_count += 1
        self.phrase_dict[phrase] = self.node_count

    def exact_match(self, phrase):
        """ Look if there is a phrase exactly like the given one in the trie """
        if phrase in self.phrase_dict:
            return {phrase}
        else:
            return set()

    def fuzzy_match(self, phrase):
        """ Walk the trie down while computing the Levenshtein distance
        and find any fuzzy matches for the phrase """
        if phrase in self.phrase_dict:
            return {phrase}

        # For such short phrases we expect exact matches
        if len(phrase) < 3:
            return set()

        self.adjust_cutoff(len(phrase))
        self._hits = set()
        try:
            start_node = self.root.children[phrase[0]].children[phrase[1]]
        except (KeyError, AttributeError):
            return self._hits

        first_row = [2, 1] + range(len(phrase) - 1)

        # recursively search each branch of the trie
        for letter in start_node.children:
            self.search_recursive(
                start_node.children[letter],
                letter,
                phrase,
                first_row,  # first row
            )

        return {hit[0] for hit in self._hits}

    def search_recursive(self, node, current_letter, phrase, previous_row):
        """
        Recursively search the trie and build a matrix for Levenshtein distance
        :param node: TrieNode object representing the node to be examined
        :param current_letter: the letter that we are computing for
        :param phrase: unicode with the phrase we look for
        :param previous_row: integer list with previously computed values

        :return: Nothing, it iterates recursively while filling the self._hits
        variable until it gets to the bottom of the trie or the allowed
        Levenshtein distance is reached
        """

        current_row = [previous_row[0] + 1] * len(previous_row)

        # Build one row for the letter, with a column for each letter in the target
        # phrase, plus one for the empty string at column 0
        for column in xrange(1, len(phrase) + 1):

            insert_cost = current_row[column - 1] + 1
            delete_cost = previous_row[column] + 1

            if phrase[column - 1] != current_letter:
                replace_cost = previous_row[column - 1] + 1
            else:
                replace_cost = previous_row[column - 1]

            current_row[column] = min(insert_cost, delete_cost, replace_cost)

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a phrase in this trie node, then add it.
        if current_row[-1] <= self.cutoff and node.phrase is not None:
            self._hits.add((node.phrase, current_row[-1]))

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min(current_row) <= self.cutoff:
            for letter in node.children:
                self.search_recursive(
                    node.children[letter],
                    letter,
                    phrase,
                    current_row
                )

    def adjust_cutoff(self, phrase_length):
        """ Determine the allowed Levenshtein distance for fuzzy matching,
         depending on the pattern length. """
        if phrase_length < 3:
            self.cutoff = 0
        elif phrase_length < 6:  # maybe 5?
            self.cutoff = 1
        else:
            self.cutoff = 2

    def __getitem__(self, item):
        return self.phrase_dict.get(item)

    def __len__(self):
        return self.node_count

    def __contains__(self, item):
        return item in self.phrase_dict
