import marisa_trie

# ALPHABET = {u' ', u'(', u')', u'*', u'+', u'-', u'/', u'0', u'1', u'2', u'3',
#             u'4', u'5', u'6', u'8', u'9', u':', u'a', u'b', u'c', u'd', u'e',
#             u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p',
#             u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'|',
#             u'.', u',', u'\'', u'[', u']'}


class OntologyTrie(object):
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
        return self.trie[item]

    def __len__(self):
        return len(self.trie)

    def __contains__(self, item):
        return item in self.trie

    def _recursive_match(self, prefix, word, prev_row=None):
        """ Walks the trie computing the Levenshtein distance """
        self.comparisons += 1
        if not prev_row:
            prev_row = range(len(word) + 1)

        distance_row = OntologyTrie.iter_levenshtein(prefix, word, prev_row)

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


class Trie(object):
    """ Class manually implementing a trie. Works faster than marisa-trie,
     because it enables walking it for fuzzy search. """
    class TrieNode:
        def __init__(self):
            self.word = None
            self.children = {}

    def __init__(self, init_nodes=None):
        self.root = Trie.TrieNode()
        self.cutoff = 2
        self.node_count = 0
        self.word_dict = dict()

        init_nodes = init_nodes or []
        for node in init_nodes:
            self.insert(node)

    def insert(self, word):
        node = self.root
        for letter in word:
            if letter not in node.children:
                node.children[letter] = Trie.TrieNode()

            node = node.children[letter]

        node.word = word
        self.node_count += 1
        self.word_dict[word] = self.node_count

    def fuzzy_match(self, word):
        if word in self.word_dict:
            return {word}

        # For such short words we expect exact matches
        if len(word) < 3:
            return set()

        self.adjust_cutoff(len(word))
        self._hits = set()
        try:
            start_node = self.root.children[word[0]].children[word[1]]
        except (KeyError, AttributeError):
            return self._hits

        first_row = [2, 1] + range(len(word) - 1)

        # recursively search each branch of the trie
        for letter in start_node.children:
            self.search_recursive(
                start_node.children[letter],
                letter,
                word,
                first_row,  # first row
            )

        return {hit[0] for hit in self._hits}

    def search_recursive(self, node, letter, word, previous_row):
        # print letter, previous_row

        current_row = [previous_row[0] + 1] * len(previous_row)

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in xrange(1, len(word) + 1):

            insert_cost = current_row[column - 1] + 1
            delete_cost = previous_row[column] + 1

            if word[column - 1] != letter:
                replace_cost = previous_row[column - 1] + 1
            else:
                replace_cost = previous_row[column - 1]

            current_row[column] = min(insert_cost, delete_cost, replace_cost)

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if current_row[-1] <= self.cutoff and node.word is not None:
            self._hits.add((node.word, current_row[-1]))

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min(current_row) <= self.cutoff:
            for letter in node.children:
                self.search_recursive(
                    node.children[letter],
                    letter,
                    word,
                    current_row
                )

    def adjust_cutoff(self, word_length):
        """ Determine the allowed Levenshtein distance for fuzzy matching,
         depending on the pattern length. """
        if word_length < 3:
            self.cutoff = 0
        elif word_length < 6:  # maybe 5?
            self.cutoff = 1
        else:
            self.cutoff = 2

    def __getitem__(self, item):
        return self.word_dict.get(item, None)

    def __len__(self):
        return self.node_count

    def __contains__(self, item):
        return item in self.word_dict
