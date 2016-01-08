def remove_unguessable_answers(answers, ontology):
    """
    Remove keywords from the ground truth set that do not exist in the ontology
    and therefore could not have been guessed
    :param answers: a set of unicodes
    :param ontology: an Ontology object
    :return: a filtered set
    """
    return {kw for kw in answers if ontology.exact_match(kw)}
