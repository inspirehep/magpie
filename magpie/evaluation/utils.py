def remove_unguessable_answers(answers, ontology):
    """
    Remove keywords from the ground truth set that do not exist in the ontology
    and therefore could not have been guessed
    :param answers: a dictionary mapping doc_ids to sets of unicodes
    :param ontology: an Ontology object
    :return: Nothing, it operates on the answers object
    """
    for doc_id, kw_set in answers.items():
        to_remove = set()
        for kw in kw_set:
            if ontology.exact_match(kw):
                to_remove.add(kw)
        answers[doc_id] = kw_set - to_remove
