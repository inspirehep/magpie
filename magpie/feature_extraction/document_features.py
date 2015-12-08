import pandas as pd


def extract_document_features(inv_index, number_of_rows):
    # Word counts
    total_words = inv_index.get_total_number_of_words()
    unique_words = inv_index.get_number_of_unique_words()

    return pd.DataFrame({
        'total_words_in_doc': [total_words] * number_of_rows,
        'unique_words_in_doc': [unique_words] * number_of_rows,
    })
