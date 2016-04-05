import numpy as np
from sklearn.preprocessing import StandardScaler

from magpie.linear_classifier.base.rank_model import RankSVM


class LearningModel(object):
    """
    Represents the model that can be trained and later used to predict
    keywords for unknown data
    """
    def __init__(self, global_index, word2vec_model):
        self.scaler = StandardScaler()
        self.classifier = RankSVM(n_jobs=-1)
        self.global_index = global_index
        self.word2vec = word2vec_model

    def maybe_fit_and_scale(self, matrix):
        """
        If the scaler is not initialized, the fit() is performed on given data.
        Exception is thrown if the data is not big enough. Input matrix is
        scaled and returned.
        :param matrix: matrix to be transformed

        :return: scaled matrix
        """
        if not hasattr(self.scaler, 'n_samples_seen_'):
            if len(matrix) < 1000:
                raise ValueError("Please user bigger batch size. "
                                 "The feature matrix is too small "
                                 "to fit the scaler.")
            else:
                self.scaler.fit(matrix)
        return self.scaler.transform(matrix)

    def partial_fit_classifier(self, input_matrix, output_vector):
        """
        Fit the classifier on X, y matrices. Can be used for online training.
        :param input_matrix: feature matrix
        :param output_vector: vector of the same length as input_matrix

        :return: None
        """
        classes = np.array([0, 1], dtype=np.bool_)
        # TODO Maybe initialize the classifier with this for balancing classes
        # weights = compute_class_weight('balanced', classes, output_vector)

        self.classifier = self.classifier.partial_fit(
            input_matrix,
            output_vector,
            classes=classes,
        )

    def fit_classifier(self, input_matrix, output_vector):
        """
        Fit the classifier on X, y matrices. Previous fit is discarded.
        :param input_matrix: feature matrix
        :param output_vector: vector of the same length as input_matrix

        :return: None
        """
        self.classifier = self.classifier.fit(input_matrix, output_vector)

    def scale_and_predict(self, input_matrix):
        """
        Predict output for given samples
        :param input_matrix: a feature matrix

        :return: matrix with predictions for each sample
        """
        scaled_matrix = self.scaler.transform(input_matrix)
        return self.classifier.predict(scaled_matrix)

    def scale_and_predict_confidence(self, input_matrix):
        """
        Predict confidence values for given samples
        :param input_matrix: a feature matrix

        :return: matrix with confidence values for each sample
        """
        scaled_matrix = self.scaler.transform(input_matrix)
        return self.classifier.decision_function(scaled_matrix)

    def get_global_index(self):
        """ Get the GlobalFrequencyIndex field. """
        return self.global_index
