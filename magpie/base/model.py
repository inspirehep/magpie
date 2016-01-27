from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils import compute_class_weight


class LearningModel(object):
    """
    Represents the model that can be trained and later used to predict
    keywords for unknown data
    """
    def __init__(self, global_index, word2vec_model):
        self.scaler = StandardScaler(copy=False)
        self.classifier = SGDClassifier(n_jobs=-1)  # try loss log
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
        if not hasattr(self.scaler, 'n_samples_seen'):
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

    def scale_and_predict(self, input_vector):
        """
        Predict output for a given sample
        :param input_vector: row of a feature matrix

        :return: output vector
        """
        scaled_vec = self.scaler.transform(input_vector)
        return self.classifier.predict(scaled_vec)

    def get_global_index(self):
        """ Get the GlobalFrequencyIndex field. """
        return self.global_index
