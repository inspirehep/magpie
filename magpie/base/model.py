from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


class LearningModel(object):
    """
    Represents the model that can be trained and later used to predict
    keywords for unknown data
    """
    def __init__(self, global_frequencies=None):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier()
        self.global_index = global_frequencies

    def fit_and_scale(self, matrix):
        return self.scaler.fit_transform(matrix)

    def scale(self, matrix):
        return self.scaler.transform(matrix)

    def fit_classifier(self, input_matrix, output_vector):
        self.classifier = self.classifier.fit(input_matrix, output_vector)

    def scale_and_predict(self, input_vector):
        scaled_vec = self.scale(input_vector)
        return self.classifier.predict(scaled_vec)

    def get_global_index(self):
        return self.global_index