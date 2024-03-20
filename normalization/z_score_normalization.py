import math
from normalization.normalization import Normalization


class ZScoreNormalization(Normalization):
    def __init__(self, score_function, maximize=True, name=None):
        self.score_function = score_function
        self.count = 0
        self.sum = 0
        self.sum_of_squares = 0
        self.maximize = maximize  # indicates whether higher values are better (True) or worse (False)
        self.name = name

    def normalize(self, df_feat_all, y_pred):
        data_point = self.score_function(df_feat_all, y_pred)
        std = self.compute_standard_deviation()
        if std == 0:
            # normalized_value = math.inf
            normalized_value = 0
        else:
            normalized_value = (data_point - self.compute_average())/std
        self.count += 1
        self.sum += data_point
        self.sum_of_squares += data_point ** 2
        return data_point, normalized_value

    def compute_average(self):
        if self.count == 0:
            return 0  # To avoid division by zero
        return self.sum / self.count

    def compute_standard_deviation(self):
        if self.count == 0:
            return 0  # To avoid division by zero
        variance = (self.sum_of_squares - (self.sum ** 2) / self.count) / self.count
        return math.sqrt(variance)
