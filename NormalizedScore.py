import math


class NormalizedScore:
    def __init__(self, score_function):
        self.score_function = score_function
        self.count = 0
        self.sum = 0
        self.sum_of_squares = 0

    def get_normalized_value(self, df_feat_all, y_pred):
        data_point = self.score_function(df_feat_all, y_pred)
        STD = self.compute_standard_deviation()
        if STD == 0:
            normalizedValue = math.inf
        else:
            normalizedValue = (data_point - self.compute_average())/STD
        self.count += 1
        self.sum += data_point
        self.sum_of_squares += data_point ** 2
        return data_point, normalizedValue

    def compute_average(self):
        if self.count == 0:
            return 0  # To avoid division by zero
        return self.sum / self.count

    def compute_standard_deviation(self):
        if self.count == 0:
            return 0  # To avoid division by zero
        variance = (self.sum_of_squares - (self.sum ** 2) / self.count) / self.count
        return math.sqrt(variance)
