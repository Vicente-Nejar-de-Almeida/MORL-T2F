import math
class OnlineDeviationMean:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.sum_of_squares = 0

    def add_data_point(self, data_point):
        self.count += 1
        self.sum += data_point
        self.sum_of_squares += data_point ** 2

    def compute_average(self):
        if self.count == 0:
            return 0  # To avoid division by zero
        return self.sum / self.count

    def compute_standard_deviation(self):
        if self.count == 0:
            return 0  # To avoid division by zero
        variance = (self.sum_of_squares - (self.sum ** 2) / self.count) / self.count
        return math.sqrt(variance)

    def compute_normalize(self, data_point):
        STD = self.compute_standard_deviation()
        if STD == 0:
            return math.inf
        else:
            return (data_point - self.compute_average())/STD
