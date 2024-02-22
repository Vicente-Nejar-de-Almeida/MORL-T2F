import numpy as np


class ZScoreNormalization:
    def __init__(self, score_function, maximize=True, name=None):
        self.score_function = score_function
        self.observed_sample = []
        self.maximize = maximize  # indicates whether higher values are better (True) or worse (False)
        self.name = name

    def get_normalized_value(self, df_feat_all, y_pred):
        try:
            data_point = self.score_function(df_feat_all, y_pred)
        except:
            return None, -1
        else:
            self.observed_sample.append(data_point)
        

        # return data_point, (data_point - min(self.observed_sample)) / (0.0001 + max(self.observed_sample) - min(self.observed_sample))
        
        std = np.std(self.observed_sample)
        if std == 0:
            # normalized_value = math.inf
            normalized_value = 0
        else:
            normalized_value = (data_point - np.mean(self.observed_sample))/std
        
        return data_point, normalized_value
