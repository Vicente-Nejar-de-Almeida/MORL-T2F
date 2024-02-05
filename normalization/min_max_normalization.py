class MinMaxNormalization:
    def __init__(self, score_function, min_val, max_val, maximize=True, name=None):
        self.score_function = score_function
        self.min_val = min_val
        self.max_val = max_val
        self.maximize = maximize  # indicates whether higher values are better (True) or worse (False)
        self.name = name

    def get_normalized_value(self, df_feat_all, y_pred):
        data_point = self.score_function(df_feat_all, y_pred)
        normalized_value = (data_point - self.min_val) / (self.max_val - self.min_val)
        normalized_value = max(normalized_value, self.min_val)
        normalized_value = min(normalized_value, self.max_val)
        if not self.maximize:
            normalized_value = 1 - normalized_value
        return data_point, normalized_value
