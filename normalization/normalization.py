from abc import ABC, abstractmethod


class Normalization(ABC):

    @abstractmethod
    def normalize(self, df_feat_all, y_pred):
        pass
