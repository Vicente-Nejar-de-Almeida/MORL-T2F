from abc import ABC, abstractmethod


class EarlyStopping(ABC):
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update_metric(self, new_metric_value):
        pass
