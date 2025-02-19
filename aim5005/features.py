import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return (x-self.minimum)/(self.maximum-self.minimum)
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)  

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Fit must be called before transform.")
        
        std_corrected = np.where(self.std_ == 0, 1, self.std_)  
        return (X - self.mean_) / std_corrected

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    


class LabelEncoder:
    def __init__(self):
        self.classes_ = []
        self.mapping_ = {}

    def fit(self, y):
        unique_labels = []
        for label in y:
            if label not in unique_labels:
                unique_labels.append(label)
        self.classes_ = np.array(unique_labels)
        self.mapping_ = {label: index for index, label in enumerate(self.classes_)}

    def transform(self, y):
        if not self.mapping_:
            raise ValueError("LabelEncoder not fitted, call 'fit' first.")
        return np.array([self.mapping_[label] for label in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        if not self.classes_:
            raise ValueError("LabelEncoder not fitted, call 'fit' first.")
        return np.array([self.classes_[index] for index in y])

