from typing import List, Tuple
from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    BOOLEAN = 0
    CLASSES = 1
    REAL = 2

class PointSet:
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.split_value_classes = None
        self.split_value_real = None

    def get_gini(self) -> float:
        if len(self.labels) == 0:
            return 0
        counter_0 = sum(1 for label in self.labels if label == 0)
        counter_1 = len(self.labels) - counter_0
        counter_0 /= len(self.labels)
        counter_1 /= len(self.labels)
        return 1 - counter_0**2 - counter_1**2

    def get_best_gain(self, min_split_points: int) -> Tuple[int, float]:
        parent_gini = self.get_gini()
        best_gini = -1
        best_index = -1
        self.split_value_classes = None
        self.split_value_real = None

        for j in range(len(self.features[0])):
            if self.types[j] == FeaturesTypes.BOOLEAN:
                column_ones = [row_idx for row_idx, row in enumerate(self.features) if row[j] == 1]
                column_zeros = [row_idx for row_idx, row in enumerate(self.features) if row[j] != 1]

                counter_1 = len(column_ones)
                counter_0 = len(column_zeros)

                if counter_1 >= min_split_points and counter_0 >= min_split_points:
                    counter_11 = sum(self.labels[i] for i in column_ones)
                    gini_1 = 1 - (counter_11 / counter_1)**2 - ((counter_1 - counter_11) / counter_1)**2

                    counter_00 = sum(self.labels[i] == 0 for i in column_zeros)
                    gini_0 = 1 - (counter_00 / counter_0)**2 - ((counter_0 - counter_00) / counter_0)**2

                    weighted_gini = parent_gini - (counter_1 / len(self.labels)) * gini_1 - (counter_0 / len(self.labels)) * gini_0

                    if weighted_gini > best_gini:
                        best_gini = weighted_gini
                        best_index = j

            elif self.types[j] == FeaturesTypes.CLASSES:
                categories_list = list(set(row[j] for row in self.features))

                for category in categories_list:
                    column_ones = [row_idx for row_idx, row in enumerate(self.features) if row[j] == category]
                    column_zeros = [row_idx for row_idx, row in enumerate(self.features) if row[j] != category]

                    counter_1 = len(column_ones)
                    counter_0 = len(column_zeros)

                    if counter_1 >= min_split_points and counter_0 >= min_split_points:
                        counter_11 = sum(self.labels[i] for i in column_ones)
                        gini_1 = 1 - (counter_11 / counter_1)**2 - ((counter_1 - counter_11) / counter_1)**2

                        counter_00 = sum(self.labels[i] == 0 for i in column_zeros)
                        gini_0 = 1 - (counter_00 / counter_0)**2 - ((counter_0 - counter_00) / counter_0)**2

                        weighted_gini = parent_gini - (counter_1 / len(self.labels)) * gini_1 - (counter_0 / len(self.labels)) * gini_0

                        if weighted_gini > best_gini:
                            best_gini = weighted_gini
                            best_index = j
                            self.split_value_classes = category

            elif self.types[j] == FeaturesTypes.REAL:
                thresholds = list(set(row[j] for row in self.features))

                for threshold in thresholds:
                    column_ones = [row_idx for row_idx, row in enumerate(self.features) if row[j] > threshold]
                    column_zeros = [row_idx for row_idx, row in enumerate(self.features) if row[j] <= threshold]

                    counter_1 = len(column_ones)
                    counter_0 = len(column_zeros)

                    if counter_1 >= min_split_points and counter_0 >= min_split_points:
                        counter_11 = sum(self.labels[i] for i in column_ones)
                        gini_1 = 1 - (counter_11 / counter_1)**2 - ((counter_1 - counter_11) / counter_1)**2

                        counter_00 = sum(self.labels[i] == 0 for i in column_zeros)
                        gini_0 = 1 - (counter_00 / counter_0)**2 - ((counter_0 - counter_00) / counter_0)**2

                        weighted_gini = parent_gini - (counter_1 / len(self.labels)) * gini_1 - (counter_0 / len(self.labels)) * gini_0

                        if weighted_gini > best_gini:
                            best_gini = weighted_gini
                            best_index = j
                            right = self.features[column_zeros, j]
                            left = self.features[column_ones, j]
                            split_value = (np.max(right) + np.min(left)) / 2
                            self.split_value_real = split_value

        if best_gini == -1:
            return None, None

        return best_index, best_gini