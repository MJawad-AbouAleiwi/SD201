from typing import List
import numpy as np
from PointSet import PointSet, FeaturesTypes

class Tree:
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes], h: int = 1, min_split_points: int = 1):
        self.points = PointSet(features, labels, types)
        self.best_feature_index, self.best_gain = self.points.get_best_gain(min_split_points)
        n_labels = len(np.unique(labels))

        if h == 0 or n_labels == 1 or self.best_feature_index is None:
            self.is_leaf = True
            counter_1 = sum(labels)
            counter_0 = len(labels) - counter_1
            self.leaf_value = counter_1 > counter_0

        else:
            self.is_leaf = False
            right_child = []
            right_labels = []
            left_child = []
            left_labels = []

            for i, feature in enumerate(features):
                if self.points.types[self.best_feature_index] == FeaturesTypes.BOOLEAN:
                    if feature[self.best_feature_index] == 1.0:
                        right_child.append(feature)
                        right_labels.append(labels[i])
                    else:
                        left_child.append(feature)
                        left_labels.append(labels[i])
                elif self.points.types[self.best_feature_index] == FeaturesTypes.CLASSES:
                    if feature[self.best_feature_index] == self.points.split_value_classes:
                        right_child.append(feature)
                        right_labels.append(labels[i])
                    else:
                        left_child.append(feature)
                        left_labels.append(labels[i])
                elif self.points.types[self.best_feature_index] == FeaturesTypes.REAL:
                    if feature[self.best_feature_index] <= self.points.split_value_real:
                        right_child.append(feature)
                        right_labels.append(labels[i])
                    else:
                        left_child.append(feature)
                        left_labels.append(labels[i])

            if len(right_child) < min_split_points or len(left_child) < min_split_points:
                self.is_leaf = True
                self.leaf_value = sum(labels) > len(labels) // 2
            else:
                self.right_node = Tree(right_child, right_labels, types, h-1, min_split_points)
                self.left_node = Tree(left_child, left_labels, types, h-1, min_split_points)

    def decide(self, features: List[float]) -> bool:
        if self.is_leaf:
            return self.leaf_value

        if self.points.types[self.best_feature_index] == FeaturesTypes.BOOLEAN:
            if features[self.best_feature_index] == 1:
                return self.right_node.decide(features)
            else:
                return self.left_node.decide(features)
            
        if self.points.types[self.best_feature_index] == FeaturesTypes.CLASSES:
            if features[self.best_feature_index] == self.points.split_value_classes:
                return self.right_node.decide(features)
            else:
                return self.left_node.decide(features)
            
        if self.points.types[self.best_feature_index] == FeaturesTypes.REAL:
            if features[self.best_feature_index] <= self.points.split_value_real:
                return self.right_node.decide(features)
            else:
                return self.left_node.decide(features)