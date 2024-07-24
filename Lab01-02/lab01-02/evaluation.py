from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    tp, fp, fn = 0, 0, 0

    for a, e in zip(actual_results, expected_results):
        if a and e: 
            tp += 1
        elif a and not e:
            fp += 1
        elif not a and e:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    precision, recall = precision_recall(expected_results, actual_results)
    if precision == 0 and recall == 0:
        return 0.0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score
