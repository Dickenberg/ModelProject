from typing import Dict, List
from random import choices

import matplotlib.pyplot as plt


class ProbsAlgo:
    def __init__(self, data_path: str, probs: List[float], n: int) -> None:
        self.true_labels = self.read_file(data_path)
        self.probs = probs
        self.n = n

        self.preds = self.make_predictions()
        self.metrics = self.get_final_metrics()

    def read_file(self, path: str) -> List[int]:
        with open(path) as file:
            labels_list = [int(line.rstrip('\n')) for line in file][0:10000]
        return labels_list

    def make_predictions(self) -> List[List[int]]:
        predictions = []
        classes = [0, 1, 2]
        for i in range(0, self.n):
            prediction = choices(classes, self.probs, k=10**4)
            predictions.append(prediction)
        assert len(predictions) == self.n
        for pred in predictions:
            assert len(pred) == len(self.true_labels)
        return predictions

    @staticmethod
    def accuracy(true_labels: List[int], predictions: List[int]) -> float:
        true_count = 0
        for i, j in zip(true_labels, predictions):
            if i == j:
                true_count += 1
            else:
                pass
        accuracy_score = true_count / len(predictions)
        return accuracy_score

    @staticmethod
    def precision(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        true_positive = 0
        false_positive = 0
        for i, j in zip(true_labels, predictions):
            if i == j and j == class_number:
                true_positive += 1
            elif i != j and j == class_number:
                false_positive += 1
            else:
                pass
        precision_score = true_positive / (true_positive + false_positive)
        return precision_score

    @staticmethod
    def recall(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        true_positive = 0
        false_negative = 0
        for i, j in zip(true_labels, predictions):
            if i == j and j == class_number:
                true_positive += 1
            elif i == class_number and j != class_number:
                false_negative += 1
            else:
                pass
        precision_score = true_positive / (true_positive + false_negative)
        return precision_score

    def get_final_metrics(self) -> Dict[str, List[float]]:
        metrics = dict()
        list_accuracy = [self.accuracy(self.true_labels, self.preds[i]) for i in range(0, self.n)]
        list_precision_0 = [self.precision(self.true_labels, self.preds[i], 0) for i in range(0, self.n)]
        list_precision_1 = [self.precision(self.true_labels, self.preds[i], 1) for i in range(0, self.n)]
        list_precision_2 = [self.precision(self.true_labels, self.preds[i], 2) for i in range(0, self.n)]
        list_recall_0 = [self.recall(self.true_labels, self.preds[i], 0) for i in range(0, self.n)]
        list_recall_1 = [self.recall(self.true_labels, self.preds[i], 1) for i in range(0, self.n)]
        list_recall_2 = [self.recall(self.true_labels, self.preds[i], 2) for i in range(0, self.n)]
        cumulative_accuracy = [sum(list_accuracy[0:i + 1]) / (i + 1) for i in range(0, len(list_accuracy))]
        cumulative_precision_0 = [sum(list_precision_0[0:i + 1]) / (i + 1) for i in range(0, len(list_precision_0))]
        cumulative_precision_1 = [sum(list_precision_1[0:i + 1]) / (i + 1) for i in range(0, len(list_precision_1))]
        cumulative_precision_2 = [sum(list_precision_2[0:i + 1]) / (i + 1) for i in range(0, len(list_precision_2))]
        cumulative_recall_0 = [sum(list_recall_0[0:i + 1]) / (i + 1) for i in range(0, len(list_recall_0))]
        cumulative_recall_1 = [sum(list_recall_1[0:i + 1]) / (i + 1) for i in range(0, len(list_recall_1))]
        cumulative_recall_2 = [sum(list_recall_2[0:i + 1]) / (i + 1) for i in range(0, len(list_recall_2))]
        metrics.update({'accuracy': cumulative_accuracy,
                        'precision_0': cumulative_precision_0,
                        'precision_1': cumulative_precision_1,
                        'precision_2': cumulative_precision_2,
                        'recall_0': cumulative_recall_0,
                        'recall_1': cumulative_recall_1,
                        'recall_2': cumulative_recall_2
                        })
        assert len(metrics) == 7
        for metric in metrics.values():
            assert len(metric) == self.n
        return metrics

    def plot_and_save_result(self, output_path: str) -> None:
        data = self.metrics
        fig, ax = plt.subplots(len(data), 1, figsize=(7, 5))
        for a, key in zip(ax, data.keys()):
            y = data[key]
            a.plot(y)
        plt.show()
        plt.savefig(output_path + '/' + 'image.png', dpi=300)