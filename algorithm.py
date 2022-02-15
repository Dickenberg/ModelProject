from collections import defaultdict
from typing import DefaultDict, List
from random import choices

import matplotlib.pyplot as plt


class ProbsAlgo:
    def __init__(self, data_path: str, probs: List[float], n: int) -> None:
        self.true_labels = self.read_file(data_path)
        self.probs = probs
        self.n = n

        self.classes = list(range(0, len(self.probs)))
        self.preds = self.make_predictions()
        self.metrics = self.get_final_metrics()

    def read_file(self, path: str) -> List[int]:
        try:
            with open(path) as file:
                labels_list = [int(line) for line in file]
        except FileNotFoundError:
            print('File not found')
        return labels_list

    def make_predictions(self) -> List[List[int]]:
        predictions = []
        assert sum(self.probs) == 1, 'Sum of probs is not equal to 1'
        for _ in range(0, self.n):
            prediction = choices(self.classes, self.probs, k=len(self.true_labels))
            predictions.append(prediction)
        assert len(predictions) == self.n
        for pred in predictions:
            assert len(pred) == len(self.true_labels)
        return predictions

    @staticmethod
    def accuracy(true_labels: List[int], predictions: List[int]) -> float:
        assert len(true_labels) == len(predictions)
        assert len(true_labels) > 0
        true_count = sum(i == j for i, j in zip(true_labels, predictions))
        accuracy_score = true_count / len(predictions)
        return accuracy_score

    @staticmethod
    def precision(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        assert len(true_labels) == len(predictions)
        assert len(true_labels) > 0
        true_positive = sum(i == j == class_number for i, j in zip(true_labels, predictions))
        false_positive = sum(i != class_number and j == class_number for i, j in zip(true_labels, predictions))
        precision_score = true_positive / (true_positive + false_positive)
        return precision_score

    @staticmethod
    def recall(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        assert len(true_labels) == len(predictions)
        assert len(true_labels) > 0
        true_positive = sum(i == j == class_number for i, j in zip(true_labels, predictions))
        false_negative = sum(i == class_number and j != class_number for i, j in zip(true_labels, predictions))
        recall_score = true_positive / (true_positive + false_negative)
        return recall_score

    @staticmethod
    def calc_cum_mean(non_cum_score_list: List[float]) -> List[float]:
        cum_score_list = [sum(non_cum_score_list[0:i + 1]) / (i + 1) for i in range(0, len(non_cum_score_list))]
        return cum_score_list

    def get_final_metrics(self) -> DefaultDict[str, List[float]]:
        metrics = defaultdict()
        metrics['accuracy'] = [self.accuracy(self.true_labels, pred) for pred in self.preds]
        for class_num in self.classes:
            metrics['precision' + ' ' + str(class_num)] = [self.precision(self.true_labels, pred, class_num)
                                                           for pred in self.preds]
            metrics['recall' + ' ' + str(class_num)] = [self.recall(self.true_labels, pred, class_num)
                                                        for pred in self.preds]
        for key in metrics.keys():
            metrics[key] = self.calc_cum_mean(metrics[key])
        assert len(metrics) == 7
        for metric in metrics.values():
            assert len(metric) == self.n
        return metrics

    def plot_and_save_result(self, output_path: str) -> None:
        data = self.metrics
        fig, ax = plt.subplots(len(data), 1, figsize=(10, 7), sharex=True)
        fig.subplots_adjust(hspace=1)
        for a, key in zip(ax, sorted(data.keys())):
            y = data[key]
            a.title.set_text(key)
            a.tick_params(axis='y', which='major', labelsize=7)
            a.tick_params(axis='y', which='minor', labelsize=5)
            a.grid(True)
            a.plot(y)
        fig.suptitle('Metrics', fontsize=16)
        plt.show()
        fig.savefig(output_path + '/' + 'image.png', dpi=fig.dpi)
