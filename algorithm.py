from typing import Dict, List


class ProbsAlgo:
    def __init__(self, data_path: str, probs: List[float], n: int) -> None:
        self.true_labels = self.read_file(data_path)
        self.probs = probs
        self.n = n

        self.preds = self.make_predictions()
        self.metrics = self.get_final_metrics()

    def read_file(self, path: str) -> List[int]:
        pass

    def make_predictions(self) -> List[List[int]]:
        predictions = []
        ...
        assert len(predictions) == self.n
        for pred in predictions:
            assert len(pred) == len(self.true_labels)
        return predictions

    @staticmethod
    def accuracy(true_labels: List[int], predictions: List[int]) -> float:
        pass

    @staticmethod
    def precision(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        pass

    @staticmethod
    def recall(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        pass

    def get_final_metrics(self) -> Dict[str, List[float]]:
        metrics = dict()
        ...
        assert len(metrics) == 7
        for metric in metrics.values():
            assert len(metric) == self.n
        return metrics

    def plot_and_save_result(self, output_path: str) -> None:
        pass
