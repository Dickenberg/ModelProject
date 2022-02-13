import algorithm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-data-path', '-i', type=str)
parser.add_argument('--n-iterations',  '-n', type=int)
parser.add_argument('--output-pic-path', '-o', type=str)
parser.add_argument('--probs', '-p', type=float, nargs='+')
args = parser.parse_args()
algorithm_test = algorithm.ProbsAlgo(data_path=args.input_data_path, probs=args.probs, n=args.n_iterations)
algorithm_test.plot_and_save_result(args.output_pic_path)


