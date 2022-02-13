import algorithm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-data-path', '-i', type=str)
parser.add_argument('--n-iterations',  '-n', type=int)
parser.add_argument('--output-pic-path', '-o', type=str)
parser.add_argument('--probs', '-p', type=float, nargs='+')
args = parser.parse_args()
print(args)