import configs
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=222)
args = parser.parse_args()

order = configs.xm3600_langs
random.seed(args.seed)
random.shuffle(order)
order = [order.pop(order.index('en'))] + order
print(args.seed, order)
