'''This file is responsible for calcualting the training time of different models
    running examples:
    - python3 scripts/get_training_time.py --task CL
    - python3 scripts/get_training_time.py --task MT
'''
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='output/coco')
parser.add_argument('--task', type=str, default='CL', choices=['CL', 'MT', 'ST'])
args = parser.parse_args()


def run(path_format):
    n = 0
    h, m, s = 0, 0, 0
    for path in glob.glob(path_format):
        n += 1
        data = open(path, 'r').read().strip().split('\n')[::-1]
        for line in data:
            for string in ['Train time: ', 'End task time: ']:
                if string in line:
                    this_h, this_m, this_s = line.lstrip(string).split(':')
                    h += float(this_h)
                    m += float(this_m)
                    s += float(this_s)
            if '\ttime: ' in line:
                s += float(line.split(' ')[1]) 

    s = h * 3600 + m * 60 + s
    return n, s


root = os.path.join(args.root, args.task)
assert os.path.exists(root), root

for model_path in sorted(glob.glob(os.path.join(root, '*'))):
    if os.path.isdir(model_path):
        if args.task == 'CL':
            path_format = os.path.join(model_path, '*', '*', 'log.txt')
        elif args.task == 'ST':
            path_format = os.path.join(model_path, '*', 'log.txt')
        else:
            path_format = os.path.join(model_path, 'log.txt')

        n, s = run(path_format)
        print(f"{model_path}, {n} tasks, {s/3600:.1f} hours")
