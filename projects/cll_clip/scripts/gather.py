import argparse
import os
import re
import json
import pandas as pd
import configs
import glob
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('type', type=str, choices=['MT', 'ST', 'CL'])
parser.add_argument('--root', type=str, default='output/coco')
parser.add_argument('-s', '--specific', type=str, default='')
parser.add_argument('--dataset', type=str, default='xm3600')

parser.add_argument('--MT_candidate_format', type=str, default='{root}/*/*/*/evaluations_{task}/{dataset}')
parser.add_argument('--MT_model_range', type=int, nargs='+', default=[1, 5])

parser.add_argument('--ST_candidate_format', type=str, default='{root}/*/*/*/*/evaluations_{task}/{dataset}')
parser.add_argument('--ST_model_range', type=int, nargs='+', default=[1, 5])

parser.add_argument('--CL_candidate_format', type=str, default='{root}/*/*/*/*/*/evaluations_{task}/{dataset}')
parser.add_argument('--CL_transfer_format', type=str, default='{root}/*/*/{order}', 
                    help='path format for calculating backward and forward transfer')
parser.add_argument('--CL_order', type=str, default='order222')
parser.add_argument('--CL_model_range', type=int, nargs='+', default=[1, 5])

parser.add_argument('--csv_path', type=str, default='./results')
parser.add_argument('--csv_fn_format', type=str, default='{dataset}_{type}_{task}{postfix}.csv')
parser.add_argument('--sep', type=str, default=',')

parser.add_argument('--task', type=str, default='retrieval', choices=['retrieval'])
parser.add_argument('--retrieval_metrics', type=str, nargs='+', default=['txt_r1', 'txt_r5', 'txt_r10', 'img_r1', 'img_r5', 'img_r10', 'r_mean'])
parser.add_argument('--score_fn', type=str, default='test_scores.json')

parser.add_argument('-langs', '--languages', type=str, nargs='+')
parser.add_argument('--metrics', type=str, nargs='+')

parser.add_argument('--multi', type=float, default=1)
parser.add_argument('--ndigits', type=int, default=1)
parser.add_argument('--epoch', type=int, default=-1)
parser.add_argument('--eval', action='store_true')

parser.add_argument('--record_Fen', action='store_true')
parser.add_argument('--sort', action='store_true')
parser.add_argument('--skip_cl_csv', action='store_true')
args = parser.parse_args()


class Table:
    def __init__(self, languages=None, metrics=['r_mean']) -> None:
        if languages is None:
            if args.dataset == 'flickr30k':
                self.languages = configs.flickr30k_langs
            else:
                self.languages = configs.xm3600_langs
        else:
            self.languages = languages
        self.metrics = metrics
        self.models = []

        self.performance_table = {}
        for language in self.languages:
            self.performance_table[language] = defaultdict(list)

    def _add_scores(self, language, scores, prefix: str=''):
        for metric in self.metrics:
            score = scores.get(prefix + metric, 0)
            self.performance_table[language][metric].append(score)
    
    def MT_loop_languages(self, path, score_fn, model, **kwargs):
        if 'fused' in score_fn:
            for fusion_type in ['feature', 'score']:
                for fusion_scale in range(0, 11):
                    fusion_scale /= 10
                    tag = "%s_%.2f" % (fusion_type, fusion_scale)
                    self.models.append(f'{model}/{tag}')
                    for language in self.languages:
                        scores_path = os.path.join(path, language, score_fn)
                        if os.path.exists(scores_path):
                            scores = json.load(open(scores_path, 'r'))
                            self._add_scores(language, scores, prefix=f'{tag}_')
                        else:
                            for metric in self.metrics:
                                self.performance_table[language][metric].append(0)
        else:
            self.models.append(model)

            for language in self.languages:
                scores_path = os.path.join(path, language, score_fn)
                if os.path.exists(scores_path):
                    scores = json.load(open(scores_path, 'r'))
                    self._add_scores(language, scores)
                else:
                    for metric in self.metrics:
                        self.performance_table[language][metric].append(0)
    
    def ST_loop_languages(self, path, score_fn, model, epoch=None, **kwargs):
        if epoch is not None:
            model = f"{model}/{epoch}"

        if model in self.models:
            return
        self.models.append(model)
        
        # ientify which language is contained in this path
        for language in self.languages:
            if f'/{language}/' in path:
                break
        
        path_format = path.replace(f'/{language}/', '/{}/')
        
        for language in self.languages:
            scores_path = os.path.join(path_format.format(language), language, score_fn)
            if os.path.exists(scores_path):
                scores = json.load(open(scores_path, 'r'))
                self._add_scores(language, scores)
            else:
                for metric in self.metrics:
                    self.performance_table[language][metric].append(0)

    def CL_loop_languages(self, path, score_fn, model, epoch, **kwargs):
        return self.ST_loop_languages(path, score_fn, model, epoch, **kwargs)

    def CL_loop_transfer(self, path, score_fn, model, order, epoch=None, **kwargs):
        if not hasattr(self, 'cl_AA'):
            self.cl_AA = {}     # Average Accuracy (AA)
            self.cl_F = {}      # Forgetting (F)
            self.cl_Fen = {}    # Forgetting on English (Fen)
            for metric in self.metrics:
                self.cl_AA[metric] = defaultdict(list)
                self.cl_F[metric] = defaultdict(list)
                self.cl_Fen[metric] = defaultdict(list)

        if epoch is not None:
            model = f"{model}/{epoch}"

        if args.dataset == 'flickr30k':
            languages = configs.flickr30k_order1
        else:
            languages = getattr(configs, f'xm3600_{order}', None)
        assert languages is not None

        if args.record_Fen:
            assert 'en' in languages, languages
            assert languages.index('en') == 0, "English should be the first task"

        for metric in self.metrics:
            scores_table = np.zeros((len(languages), len(languages)))
            for current_idx in range(0, len(languages)):
                for past_idx in range(0, current_idx+1):
                    scores_path = os.path.join(
                        path.format(language=languages[current_idx]), 
                        languages[past_idx], 
                        score_fn
                    )

                    if os.path.exists(scores_path):
                        scores = json.load(open(scores_path, 'r'))
                        scores_table[current_idx, past_idx] = scores[metric]

            for i in range(len(languages)):
                AA = scores_table[i, :(i+1)].mean()
                self.cl_AA[metric][model].append(AA)
                
                if i > 0:
                    forget = []
                    for j in range(0, i):
                        current = scores_table[i, j]
                        maximun = scores_table[:i, j].max()
                        forget.append(maximun - current)
                    
                    # Forgetting (F)
                    F = sum(forget) / len(forget)
                    self.cl_F[metric][model].append(F)

                    # Forgetting on English (Fen)
                    Fen = forget[0]
                    self.cl_Fen[metric][model].append(Fen)
    
    def read_the_best_score(self, log_file, pattern='\[BEST\] score: (.*?),') -> float:
        if os.path.exists(log_file):    
            log_data = open(log_file, 'r').read().strip().split('\n')
            for line in log_data[::-1]:
                results = re.findall(pattern, line)
                assert len(results) in [0, 1]
                if len(results):
                    return float(results[0])
        return 0

    def EVAL_loop_languages(self, path, model, **kwargs):
        if not hasattr(self, 'validation_results'):
            self.validation_results = defaultdict(list)

        if model in self.validation_results:
            return
        
        # ientify which language is contained in this path
        for language in self.languages:
            if f'/{language}/' in path:
                break
        
        path_format = path.replace(f'/{language}/', '/{}/')
        log_file_format = os.path.join(path_format.split('{}')[0], '{}', 'log.txt')
        
        for language in self.languages:
            log_file = log_file_format.format(language)
            score = self.read_the_best_score(log_file)
            self.validation_results[model].append(score)

    def concat_data(self, prefix_data, data):
        assert len(prefix_data) == len(data)
        final_data = [item1 + item2 for item1, item2 in zip(prefix_data, data)]
        return final_data

    def convert_data(self, data, multi=args.multi, ndigits=args.ndigits):
        new_data = []
        for line in data:
            new_line = [(item if type(item) is str else round(item * multi, ndigits)) for item in line]
            new_data.append(new_line)
        return new_data

    def sort_(self, df):
        if args.sort:
            df.sort_values(by=['model'], ascending=True, inplace=True, kind='stable')

    def to_df(self, languages=None, metrics=None, add_average=False):
        languages = languages or self.languages
        metrics = metrics or self.metrics
        
        prefix_columns = ['model', 'metric']
        columns = [l for l in languages]
        
        if add_average:
            columns.append('Average')

        prefix_data, data = [], []
        for i in range(len(self.models)):
            for metric in self.metrics:
                prefix_line = [self.models[i], metric]
                line = [self.performance_table[l][metric][i] for l in languages]
                        
                if sum(line) == 0:
                    continue

                if add_average:
                    line.append(sum(line) / len(line))

                prefix_data.append(prefix_line)
                data.append(line)

        final_data = self.concat_data(prefix_data, self.convert_data(data))
        final_data = sorted(final_data, key=lambda x: x[0])
        
        df = pd.DataFrame(final_data, columns=prefix_columns + columns)
        self.sort_(df)
        return df.set_index('model')

    def to_cl_df_dict(self):
        df_dict = {}

        prefix_names = ['model', 'metric']        
        names = ['cl_AA', 'cl_F']
        names = names + (['cl_Fen'] if args.record_Fen else [])
        prefix_data, data = [], []

        models = sorted(list(self.cl_AA[self.metrics[0]].keys()))
        
        for model in models:
            for metric in self.metrics:
                prefix_data.append([model, metric])
                line = [getattr(self, name)[metric][model][-1] for name in names]
                data.append(line)
        
        final_data = self.concat_data(prefix_data, self.convert_data(data))
        
        df = pd.DataFrame(final_data, columns=prefix_names + names)
        df_dict['cl_overall'] = df.set_index('model')

        for name in names:
            prefix_data = [[model, metric] for model in models for metric in self.metrics]
            data = [getattr(self, name)[metric][model] for model in models for metric in self.metrics]
            final_data = self.concat_data(prefix_data, self.convert_data(data))
            columns = prefix_names + [str(_) for _ in range(len(final_data[0]) - len(prefix_names))]
            df = pd.DataFrame(final_data, columns=columns)
            self.sort_(df)
            df_dict[name] = df.set_index('model')

        return df_dict

    def to_eval_df(self):
        models = sorted(list(self.validation_results.keys()))
        columns = ['model'] + self.languages + ['Average', 'Valid Average']
        data = []
        for model in models:
            avg = sum(self.validation_results[model]) / len(self.languages)
            
            valid_nums = len([1 for score in self.validation_results[model] if score > 0])
            valid_avg = sum(self.validation_results[model]) / (valid_nums + 1e-9)
            
            line = [model] + self.validation_results[model] + [avg, valid_avg]
            data.append(line)

        df = pd.DataFrame(self.convert_data(data), columns=columns)
        self.sort_(df)
        return df.set_index('model')

metrics = getattr(args, f'{args.task}_metrics')
table = Table(metrics=metrics)

candidate_format = getattr(args, f'{args.type}_candidate_format')
if args.epoch == -1:
    index = candidate_format.index('/*')
    candidate_format = candidate_format[:index] + candidate_format[index+2:]

candidate_paths = glob.glob(candidate_format.format(root=args.root, task=args.task, dataset=args.dataset))

all_epochs = []
for path in candidate_paths:
    if args.specific not in path:
        continue

    if args.epoch == -1:
        model_range = getattr(args, f'{args.type}_model_range')
        model = '/'.join(path.split('/')[model_range[0]:model_range[1]-(1 if args.type != 'CL' else 0)])
        getattr(table, f'{args.type}_loop_languages')(path, args.score_fn, model=model, epoch=None)
        if args.eval and args.type in ['ST', 'CL']:
            table.EVAL_loop_languages(path, model=model)
    else:
        epoch = path.split('/')[-3]
        try:
            _ = int(epoch)
            all_epochs.append(epoch)
        except:
            continue
        if args.epoch is not None and epoch != str(args.epoch):
            continue
        model_range = getattr(args, f'{args.type}_model_range')
        model = '/'.join(path.split('/')[model_range[0]:model_range[1]])
        getattr(table, f'{args.type}_loop_languages')(path, args.score_fn, model=model, epoch=epoch)

df = table.to_df(args.languages, args.metrics, add_average=True)
print(df)

os.makedirs(args.csv_path, exist_ok=True)
save_path = os.path.join(
    args.csv_path, 
    args.csv_fn_format.format(dataset=args.dataset, type=args.type, task=args.task, postfix="")
)
df.to_csv(save_path, sep=args.sep)

if args.eval and args.type in ['ST', 'CL']:
    df = table.to_eval_df()
    print(df)
    save_path = os.path.join(
        args.csv_path, 
        args.csv_fn_format.format(dataset=args.dataset, type=args.type, task=args.task, postfix="_eval")
    )
    df.to_csv(save_path, sep=args.sep)

if args.type == 'CL' and not args.skip_cl_csv:
    paths = glob.glob(args.CL_transfer_format.format(root=args.root, order=args.CL_order))
    for path in paths:
        if args.specific not in path:
            continue

        model_range = getattr(args, f'{args.type}_model_range')
        model = '/'.join(path.split('/')[model_range[0]:model_range[1]])

        if args.epoch == -1:
            candidate_path = os.path.join(path, '{language}', f'evaluations_{args.task}', args.dataset)
            table.CL_loop_transfer(candidate_path, args.score_fn, model=model, epoch=None, order=args.CL_order)
        else:
            all_epochs = [args.epoch] if args.epoch else all_epochs
            for epoch in all_epochs:
                candidate_path = os.path.join(path, '{language}', f'{epoch}', f'evaluations_{args.task}', args.dataset)
                table.CL_loop_transfer(candidate_path, args.score_fn, model=model, epoch=epoch, order=args.CL_order)

    df_dict = table.to_cl_df_dict()
    print(df_dict['cl_overall'])
    for k, df in df_dict.items():
        save_path = os.path.join(
            args.csv_path, 
            args.csv_fn_format.format(dataset=args.dataset, type=args.type, task=args.task, postfix=f"_{k}")
        )
        df.to_csv(save_path, sep=args.sep)
