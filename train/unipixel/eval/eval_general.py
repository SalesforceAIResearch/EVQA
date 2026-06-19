# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse

import nncore
from tabulate import tabulate


class SafeInt(int):

    def __truediv__(self, other):
        try:
            return SafeInt(super().__truediv__(other))
        except ZeroDivisionError:
            return SafeInt(0)


def check_ans(options, ans, response):
    a = ans.lower()
    b = response.lower().split(' ')[0].replace('(', '').replace(')', '').replace('.', '')
    if len(b) != 1:
        b = b[0]
        nncore.log(f'WARNING: {response} -> {b}')
    if b not in [chr(ord('a') + i) for i in range(len(options))]:
        nncore.log(f'ERROR: {response} -> {b}')
        return
    return a == b


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path')
    parser.add_argument('--out_name', default='metrics.log')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert nncore.is_dir(args.pred_path)

    log_file = nncore.join(args.pred_path, args.out_name)
    nncore.set_default_logger(logger='eval', fmt=None, log_file=log_file)

    pred_paths = nncore.ls(args.pred_path, ext=['json', 'jsonl'], join_path=True)
    nncore.log(f'Total number of files: {len(pred_paths)}')

    tab_ans = dict()
    tab_ans_all = [SafeInt(0) for _ in range(3)]

    for path in pred_paths:
        data = nncore.load(path)

        for sample in data:
            task = sample.get('task', 'unknown')

            if isinstance(task, str):
                task = [task]

            for t in task:
                if t not in tab_ans:
                    tab_ans[t] = [SafeInt(0) for _ in range(3)]

            if 'response' in sample:
                for t in task:
                    tab_ans[t][0] += 1
                tab_ans_all[0] += 1

                correct = check_ans(sample['options'], sample['ans'], sample['response'])

                if correct:
                    for t in task:
                        tab_ans[t][2] += 1
                    tab_ans_all[2] += 1
                elif correct is None:
                    for t in task:
                        tab_ans[t][1] += 1
                    tab_ans_all[1] += 1

    tasks = sorted(list(tab_ans.keys()))

    tab = tabulate(
        [[task, tab_ans[task][0], tab_ans[task][1], f'{tab_ans[task][2] / tab_ans[task][0] * 100:.2f}']
         for task in tasks if task in tab_ans] +
        [['all', tab_ans_all[0], tab_ans_all[1], f'{tab_ans_all[2] / tab_ans_all[0] * 100:.2f}']],
        headers=['Task', '#Samples', 'Failed', 'Acc'],
        tablefmt='pretty',
        stralign='left')
    nncore.log(f'\n{tab}')
