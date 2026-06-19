# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import copy

import nncore

from unipixel.dataset.hybrid import DATASETS
from unipixel.dataset.utils import process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import get_duration, load_subtitle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--pred_path')
    parser.add_argument('--model_path')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--style', default='mcq', choices=['mcq', 'options', 'direct', 'short'])
    parser.add_argument('--use_subtitle', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'output.json')

    print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    model, processor = build_model(args.model_path, device=args.device)
    device = next(model.parameters()).device

    annos = DATASETS.get(args.dataset).load_annos(split=args.split)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        dump = copy.deepcopy(annos[i])

        video_path, duration, span = anno['video_path'], anno.get('duration'), anno.get('span')

        print()
        print(video_path)

        if args.style in ('mcq', 'options'):
            prompt = anno['question'] + '\nOptions:'
            for idx, opt in enumerate(anno['options']):
                prompt += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
            prompt += '\nPlease only give the best option.'
        elif args.style == 'short':
            prompt = anno['question'] + ' Please answer with a single word or phrase.'
        else:
            prompt = anno['question']

        print(prompt)
        print(anno.get('ans'))
        print(anno.get('answer'))

        if args.use_subtitle and 'subtitle_path' in anno and nncore.is_file(anno['subtitle_path']):
            if duration is None:
                duration = get_duration(video_path, num_threads=args.num_threads)

            # use only the first 100 subtitles to save memory
            subs = load_subtitle(anno['subtitle_path'])[:100]
            subs = [f'{round(s, 1)}s - {round(e, 1)}s, {t}\n' for s, e, t in subs]
            subs = ''.join(subs)
            prompt = f'You are given a video with {round(duration, 1)} seconds long.\nSubtitles:\n{subs}' + prompt

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'num_threads': args.num_threads,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28,
                'max_frames': 16,
                'fps': 2.0
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        text += 'Best Option: (' if args.style == 'mcq' else ''
        print(text)
        images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)
        data = data.to(device)

        output_ids = model.generate(
            **data,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=256)

        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]

        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]

        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

        print(response)

        dump['response'] = response

        dumps.append(dump)

    nncore.dump(dumps, pred_path)
