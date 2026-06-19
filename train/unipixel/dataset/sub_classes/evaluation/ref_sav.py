# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

from collections import OrderedDict

import nncore
from torch.utils.data import Dataset

from unipixel.dataset.hybrid import DATASETS


@DATASETS.register(name='ref_sav_eval')
class ReSAVEvalDataset(Dataset):

    META_DICT = 'data/ref_sav/valid/meta_expressions_valid.json'

    MASK_DICT = 'data/ref_sav/valid/mask_dict.json'

    DATA_ROOT = 'data/ref_sav/valid/videos'

    SOURCE = 'ref_sav_eval'

    @classmethod
    def load_annos(self, split='valid'):
        assert split == 'valid'

        meta_dict = nncore.load(self.META_DICT, object_pairs_hook=OrderedDict)['videos']
        mask_dict = nncore.load(self.MASK_DICT)

        annos = []
        for vid, meta in meta_dict.items():
            anno = dict(
                source=self.SOURCE,
                data_type='seg_video' if len(meta['frames']) > 1 else 'seg_image',
                frames=[nncore.join(self.DATA_ROOT, vid, f'{n}.jpg') for n in meta['frames']],
                vid=vid,
                samples=[])

            for qid, obj in meta['expressions'].items():
                assert len(obj['obj_id']) == len(obj['anno_id'])

                sample = dict(
                    qid=qid,
                    type='description',
                    query=obj['exp'].strip(),
                    mask_type='rle',
                    masks=[[m for m in zip(*[mask_dict[str(a)] for a in obj['anno_id']])]])

                assert len(sample['masks'][0]) == len(meta['frames'])
                assert any(any(o is not None for o in m) for m in sample['masks'][0])

                anno['samples'].append(sample)

            assert len(anno['samples']) > 0
            annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos
