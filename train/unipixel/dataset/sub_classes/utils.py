# Modified from https://github.com/lichengunc/refer/blob/master/refer.py
# Originally writtern in Python 2, now supports Python 3

import itertools
import json
import os
import pickle


class REFER:

    def __init__(self, data_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        self.ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
        self.DATA_DIR = os.path.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = os.path.join(data_root, 'images/mscoco/images/train2014')
        elif dataset == 'refclef':
            self.IMAGE_DIR = os.path.join(data_root, 'images/saiapr_tc-12')
        else:
            print(f'No refer dataset is called [{dataset}]')
            exit()

        # load refs from data/dataset/refs(dataset).json
        ref_file = os.path.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset
        self.data['refs'] = pickle.load(open(ref_file, 'rb'))

        # load annotations from data/dataset/instances.json
        instances_file = os.path.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if isinstance(image_ids, list) else [image_ids]
        cat_ids = cat_ids if isinstance(cat_ids, list) else [cat_ids]
        ref_ids = ref_ids if isinstance(ref_ids, list) else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print(f'No such split [{split}]')
                    exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if isinstance(image_ids, list) else [image_ids]
        cat_ids = cat_ids if isinstance(cat_ids, list) else [cat_ids]
        ref_ids = ref_ids if isinstance(ref_ids, list) else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids
                         if image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if isinstance(ref_ids, list) else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if isinstance(ref_ids, list):
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif isinstance(ref_ids, int):
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if isinstance(ann_ids, list):
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif isinstance(ann_ids, int) or isinstance(ann_ids, str):
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if isinstance(image_ids, list):
            return [self.Imgs[image_id] for image_id in image_ids]
        elif isinstance(image_ids, int):
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if isinstance(cat_ids, list):
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif isinstance(cat_ids, int):
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]
