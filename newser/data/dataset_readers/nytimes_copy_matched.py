import logging
import os
import random
import re
from collections import OrderedDict
from datetime import datetime
from typing import Dict

import numpy as np
import pymongo
import torch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from newser.data.fields import CopyTextField, ImageField, ListTextField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


@DatasetReader.register('nytimes_copy_matched')
class NYTimesCopyMatchedReader(DatasetReader):
    """Read from the New York Times dataset.

    See the repo README for more instruction on how to download the dataset.

    Parameters
    ----------
    tokenizer : ``Tokenizer``
        We use this ``Tokenizer`` for both the premise and the hypothesis.
        See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``
        We similarly use this for both the premise and the hypothesis.
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 image_dir: str,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.client.nytimes
        self.image_dir = image_dir
        self.preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        # validation and test sets contain 10K examples each
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        logger.info('Grabbing all article IDs')
        sample_cursor = self.db.articles.find({
            'split': split,
        }, projection=['_id']).sort('_id', pymongo.ASCENDING)
        ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
        sample_cursor.close()
        self.rs.shuffle(ids)

        projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                      'parsed_section.hash', 'parsed_section.parts_of_speech',
                      #   'parsed_section.facenet_details',
                      'parsed_section.named_entities',
                      'image_positions', 'headline',
                      'web_url', 'n_images_with_faces']

        for article_id in ids:
            article = self.db.articles.find_one(
                {'_id': {'$eq': article_id}}, projection=projection)
            sections = article['parsed_section']
            image_positions = article['image_positions']
            for pos in image_positions:
                title = ''
                if 'main' in article['headline']:
                    title = article['headline']['main'].strip()
                paragraphs = []
                pos_pars = []
                # named_entities = set()
                n_words = 0
                if title:
                    paragraphs.append(title)
                    pos_pars.append(article['headline']['parts_of_speech'])
                    # named_entities.union(
                    #     self._get_named_entities(article['headline']))
                    n_words += len(self.to_token_ids(title))

                caption = sections[pos]['text'].strip()
                if not caption:
                    continue

                copy_infos = self._get_caption_names(sections[pos])

                # n_persons = len(self._get_person_names(sections[pos]))

                before = []
                before_pos = []
                after = []
                after_pos = []
                i = pos - 1
                j = pos + 1
                for k, section in enumerate(sections):
                    if section['type'] == 'paragraph':
                        paragraphs.append(section['text'])
                        pos_pars.append(section['parts_of_speech'])
                        # named_entities |= self._get_named_entities(section)
                        break

                while True:
                    if i > k and sections[i]['type'] == 'paragraph':
                        text = sections[i]['text']
                        before.insert(0, text)
                        before_pos.insert(0, sections[i]['parts_of_speech'])
                        # named_entities |= self._get_named_entities(sections[i])
                        n_words += len(self.to_token_ids(text))
                    i -= 1

                    if k < j < len(sections) and sections[j]['type'] == 'paragraph':
                        text = sections[j]['text']
                        after.append(text)
                        after_pos.append(sections[j]['parts_of_speech'])
                        # named_entities |= self._get_named_entities(sections[j])
                        n_words += len(self.to_token_ids(text))
                    j += 1

                    if n_words >= 510 or (i <= k and j >= len(sections)):
                        break

                image_path = os.path.join(
                    self.image_dir, f"{sections[pos]['hash']}.jpg")
                try:
                    image = Image.open(image_path)
                except (FileNotFoundError, OSError):
                    continue

                # if 'facenet_details' not in sections[pos] or n_persons == 0:
                #     face_embeds = np.array([[]])
                # else:
                #     face_embeds = sections[pos]['facenet_details']['embeddings']
                #     # Keep only the top faces (sorted by size)
                #     face_embeds = np.array(face_embeds[:n_persons])

                paragraphs = paragraphs + before + after
                pos_pars = pos_pars + before_pos + after_pos
                self._process_copy_tokens(copy_infos, paragraphs, pos_pars)
                # named_entities = sorted(named_entities)

                yield self.article_to_instance(copy_infos, paragraphs, image, caption, image_path, article['web_url'], pos)

    def article_to_instance(self, copy_infos, paragraphs, image, caption, image_path, web_url, pos) -> Instance:
        context = '\n'.join(paragraphs).strip()

        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)
        # name_token_list = [self._tokenizer.tokenize(n) for n in named_entities]

        # if name_token_list:
        #     name_field = [TextField(tokens, self._token_indexers)
        #                   for tokens in name_token_list]
        # else:
        #     stub_field = ListTextField(
        #         [TextField(caption_tokens, self._token_indexers)])
        #     name_field = stub_field.empty_field()

        fields = {
            'context': CopyTextField(context_tokens, self._token_indexers, copy_infos, 'context'),
            # 'names': ListTextField(name_field),
            'image': ImageField(image, self.preprocess),
            'caption': CopyTextField(caption_tokens, self._token_indexers, copy_infos, 'caption'),
            # 'face_embeds': ArrayField(face_embeds, padding_value=np.nan),
        }

        metadata = {'context': context,
                    'caption': caption,
                    # 'names': named_entities,
                    'web_url': web_url,
                    'image_path': image_path,
                    'image_pos': pos}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def _get_named_entities(self, section):
        # These name indices have the right end point excluded
        names = set()

        if 'named_entities' in section:
            ners = section['named_entities']
            for ner in ners:
                if ner['label'] in ['PERSON', 'ORG', 'GPE']:
                    names.add(ner['text'])

        return names

    def _get_person_names(self, section):
        # These name indices have the right end point excluded
        names = set()

        if 'named_entities' in section:
            ners = section['named_entities']
            for ner in ners:
                if ner['label'] in ['PERSON']:
                    names.add(ner['text'])

        return names

    def to_token_ids(self, sentence):
        bpe_tokens = self.bpe.encode(sentence)
        words = tokenize_line(bpe_tokens)

        token_ids = []
        for word in words:
            idx = self.indices[word]
            token_ids.append(idx)
        return token_ids

    def _get_caption_names(self, section):
        copy_infos = {}

        parts_of_speech = section['parts_of_speech']
        for pos in parts_of_speech:
            if pos['pos'] == 'PROPN' and self.is_in_ner(pos['text'], section):
                if pos['text'] not in copy_infos:
                    copy_infos[pos['text']] = OrderedDict({
                        'caption': [(pos['start'], pos['end'])],
                        'context': []
                    })
                else:
                    copy_infos[pos['text']]['caption'].append(
                        (pos['start'], pos['end']))

        return copy_infos

    def _process_copy_tokens(self, copy_infos, paragraphs, pos_pars):
        for name, info in copy_infos.items():
            offset = 0
            for par, pos_par in zip(paragraphs, pos_pars):
                for pos in pos_par:
                    if pos['pos'] == 'PROPN' and pos['text'] == name:
                        info['context'].append((
                            pos['start'] + offset,
                            pos['end'] + offset,
                        ))
                offset += len(par) + 1

    def is_in_ner(self, text, section):
        ners = section['named_entities']
        for ner in ners:
            if text in ner['text']:
                return True
        return False
