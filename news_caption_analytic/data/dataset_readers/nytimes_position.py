import logging
import os
import random
import re
from typing import Dict

import numpy as np
import pymongo
import torch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm

from tell.data.fields import ImageField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


@DatasetReader.register('nytimes_position')
class NYTimesPositionReader(DatasetReader):
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

        roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
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
                      'parsed_section.hash',
                      'image_positions', 'headline', 'web_url']

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
                n_words = 0
                if title:
                    paragraphs.append(title)
                    n_words += len(self.to_token_ids(title))

                caption = sections[pos]['text'].strip()
                if not caption:
                    continue

                before = []
                after = []
                i = pos - 1
                j = pos + 1
                for k, section in enumerate(sections):
                    if section['type'] == 'paragraph':
                        paragraphs.append(section['text'])
                        break

                while True:
                    if i > k and sections[i]['type'] == 'paragraph':
                        text = sections[i]['text']
                        before.insert(0, text)
                        n_words += len(self.to_token_ids(text))
                    i -= 1

                    if k < j < len(sections) and sections[j]['type'] == 'paragraph':
                        text = sections[j]['text']
                        after.append(text)
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

                paragraphs = paragraphs + before + after

                yield self.article_to_instance(paragraphs, image, caption, image_path, article['web_url'], pos)

    def article_to_instance(self, paragraphs, image, caption, image_path, web_url, pos) -> Instance:
        context = '\n'.join(paragraphs).strip()

        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)

        fields = {
            'context': TextField(context_tokens, self._token_indexers),
            'image': ImageField(image, self.preprocess),
            'caption': TextField(caption_tokens, self._token_indexers),
        }

        metadata = {'context': context,
                    'caption': caption,
                    'web_url': web_url,
                    'image_path': image_path,
                    'image_pos': pos}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def to_token_ids(self, sentence):
        bpe_tokens = self.bpe.encode(sentence)
        words = tokenize_line(bpe_tokens)

        token_ids = []
        for word in words:
            idx = self.indices[word]
            token_ids.append(idx)
        return token_ids
