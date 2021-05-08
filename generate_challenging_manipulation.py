import pymongo
from pymongo import MongoClient
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
from allennlp.training.util import datasets_from_params
from tell.commands.train import yaml_to_params

import numpy as np
import os

client = MongoClient(host='localhost', port=27017)
goodnews = client.goodnews
config = yaml_to_params("expt/goodnews/9_transformer_objects/config.yaml", overrides=None)
all_datasets = datasets_from_params(config)


sample_cursor = goodnews.splits.find({
            'split': {'$eq': 'train'},
        }, projection=['_id'], limit=0).sort('_id', pymongo.ASCENDING)

ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
sample_cursor.close()

ner_mapping = {}
same_ner_type_retrieval = {}
same_ner_retrieval = {}
image_dir = "data/goodnews/images_processed"
FOCUSED_NER_TYPE = ["ORG", "GPE", "PERSON"]
for sample_id in tqdm(ids):
    sample = goodnews.splits.find_one({'_id': {'$eq': sample_id}})
    article = goodnews.articles.find_one({
                    '_id': {'$eq': sample['article_id']},
                }, projection=['_id', 'context', 'images', 'web_url', 'caption_ner', 'context_ner', 'caption_parts_of_speech'])
    
    #Now we want to find another article to play with
    image_index = sample['image_index']
    image_path = os.path.join(image_dir, f"{sample['_id']}.jpg")
    image_caption = article['images'][image_index]
    caption_ner = article['caption_ner'][image_index]
    caption_pos = article['caption_parts_of_speech'][image_index]
#     print(image_caption)
#     print(caption_ner)
#     print(caption_pos)
    for element in caption_ner:
        text = element['text']
        label = element['label']
        same_ner_type_retrieval[label] = same_ner_type_retrieval.get(label, []) + [sample_id]
        same_ner_retrieval[f"{text}_{label}"] = same_ner_retrieval.get(f"{text}_{label}", []) + [sample_id]
    ner_mapping[sample_id] = {"caption": image_caption, "caption_ner": caption_ner}
    #break

output_dir = "/home/zmykevin/semafor/code/transform-and-tell/data/goodnews/challenging_subset"
#Save the generated ner_mapping
with open(os.path.join(output_dir, "ner_swap_nermapping.json"), "w") as f:
	json.dump(ner_mapping, f)

with open(os.path.join(output_dir, "ner_swap_same_nertype.json"), "w") as f:
	json.dump(same_ner_type_retrieval, f)

with open(os.path.join(output_dir, "ner_swap_same_ner.json"), "w") as f:
	json.dump(same_ner_retrieval, f)