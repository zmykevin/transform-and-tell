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

def extract_goodnews_data(ids, db, image_dir="data/goodnews/images"):
	result = {}
	for id_ in tqdm(ids):
	    sample = db.splits.find_one({'_id':{'$eq': id_}})
	    
	    article = db.articles.find_one({'_id': {'$eq': sample['article_id']}})
	    image_index = sample['image_index']
	    image_path = os.path.join(image_dir, f"{sample['_id']}.jpg")
	    if not os.path.isfile(image_path):
	        continue
	    image_caption = article['images'][image_index]
	    image_ner = article['caption_ner'][image_index]
	    
	    #convert the image_ner to the right format
	    spacy_image_ner = [[x['text'], x['label']] for x in image_ner]
	    has_person = True if "facenet_details" in sample else False
	    article_url = article['article_url']
	    #Y_M_D = article_url.split('/')[3:6]
	    found_year = False
	    i = 3
	    while not found_year:
	        Y_M_D = article_url.split('/')[i:i+3]
	        if len(Y_M_D[0]) == 4 and Y_M_D[0].isdigit():
	            found_year = True
	        else:
	            i += 1
	    assert found_year
	    time_stamp = f"{Y_M_D[0]}-{Y_M_D[1]}-{Y_M_D[2]}T00:00:00Z"
	    #print(time_stamp)
	    result[id_] = {"id": id_, "image_path": image_path, "caption": image_caption, "caption_entities_space": spacy_image_ner, "image_has_person": True, "timestamp": time_stamp}
	return result


def extract_nytimes_data(ids, db, image_dir = "data/nytimes/images"):
	
client = MongoClient(host='localhost', port=27017)
goodnews = client.goodnews
config = yaml_to_params("expt/goodnews/9_transformer_objects/config.yaml", overrides=None)
all_datasets = datasets_from_params(config)

#Load the Train Ids
sample_cursor = goodnews.splits.find({
            'split': {'$eq': 'train'},
        }, projection=['_id'], limit=0).sort('_id', pymongo.ASCENDING)

train_ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
sample_cursor.close()

#Load the Test Ids
# sample_cursor = goodnews.splits.find({
#             'split': {'$eq': 'test'},
#         }, projection=['_id'], limit=0).sort('_id', pymongo.ASCENDING)

# test_ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
# sample_cursor.close()

#Load the train data
goodnews_train = extract_goodnews_data(train_ids, goodnews)
#goodnews_test = extract_goodnews_data(test_ids, goodnews)

output_dir = "/home/zmykevin/semafor/code/news_clippings_generation/data/goodnews"
with open(os.path.join(output_dir,  "goodnews_train.json"), "w") as f:
    json.dump(goodnews_train, f)


