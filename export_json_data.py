import os
from pymongo import MongoClient
import pymongo
import numpy as np
from tqdm import tqdm
import json

client = MongoClient(host='localhost', port=27017)
db = client.goodnews

sample_cursor = db.splits.find({'split': {'$eq': 'val'},}, projection=['_id'], limit=0).sort('_id', pymongo.ASCENDING)

ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
sample_cursor.close()
print(ids.shape)

# sample_cursor = db.articles.find({'split': 'train',}, projection=['_id']).sort('_id', pymongo.ASCENDING)
# ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
# sample_cursor.close()
# print(ids.shape)

#Export the data to json file

data = []
image_dir = 'data/goodnews/images_processed'
for sample_id in tqdm(ids):
	try:
		sample = db.splits.find_one({'_id': {'$eq': sample_id}})
		article = db.articles.find_one({'_id': {'$eq': sample['article_id']}})
		url = article['web_url']
		title = article['headline']['main'].strip()

		#Get all the paragraphs
		#article_text = article['article']
		
		#print(article['context'])
		# sections = article['parsed_section']
		# paragraphs = []
		# for section in sections:
		#     if section['type'] == 'paragraph':
		#         paragraphs.append(section['text'])
		# article_text = '\n'.join(paragraphs)

		# pos = article['image_positions'][0]
		# current_caption = sections[pos]['text'].strip()
		for image_id, image_caption in article['images'].items():
			current_image_path = os.path.join(image_dir, f"{article['_id']}_{image_id}.jpg")
			if not os.path.isfile(current_image_path):
				continue
			current_caption = image_caption.strip()
			data.append({"url": url, "title": title, "image_dir": current_image_path, "caption": current_caption})
	except:
		continue
	#print(data)

with open('data/goodnews/goodnews_val_image_caption.json', "w") as f:
	json.dump(data, f)





