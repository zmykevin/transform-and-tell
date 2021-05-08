	#Explore the bias in the data
from pymongo import MongoClient
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
STOP_WORDS = {x:True for x in stopwords.words('english')}

def remove_stopwords(word_list):
    filtered_word_list = []
    for w in word_list:
        if STOP_WORDS.get(w, None) is not None:
            filtered_word_list.append(w)
    return filtered_word_list
def process_text(raw_string):
    #tokenize the string
    raw_words = word_tokenize(raw_string)
    #remove the stop words from the row_words
    filtered_words = list(set(remove_stopwords(raw_words)))
    return filtered_words


if __name__ == "__main__":
	#Get the Client
	client = MongoClient(host='localhost', port=27017)
	goodnews = client.goodnews
	sample_cursor = goodnews.splits.find(
	        {'split': 'train'}, no_cursor_timeout=True).batch_size(1)
	overlap_summary = []
	for sample in tqdm(sample_cursor):
	#     print(sample.keys())
	    article_id  = sample['article_id']
	    article = goodnews.articles.find_one({
	            '_id': {'$eq': sample['article_id']},
	        })
	    #print(article.keys())
	    #Print caption
	    #Print article
	    body = article['article']
	    body_words = {x: True for x in process_text(body.strip())}
	    abstract = article['abstract']
	    captions = article['images']
	    for key, cap in captions.items():
	        #tokneize the cap
	        cap_words = process_text(cap.strip())
	        if len(cap_words) == 0:
	            continue
	        #Check overlap
	        overlap_count = 0
	        for w in cap_words:
	            if body_words.get(w, None) is not None:
	                overlap_count += 1
	        overlap_ratio = overlap_count / len(cap_words)
	        overlap_summary.append(overlap_ratio)

	print(sum(overlap_summary)/len(overlap_summary))
