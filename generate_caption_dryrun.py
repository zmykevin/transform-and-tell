import json
import os
import torch
from Captioner import Captioner


if __name__ == "__main__":
	print("Loading data ...")
	data_root = "/home/zmykevin/semafor/code/transform-and-tell/data/semafor/dry_run"
	label = 'Pristine'
	all_dir = [x for x in os.listdir(os.path.join(data_root, label)) if not x.endswith(".tar.gz")]
	#print(all_dir)


	articles = []
	for article_id in all_dir:
		full_data_path = '/'.join([data_root, label, article_id, "{}.json".format(article_id)])
		article_data = json.load(open(full_data_path, "r"))
		articles.append((article_data, article_id))
		# if article_id == "b25eaff0c9e62d740c4e9eeb31d64e76":
		# 	articles.append((article_data, article_id))

	# sample_data = articles[0][0]
	#print(sample_data)
	results = {}
	for article in articles:
		article_id = article[1]
		sample_data = article[0]
		for content_item in sample_data['content']:
			if content_item['Type'] == "Figure":
				#print(content_item['Media'][0]['Type'])
				assert content_item['Media']
				assert content_item['Media'][0]['Type'] in ["image", "Image"]
				#print("The original caption is: {}".format(content_item['Caption']))
				results[article_id] = {'caption': content_item['Caption'], 'label': "entailment"}
				break

	
	# for content_item in sample_data['figures']:
	# 	print(content_item['media'][0]['uri'].split('\\'))
	with torch.no_grad():
		captioner = Captioner('data/semafor/dry_run/{}'.format(label))
		captioner.initialize()

		#generate the caption
		#output_dict = captioner.generate_caption(articles)
		# caption = output_dict[0]['caption'][0]
		# print("The generated caption is: {}".format(caption))
		# for od in output_dict:
		# 	current_aid = od['article_id']
		# 	current_caption = od['caption'][0]
		# 	results[current_aid]['generation'] = current_caption
		for article in articles:
			sample_data = [article]
			current_aid = article[1]
			#try:
			output_dict = captioner.generate_caption(sample_data)
			current_caption = output_dict[0]['caption'][0]
			results[current_aid]['generation'] = current_caption
			# except:
			# 	results[current_aid]['generation'] = None
			# 	continue


	print(results)
	with open('data/semafor/dry_run_{}.json'.format(label), "w") as f:
		json.dump(results, f)