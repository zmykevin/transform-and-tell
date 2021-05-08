import os
import shutil

if __name__ == "__main__":
	data_path = "/home/zmykevin/semafor/code/transform-and-tell/data/semafor/dry_run"

	#list all the tar.gz file from data_path
	label = ["Manipulated", "Pristine"]
	for l in label:
		gz_data = [x for x in os.listdir(os.path.join(data_path, l)) if not x.endswith(".tar.gz")]
		print(gz_data)
    