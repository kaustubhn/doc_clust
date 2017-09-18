import json, bz2
import pandas as pd
import logging
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Function to convert bson(mongodb dumps) to a csv file.
# Used to convert original crawl dumps to zipped files
def strip_data(filename):
	opstring = []
	with open(filename, 'r') as df:
		for l in df:
			d = json.loads(l)
			if(d['title'].strip()!= ""):
				if(d['description'].strip()!=""):
					opstring.append((d['title'], d['description']))
	df = pd.DataFrame(opstring)
	df.columns = ["title", "description"]
	df.to_csv(filename.split(".txt")[0]+".csv", index=False)
	return

# Create wordvectors
def create_vectors(filename):
	sentences = []
	with bz2.open(filename, 'rt') as df:
		for l in df:
			sentences.append(l.split())
	model = Word2Vec(sentences, size=25, window=5, min_count=5, workers=4)
	model.save(r"../vectors/"+filename.split("/")[-1].split(".")[0]+".vec")
	return model

	
# Call function on each file to create a csv file with title and description fields
# strip_data(r"../data/bengali.txt")
# strip_data(r"../data/hindi.txt")
# strip_data(r"../data/marathi.txt")
# strip_data(r"../data/tamil.txt")
# strip_data(r"../data/telugu.txt")

# Create vectors
# create_vectors(r"../data/bengali.csv.bz2")
# create_vectors(r"../data/hindi.csv.bz2")
# create_vectors(r"../data/marathi.csv.bz2")
# create_vectors(r"../data/tamil.csv.bz2")
# create_vectors(r"../data/telugu.csv.bz2")