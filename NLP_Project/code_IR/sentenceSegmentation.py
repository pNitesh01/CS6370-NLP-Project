from util import *

# Add your import statements here

#import nltk
from nltk.tokenize import sent_tokenize


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		segmentedText=[]
		sentence= ""
		for i in range(len(text)):
			if text[i] in (".", "?", "!",):
					if i<len(text)-2 and text[i+1]==" " : # and text[i+2].isupper()
						sentence+=text[i]
						segmentedText.append(sentence.strip())
						sentence=""
					else:
						sentence+=text[i]
					
			else:
				sentence+=text[i]
		if sentence!="":
			segmentedText.append(sentence.strip())
		

		return segmentedText





	# def punkt(self, text):
	# 	"""
	# 	Sentence Segmentation using the Punkt Tokenizer

	# 	Parameters
	# 	----------
	# 	arg1 : str
	# 		A string (a bunch of sentences)

	# 	Returns
	# 	-------
	# 	list
	# 		A list of strings where each strin is a single sentence
	# 	"""
	# 	segmentedText = None
	# 	tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
	# 	segmentedText = tokenizer.tokenize(text)
	# 	#return sent_tokenize(text)
	# 	return segmentedText
	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		sentences = sent_tokenize(text)
		return sentences

