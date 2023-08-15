from util import *

# Add your import statements here


#import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		tokenizedText=[]
		for elm in text:
			word = ''
			temp = []
			for i in range(len(elm)):
				if elm[i] in (" ",'.',',','?','!','@','#','%','^','&','*','(',')',':','-','+','{','[','}',']',';',':',"'",'"','<','>','/'):
					if word!='':
						temp.append(word.lower())
						word=''
				else:
					word+=elm[i]
			if word != '':
				temp.append(word.lower())
			tokenizedText.append(temp)
			

		return tokenizedText
		#tokenizedText = None

		#Fill in code here

		#return tokenizedText


	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		tokenizer = TreebankWordTokenizer()
		tokenized_sentences = []
		for sentence in text:
			tokenized_sentence = tokenizer.tokenize(sentence)
			tokenized_sentences.append(tokenized_sentence)
		return tokenized_sentences
	# def pennTreeBank(self, text):
	# 	"""
	# 	Tokenization using the Penn Tree Bank Tokenizer

	# 	Parameters
	# 	----------
	# 	arg1 : list
	# 		A list of strings where each string is a single sentence

	# 	Returns
	# 	-------
	# 	list
	# 		A list of lists where each sub-list is a sequence of tokens
	# 	"""
		
	# 	#Fill in code here
	# 	tokenizedText = []
	# 	for elm in text:
	# 		tokenizedText.append(nltk.word_tokenize(elm))
	# 	return tokenizedText
	






