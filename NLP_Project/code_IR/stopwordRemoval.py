from util import *

# Add your import statements here

import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords


class StopwordRemoval():

	def fromList(self, text):
		# """
		# Sentence Segmentation using the Punkt Tokenizer

		# Parameters
		# ----------
		# arg1 : list
		# 	A list of lists where each sub-list is a sequence of tokens
		# 	representing a sentence

		# Returns
		# -------
		# list
		# 	A list of lists where each sub-list is a sequence of tokens
		# 	representing a sentence with stopwords removed
		# """
		# stop_words=set(stopwords.words('english'))
		# stop_words.add(".")
		# stopwordRemovedText = []
		# for sentence in text:
		# 	stopword_free_sentence = [i for i in sentence if i.lower() not in stop_words]
		# 	stopwordRemovedText.append(stopword_free_sentence)
		# return stopwordRemovedText

		"""
        Stopword Removal using NLTK Stopwords

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """
		stop_words = set(stopwords.words('english'))
		stop_words.add(".")
		filtered_text = []
		for sentence in text:
			filtered_sentence = [word for word in sentence if word.lower() not in stop_words]
			filtered_text.append(filtered_sentence)
		return filtered_text

# temp=StopwordRemoval()
# x=temp.fromList([["the","bird","was",",","blue","."]])
# print(x)



	
