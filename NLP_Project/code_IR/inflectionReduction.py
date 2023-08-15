from util import *

# Add your import statements here
from nltk.stem import PorterStemmer
import spacy

class InflectionReduction:
    def __init__(self, language_model="en_core_web_sm"):
        self.nlp = spacy.load(language_model)
    
    def reduce(self, text):
        """Lemmatize each token in a list of sentences."""
        lemmatized_text = []
        for sentence in text:
            # Join the tokens in the sentence back into a string
            sentence = ' '.join(sentence)
            
            # Parse the sentence with spaCy
            doc = self.nlp(sentence)
            
            # Lemmatize each token and add it to the new sentence
            lemmatized_sentence = []
            for token in doc:
                lemmatized_sentence.append(token.lemma_)
            
            # Add the lemmatized sentence to the new text
            lemmatized_text.append(lemmatized_sentence)
        
        return lemmatized_text

# class InflectionReduction:

# 	def reduce(self, text):
# 		"""
# 		Stemming/Lemmatization

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of lists where each sub-list a sequence of tokens
# 			representing a sentence

# 		Returns
# 		-------
# 		list
# 			A list of lists where each sub-list is a sequence of
# 			stemmed/lemmatized tokens representing a sentence
# 		"""
# 		porter = PorterStemmer()




# 		reducedText = []

# 		for i in range(len(text)):
# 				temp=[]
# 				for j in range(len(text[i])):
# 					temp.append(porter.stem(text[i][j]))
# 				reducedText.append(temp)
# 				del temp
					

# 		#Fill in code here
# 		return reducedText


