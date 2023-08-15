from util import *
import math
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import sys
from tqdm import tqdm 
# Add your import statements here



class InformationRetrieval_VSM():

	def __init__(self):
		self.td_matrix=None
		self.U = None
		self.Sigma = None
		self.VT = None
		self.terms=None
		self.sigma_inv=None
		self.k=None
		self.doc_projections=None
		self.td_matrix_tfidf=None
	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""


		# print("\n\ndocs=",len(docs))
		# print("\n\nnumber of documents docIDs=",len(docIDs)
		print("VSM started")
		index = {}
		# Build inverted index for each term in the document
		for i, doc in enumerate(docs):
			docID = docIDs[i]
			for sentence in doc:
				for term in sentence:
					if term not in index:
						index[term] = {}
					if docID not in index[term]:
						index[term][docID] = 0
					index[term][docID] += 1
		# Compute IDF scores for each term
		N = len(docs)
		idf = {term: math.log10(N/len(index[term])) for term in index}
		self.idf=idf
		#print("idf=\n",idf)
		# Compute TF-IDF scores for each term in each document
		tfidf = {}
		for term in index:
			for docID in index[term]:
				tf = index[term][docID]
				tfidf.setdefault(docID, {})[term] = tf * idf[term]

		self.index = tfidf
		#print("self.index=\n",self.index)

		# Convert the dictionary to a matrix for SVD
		terms = list(index.keys())
		#print("\n\nterms=",terms)
		self.vocab=terms
		#print("y axis=",terms)
		docs = list(tfidf.keys())
		#print("\n\ndocs=",docs)
		A = np.zeros((len(terms), len(docs)))
		for i, term in enumerate(terms):
			for j, doc in enumerate(docs):
				if doc in tfidf and term in tfidf[doc]:
					A[i, j] = tfidf[doc][term]
		#print("\n\nA shape=",A.shape)		
		self.td_matrix_tfidf=A

		terms = list(index.keys())
		self.terms=terms
		# Create a list of unique document IDs in the corpus
		docIDs = list(set([docID for term in index for docID in index[term]]))

		# Initialize the term-document matrix with zeros
		td_matrix = np.zeros((len(terms), len(docIDs)))

		# Populate the term-document matrix with term frequencies
		for i, term in enumerate(terms):
			for j, docID in enumerate(docIDs):
				if docID in index[term]:
					td_matrix[i, j] = index[term][docID]
		self.td_matrix=td_matrix
		# Transpose the term-document matrix to get the document-term matrix		
		return
  
  
		
	
	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""
		# print("\n\nqueries=",queries)
		ans=[]
		print("\n\nProcessing through queries")
		query_number=0
		for query in queries:
			query_number=query_number+1
			#print("\nQuery_Number ",query_number)
			term_count_of_q={}
			for sentence in query:
				for word in sentence:
					if(word in term_count_of_q.keys()):
						term_count_of_q[word]+=1
					else:
						term_count_of_q[word]=1
			# print("\n\nterm_count_of_q=",term_count_of_q)
			query_vector={}
			for word in term_count_of_q.keys():
				if(word in self.terms):
					query_vector[word]=term_count_of_q[word]
				else:
					query_vector[word]=0

			final_query_vector=np.zeros(len(self.terms))
			for i in range(len(final_query_vector)):
				if(self.terms[i] in query_vector.keys()):
					final_query_vector[i]=query_vector[self.terms[i]]*self.idf[self.terms[i]]
			

			sims=[]
			#print("final_query_vector=",final_query_vector)
			#print("self.index.keys()=",self.index.keys())
			docs=list(self.index.keys())
			for d in range(len(docs)):
				sim_value=np.dot(final_query_vector,self.td_matrix_tfidf[:,d])
				# print("doc no=",d)
				# print("self.td_matrix[:,d]=",self.td_matrix[:,d])
				# print("query",final_query_vector)
				# print("query_mag=",np.linalg.norm(final_query_vector))
				# print("doc=",self.td_matrix_tfidf[:,d])
				# print("doc_mag=",np.linalg.norm(self.td_matrix_tfidf[:,d]))
				sim_value=sim_value/(np.linalg.norm(final_query_vector)*np.linalg.norm(self.td_matrix_tfidf[:,d]))
				sims.append((docs[d],sim_value))
			sims.sort(key=lambda x:x[1],reverse=True)
			sims=[x[0] for x in sims]
			#print("sim=",sims)
			ans.append(sims)
		
		# print("\n\nVSM Ranking=",ans)
		#print("\n\nans.max=",max(ans[0]))
		# print("VSM_RANKING:-",ans)
		return(ans)





