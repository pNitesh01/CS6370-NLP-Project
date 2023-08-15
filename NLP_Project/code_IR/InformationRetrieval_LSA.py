from util import *
import math
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import sys
from tqdm import tqdm 
# Add your import statements here
import matplotlib.pyplot as plt
import time

import time




class InformationRetrieval_LSA():

	def __init__(self):
		self.td_matrix=None
		self.U = None
		self.Sigma = None
		self.VT = None
		self.terms=None
		self.sigma_inv=None
		self.k=None
		self.doc_projections=None
		self.index=None
		self.idf=None
		self.td_matrix_tfidf=None
	def buildIndex(self, docs, docIDs,rank):
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

		self.k=rank
		# print("\n\ndocs=",len(docs))
		# print("\n\nnumber of documents docIDs=",len(docIDs)
		print("\n\nLSA STARTED")
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

		# Convert the dictionary to a matrix for SVD
		terms = list(index.keys())
		#print("\n\nterms=",terms)
		self.vocab=terms
		docs = list(tfidf.keys())
		docs_left=set(docIDs).difference(set(docs))
		print("\n\ndocs_left",docs_left)
		print("\n\nlist(tfidf.keys())=",len(docs))
		A = np.zeros((len(terms), len(docs)))
		for i, term in enumerate(terms):
			for j, doc in enumerate(docs):
				if doc in tfidf and term in tfidf[doc]:
					A[i, j] = tfidf[doc][term]
		print("\n\nA.shape=",A.shape)	
		print("\n\nA=",A)
  
  
  
		# TF-ID MATRIX CREATED	
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
		dt_matrix = td_matrix.T

		


		U, s, VT = svd(self.td_matrix_tfidf)
		self.U = U
		self.Sigma = s
		self.VT = VT
		ssq_total = np.sum(s**2)
		
		#k-Rank approximation
		# print("Before trimming, U.shape=",self.U.shape)
		self.U=self.U[:,:self.k]
		# print("After trimming, U.shape=",self.U.shape)
  
		# print("Print U=",self.U)
		# print("After trimming, U=",self.U[:,:self.k])
  
		# Compute the explained variance ratio for different numbers of singular values
		# var_ratio = np.cumsum(s**2) / ssq_total

		# # Plot the explained variance ratio as a function of the number of singular values
		# plt.plot(var_ratio)
		# plt.xlabel('Number of Singular Values')
		# plt.ylabel('Explained Variance Ratio')
		# plt.show()
		# plt.clf()
		# # Initialize the variance explained and number of principal components
		# var_explained = 0.0
		# num_pc = 0

		# Compute the variance explained and number of principal components
		# while var_explained < 0.8:
		# 	var_explained += s[num_pc]**2 / ssq_total
		# 	num_pc += 1
		# print("\n\nNUM_PC=",num_pc)

  
		#k=len(s)
		

		self.Sigma=1/s[:self.k]
		#self.Sigma=self.Sigma[:self.k,:self.k]
		self.sigma_inv=np.diag(1/self.Sigma)
  
		
		doc_projections=[]
		num_of_docs=self.td_matrix_tfidf.shape[1]
		print("\n\n num_of_docs=",num_of_docs)
		for i in tqdm(range(num_of_docs)):
			# print("\n\nself.sigma_inv=",self.sigma_inv.shape)
			# print("\n\nself.U.T=",self.U.T.shape)
			# print("\n\nself.td_matrix_tfidf[:,i].T=",self.td_matrix_tfidf[:,i].T.shape)
			#print("docnumber=",i,"  doc_vector=",td_matrix[:,i].T)
			#doc_projections.append(np.dot(np.dot(sigma_inv[:self.k,:self.k],U[:,:self.k].T),self.td_matrix_tfidf[:,i].T))
			doc_projections.append(np.dot(np.dot(self.sigma_inv,self.U.T),self.td_matrix_tfidf[:,i].T))
		self.doc_projections=doc_projections
		# print("\n\ndoc_projections=",doc_projections)
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
		print("\n\nLSA Rank Function Called. Processing through queries now..")
		query_number=0
		for query in queries:
			query_number+=1
			#print("\nQuery_Number ",query_number+1)
			term_count_of_q={}
			for sentence in query:
				for word in sentence:
					if(word in term_count_of_q.keys()):
						term_count_of_q[word]+=1
					else:
						term_count_of_q[word]=1
			#print("\n\nterm_count_of_q=",term_count_of_q)
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
			#print("\nquery_vector=",final_query_vector)
			#query_projection=np.dot(np.dot(self.sigma_inv[:self.k,:self.k],self.U[:,:self.k].T),final_query_vector.T)
			query_projection=np.dot(np.dot(self.sigma_inv,self.U.T),final_query_vector.T)

			sims=[]
			docs=list(self.index.keys())
			for d in range(self.td_matrix_tfidf.shape[1]):
				sim_value=np.dot(query_projection,self.doc_projections[d])
				sim_value=sim_value/(np.linalg.norm(query_projection)*np.linalg.norm(self.doc_projections[d]))
				sims.append((docs[d],sim_value))
			sims.sort(key=lambda x:x[1],reverse=True)
			sims=[x[0] for x in sims]
			ans.append(sims)
		# print("\n\nLSA Ranking=",ans)
		return(ans)





