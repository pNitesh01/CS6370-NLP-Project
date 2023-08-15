from util import *
import math
# Add your import statements here




class Evaluation():
			
	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""
		# print("query_doc_IDs_ordered=",query_doc_IDs_ordered)
		# print("query_id=",query_id)
		# print("true_doc_IDs=",true_doc_IDs)
		# print("k=",k)
		# Get the top k documents from the ordered list of predicted document IDs
		top_k_docs = query_doc_IDs_ordered[:k]
		# print("top_k_docs=",top_k_docs)
		# Calculate the number of true positives (relevant documents in the top k)
		true_positives = len(set(top_k_docs).intersection(set(true_doc_IDs)))
		# print("true_positives=",true_positives)
		# Calculate precision as true positives divided by k
		precision = true_positives / k
		# print("precision=",precision)
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""
		# print("doc_IDs_ordered=",doc_IDs_ordered)
		# print("query_ids=",query_ids)
		# print("qrels=",qrels)
		# print("k=",k)
		# Initialize the sum of precision values across all queries to 0
		total_precision = 0
		
		# Loop through each query and compute its precision value at k
		for i, query_id in enumerate(query_ids):
			# print("i,query_id=",i,query_id)
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			# print("query_doc_IDs_ordered=",query_doc_IDs_ordered)
			# for d in qrels:
				#print(int(d["position"])<=2)
				# print("d['query_num'],d['position']=",d['query_num'],d['position'])
				# print(int(d['query_num'])==query_id and int(d["position"])<=2)
			# Retrieve the ground truth relevant documents for the query
			# true_doc_IDs = [int(d["id"]) for d in qrels if int(d["query_num"])==query_id and int(d["position"])<=2]
			# true_doc_IDs = list(set([int(d["id"]) for d in qrels if int(d["query_num"])==query_id and int(d["position"])==1]))
			true_doc_IDs = [int(d["id"]) for d in qrels if int(d["query_num"])==query_id]
			# print("true_doc_IDs=",true_doc_IDs)
			
			# Compute the precision value for the query at k
			precision_k = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
			
			# Add the precision value for the query to the total
			total_precision += precision_k
		
		# Compute the mean precision value across all queries
		meanPrecision = total_precision / len(query_ids)
		
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		# Get the top k documents from the ordered list of predicted document IDs
		top_k_docs = query_doc_IDs_ordered[:k]
		# Calculate the number of true positives (relevant documents in the top k)
		true_positives = len(set(top_k_docs).intersection(set(true_doc_IDs)))
		# Calculate the total number of relevant documents in the ground truth
		relevant_docs = len(true_doc_IDs)
		# Calculate recall as true positives divided by total number of relevant documents
		recall = true_positives / relevant_docs if relevant_docs != 0 else 0
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		# Initialize the sum of recall values across all queries to 0
		total_recall = 0

		# Loop through each query and compute its recall value at k
		for i, query_id in enumerate(query_ids):
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			# Retrieve the ground truth relevant documents for the query
			# true_doc_IDs = [int(d["id"]) for d in qrels if int(d["query_num"])==query_id and int(d["position"])<=2]
			# true_doc_IDs = list(set([int(d["id"]) for d in qrels if int(d["query_num"])==query_id and int(d["position"])==1]))
			true_doc_IDs = [int(d["id"]) for d in qrels if int(d["query_num"])==query_id]

			# Compute the recall value for the query at k
			recall_k = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

			# Add the recall value for the query to the total
			total_recall += recall_k

		# Compute the mean recall value across all queries
		meanRecall = total_recall / len(query_ids)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1
		P=self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		R=self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		if(P==0 and R==0):
			return(0)
		fscore=(2*P*R)/(P+R)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		fscores = []
		for i, query_id in enumerate(query_ids):
			# true_doc_IDs = [int(d["id"]) for d in qrels if int(d["query_num"])==query_id and int(d["position"])<=2]
			# true_doc_IDs = list(set([int(d["id"]) for d in qrels if int(d["query_num"])==query_id and int(d["position"])==1]))
			true_doc_IDs = [int(d["id"]) for d in qrels if int(d["query_num"])==query_id]
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			fscore = self.queryFscore(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
			fscores.append(fscore)

		meanFscore = sum(fscores) / len(fscores)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs,k,qrels):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query 
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""
		pred_rel_scores=[]
		for doc_id in query_doc_IDs_ordered[:k]:
			# print("\n\ndoc_id=",doc_id)
			for item in qrels:
				if(int(item["query_num"])==query_id and int(item["id"])==doc_id):
					pred_rel_scores.append(4-item["position"])
		# print("\n\npred_rel_scores=",pred_rel_scores)
		dcg=0
		for i in range(len(pred_rel_scores)):
			index=i+1
			rel_index=pred_rel_scores[i]
			temp=rel_index/math.log2(index+1)
			dcg+=temp
   
		#Calculate actual rel scores
		# actual_rel_scores=[]
		# for doc_id in true_doc_IDs[:k]:
		# 	for item in qrels:
		# 		if(int(item["query_num"])==query_id and int(item["id"])==doc_id):
		# 			actual_rel_scores.append(5-item["position"])
  
		actual_rel_scores = pred_rel_scores
		actual_rel_scores.sort(reverse=True)
		#Calculate IDCG
		idcg=0
		for i in range(len(actual_rel_scores)):
			index=i+1
			rel_index=actual_rel_scores[i]
			temp=rel_index/math.log2(index+1)
			idcg+=temp
			
		ndcg = dcg/idcg if idcg > 0 else 0
  
		# # Calculate IDCG
		# ideal_rel_scores = sorted(pred_rel_scores, reverse=True)
		# idcg = ideal_rel_scores[0] + sum([ideal_rel_scores[i]/math.log2(i+1) for i in range(1, min(k, num_rel_docs))])

		# # Calculate nDCG
		# ndcg = dcg/idcg if idcg > 0 else 0

		return ndcg

	def queryNDCG_LSA(self, query_doc_IDs_ordered, query_id, true_doc_IDs,k,qrels):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query 
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""
		pred_rel_scores=[]
		for doc_id in query_doc_IDs_ordered[:k]:
			# print("\n\ndoc_id=",doc_id)
			for item in qrels:
				if(int(item["query_num"])==query_id and int(item["id"])==doc_id):
					pred_rel_scores.append(5-item["position"])
		# print("\n\npred_rel_scores=",pred_rel_scores)
		dcg=0
		for i in range(len(pred_rel_scores)):
			index=i+1
			rel_index=pred_rel_scores[i]
			temp=rel_index/math.log2(index+1)
			dcg+=temp
   
		#Calculate actual rel scores
		# actual_rel_scores=[]
		# for doc_id in true_doc_IDs[:k]:
		# 	for item in qrels:
		# 		if(int(item["query_num"])==query_id and int(item["id"])==doc_id):
		# 			actual_rel_scores.append(5-item["position"])
  
		actual_rel_scores = pred_rel_scores
		actual_rel_scores.sort(reverse=True)
		#Calculate IDCG
		idcg=0
		for i in range(len(actual_rel_scores)):
			index=i+1
			rel_index=actual_rel_scores[i]
			temp=rel_index/math.log2(index+1)
			idcg+=temp
			
		ndcg = dcg/idcg if idcg > 0 else 0
  
		# # Calculate IDCG
		# ideal_rel_scores = sorted(pred_rel_scores, reverse=True)
		# idcg = ideal_rel_scores[0] + sum([ideal_rel_scores[i]/math.log2(i+1) for i in range(1, min(k, num_rel_docs))])

		# # Calculate nDCG
		# ndcg = dcg/idcg if idcg > 0 else 0

		return ndcg
	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		total_ndcg = 0
		num_queries = len(query_ids)

		for i in range(num_queries):
			query_id = query_ids[i]
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			# true_doc_IDs = [int(d["id"]) for d in qrels if int(d["query_num"])==query_id and int(d["position"])<=2]
			true_doc_IDs = list(set([int(d["id"]) for d in qrels if int(d["query_num"])==query_id and int(d["position"])==1]))
			# true_doc_IDs = [int(d["id"]) for d in qrels if int(d["query_num"])==query_id]
			ndcg = self.queryNDCG(query_doc_IDs_ordered, query_id, true_doc_IDs, k, qrels)
			total_ndcg += ndcg

		mean_ndcg = total_ndcg / num_queries

		return mean_ndcg
	
	def meanNDCG_LSA(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		total_ndcg = 0
		num_queries = len(query_ids)

		for i in range(num_queries):
			query_id = query_ids[i]
			query_doc_IDs_ordered = doc_IDs_ordered[i]
			# true_doc_IDs = [int(d["id"]) for d in qrels if int(d["query_num"])==query_id and int(d["position"])<=2]
			true_doc_IDs = list(set([int(d["id"]) for d in qrels if int(d["query_num"])==query_id and int(d["position"])==1]))
			# true_doc_IDs = [int(d["id"]) for d in qrels if int(d["query_num"])==query_id]
			ndcg = self.queryNDCG_LSA(query_doc_IDs_ordered, query_id, true_doc_IDs, k, qrels)
			total_ndcg += ndcg

		mean_ndcg = total_ndcg / num_queries

		return mean_ndcg


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		count_relevant = 0
		sum_precision = 0

		for i, doc_id in enumerate(query_doc_IDs_ordered[0][:k]):
			if doc_id in true_doc_IDs:
				count_relevant += 1
				sum_precision += count_relevant / (i + 1)

		if count_relevant == 0:
        
			return 0

		avgPrecision = sum_precision / count_relevant

		return avgPrecision

	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1
		temp=0.0
		for i in range(len(query_ids)):
			#true_doc_IDs = [int(d["id"]) for d in q_rels if int(d["query_num"])==query_ids[i] and int(d["position"])<=2]
			#true_doc_IDs = [int(d["id"]) for d in q_rels if int(d["query_num"])==query_ids[i] and int(d["position"])>2]
			true_doc_IDs = [int(d["id"]) for d in q_rels if int(d["query_num"])==query_ids[i] ]
			# true_doc_IDs = list(set([int(d["id"]) for d in q_rels if int(d["query_num"])==query_ids[i] and int(d["position"])==1]))
			temp+=self.queryAveragePrecision(doc_IDs_ordered,query_ids[i],true_doc_IDs,k)
		meanAveragePrecision=temp/(len(query_ids))
		return 0*meanAveragePrecision

# x=[{"query_num": "1", "position": 2, "id": "3"},
# {"query_num": "1", "position": 3, "id": "5"},
# {"query_num": "1", "position": 2, "id": "2"},
# {"query_num": "1", "position": 5, "id": "4"},
# {"query_num": "1", "position": 4, "id": "9"},
# {"query_num": "1", "position": 3, "id": "11"}]

# y=Evaluation()
# print(y.queryNDCG([3,5,2,4,9,11], 1, [],6,x))