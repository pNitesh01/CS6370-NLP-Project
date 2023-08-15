from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from InformationRetrieval_LSA import InformationRetrieval_LSA
from informationRetrieval_VSM import InformationRetrieval_VSM
from evaluation import Evaluation
from scipy.stats import ttest_ind
import numpy as np
from scipy import stats

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt
from scipy.stats import t

import time

print("NDCG Plot")


# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		self.informationRetriever_LSA = InformationRetrieval_LSA()
		self.informationRetriever_VSM = InformationRetrieval_VSM()
		self.evaluator = Evaluation()

	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		#print("\n\nSegmentedDocs=",segmentedDocs)
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		#print("\n\tokenizedDocs=",tokenizedDocs)
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		#print("\n\nreducedDocs=",reducedDocs)
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))
		#print("\n\nstopwordRemovedDocs=",stopwordRemovedDocs)
		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs


	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)
		#print(processedQueries)
		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		print("\n\nlen(docs)",len(docs))
		print("\n\nlen(doc_ids)",len(doc_ids))
		# Process documents
		processedDocs = self.preprocessDocs(docs)
		#print(processedDocs)
		# Build document index
		start_rank=450
		end_rank=650
		diff=50
		for r in range(start_rank,end_rank,diff):
			start_time = time.time()
			print("\n\nStarting LSA Index Building")
			self.informationRetriever_LSA.buildIndex(processedDocs, doc_ids,r)
			print("\n\nFinished.")
			print("\n\nStarting LSA Rank function")
			doc_IDs_ordered_LSA = self.informationRetriever_LSA.rank(processedQueries)
			print("\n\nFinished.")
			end_time = time.time()
			running_time = end_time - start_time
			print("\n\nLSA running time=",running_time)
			start_time = time.time()
			print("\n\nStarting VSM Index Building")
			self.informationRetriever_VSM.buildIndex(processedDocs, doc_ids)
			print("\n\nFinished.")
			print("\n\nStarting VSM Rank function")
			doc_IDs_ordered_VSM = self.informationRetriever_VSM.rank(processedQueries)
			print("\n\nFinished.")
			end_time = time.time()
			running_time = end_time - start_time
			print("\n\nVSM running time=",running_time)
			# Read relevance judements
			qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

			# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
			precisions_LSA, recalls_LSA, fscores_LSA, MAPs_LSA, nDCGs_LSA = [], [], [], [], []
			precisions_VSM, recalls_VSM, fscores_VSM, MAPs_VSM, nDCGs_VSM = [], [], [], [], []
			
			for k in range(1, 11):
			#for k in range(1, 4):
				precision_LSA = self.evaluator.meanPrecision(
					doc_IDs_ordered_LSA, query_ids, qrels, k)
				precisions_LSA.append(precision_LSA)
				recall_LSA = self.evaluator.meanRecall(
					doc_IDs_ordered_LSA, query_ids, qrels, k)
				recalls_LSA.append(recall_LSA)
				fscore_LSA = self.evaluator.meanFscore(
					doc_IDs_ordered_LSA, query_ids, qrels, k)
				fscores_LSA.append(fscore_LSA)
				# print("F-score @ " +  
				# 	str(k) + " : " + str(fscore_LSA))
				print("Precision_LSA, Recall_LSA and F-score_LSA @ " +  
					str(k) + " : " + str(precision_LSA) + ", " + str(recall_LSA) + 
					", " + str(fscore_LSA))
				MAP_LSA = self.evaluator.meanAveragePrecision(
					doc_IDs_ordered_LSA, query_ids, qrels, k)
				MAPs_LSA.append(MAP_LSA)
				nDCG_LSA = self.evaluator.meanNDCG_LSA(
					doc_IDs_ordered_LSA, query_ids, qrels, k)
				nDCGs_LSA.append(nDCG_LSA)
				print("MAP_LSA, nDCG_LSA @ " +  
					str(k) + " : " + str(MAP_LSA) + ", " + str(nDCG_LSA))

			for k in range(1, 11):
			#for k in range(1, 4):
				plt.clf()
				precision_VSM = self.evaluator.meanPrecision(
					doc_IDs_ordered_VSM, query_ids, qrels, k)
				precisions_VSM.append(precision_VSM)
				recall_VSM = self.evaluator.meanRecall(
					doc_IDs_ordered_VSM, query_ids, qrels, k)
				recalls_VSM.append(recall_VSM)
				fscore_VSM = self.evaluator.meanFscore(
					doc_IDs_ordered_VSM, query_ids, qrels, k)
				fscores_VSM.append(fscore_VSM)
				# print("F-score @ " +  
				# 	str(k) + " : " + str(fscore_VSM))
				print("Precision_VSM, Recall_VSM and F-score_VSM @ " +  
					str(k) + " : " + str(precision_VSM) + ", " + str(recall_VSM) + 
					", " + str(fscore_VSM))
				MAP_VSM = self.evaluator.meanAveragePrecision(
					doc_IDs_ordered_VSM, query_ids, qrels, k)
				MAPs_VSM.append(MAP_VSM)
				nDCG_VSM = self.evaluator.meanNDCG(
					doc_IDs_ordered_VSM, query_ids, qrels, k)
				nDCGs_VSM.append(nDCG_VSM)
				print("MAP_VSM, nDCG_VSM @ " +  
					str(k) + " : " + str(MAP_VSM) + ", " + str(nDCG_VSM))
			print("Precision diff= ",precisions_LSA[0]-precisions_VSM[0])
			if(precisions_LSA[0]-precisions_VSM[0]>0):
				print("at rank=",r," precision_diff=",precisions_LSA[0]-precisions_VSM[0])
			plt.ylim(0.2, 1.0)  # set the y-axis limits
			plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
			#Plot the metrics and save plot 
			# plt.plot(range(1, 11), precisions_LSA,color="blue",linestyle='-', label="Precision_LSA")
			# plt.plot(range(1, 11), precisions_VSM,color="blue",linestyle=':', label="Precision_VSM")
	
			# plt.plot(range(1, 11), recalls_LSA, color="red",linestyle='-',label="Recall_LSA")
			# plt.plot(range(1, 11), recalls_VSM, color="red",linestyle=':',label="Recall_VSM")
	
			# plt.plot(range(1, 11), fscores_LSA, color="green",linestyle='-',label="F-Score_LSA")
			# plt.plot(range(1, 11), fscores_VSM, color="green",linestyle=':',label="F-Score_VSM")
	
			# plt.plot(range(1, 11), MAPs_LSA, color="black",linestyle='-',label="MAP_LSA")
			# plt.plot(range(1, 11), MAPs_VSM, color="black",linestyle=':',label="MAP_VSM")
	
			plt.plot(range(1, 11), nDCGs_LSA, color="purple",linestyle='-',label="nDCG_LSA")
			plt.plot(range(1, 11), nDCGs_VSM, color="purple",linestyle=':',label="nDCG_VSM")
			# print("\n\nnDCGs_VSM=",nDCGs_VSM)
			# print("\n\nnDCGs_LSA=",nDCGs_LSA)
			# print('\n\nprecisions_LSA=',precisions_LSA)
			# print('\n\nprecisions_VSM=',precisions_VSM)
	
			# print('\n\nrecalls_LSA=',recalls_LSA)
			# print('\n\nrecalls_VSM=',recalls_VSM)
	
			# print('\n\nfscores_LSA=',fscores_LSA)
			# print('\n\nfscores_VSM=',fscores_VSM)
	
			# print('\n\nMAPs_LSA=',MAPs_LSA)
			# print('\n\nMAPs_VSM=',MAPs_VSM)
	
			# print('\n\nDCGs_LSA=',nDCGs_LSA)
			# print('\n\nDCGs_VSM=',nDCGs_VSM)

			

			# f_score_t, f_score_p = ttest_ind(fscores_VSM, fscores_LSA)
			# print('t-statistic for F-score:', f_score_t)
			# print('p-value for F-score:', f_score_p)

			# # Calculate the mean and standard deviation of the precision values for LSA and VSM
			# mean_precisions_LSA = np.mean(np.array(precisions_LSA))
			# std_precisions_LSA = np.std(np.array(precisions_LSA))
			# mean_precisions_VSM = np.mean(np.array(precisions_VSM))
			# std_precisions_VSM = np.std(np.array(precisions_VSM))

			# # Define the significance level
			# alpha = 0.05

			# # Calculate the degrees of freedom
			# n1 = len(precisions_LSA)
			# n2 = len(precisions_VSM)
			# df = n1 + n2 - 2

			# # Calculate the pooled standard deviation
			# sp = np.sqrt(((n1 - 1) * std_precisions_LSA ** 2 + (n2 - 1) * std_precisions_VSM ** 2) / df)

			# # Calculate the t-statistic
			# t = (mean_precisions_LSA - mean_precisions_VSM) / (sp * np.sqrt(1 / n1 + 1 / n2))

			# # Calculate the p-value
			# p = 2 * (1 - stats.t.cdf(abs(t), df))

			# # Print the results
			# print(f"t-statistic for precision: {t}")
			# print(f"p-value for precision: {p}")

			# # Compare the p-value with the significance level to determine if the null hypothesis can be rejected
			# if p < alpha:
			# 	print("The difference in precision between LSA and VSM is statistically significant")
			# else:
			# 	print("The difference in precision between LSA and VSM is not statistically significant")
			
			mean_vsm = np.mean(precisions_VSM)
			mean_lsa = np.mean(precisions_LSA)
			std_vsm = np.std(precisions_VSM, ddof=1)
			std_lsa = np.std(precisions_LSA, ddof=1)
			n_vsm = len(precisions_VSM)
			n_lsa = len(precisions_LSA)

			s_p = np.sqrt(((n_vsm-1)*std_vsm**2 + (n_lsa-1)*std_lsa**2)/(n_vsm + n_lsa - 2))
			t_statistic = (mean_lsa - mean_vsm) / (s_p*np.sqrt(1/n_vsm + 1/n_lsa))
			p_value = t.cdf(t_statistic, n_vsm + n_lsa - 2) * 2  # two-tailed test

			# print results
			print("t-statistic: ", t_statistic)
			print("p-value: ", p_value)
			
			
			
			plt.legend()
			plt.title("Evaluation Metrics - Cranfield Dataset")
			plt.xlabel("k")
			plt.savefig(args.out_folder + "eval_plot_at_rank_"+str(r)+".png")
		#plt.show()

		
	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		print("Enter query below")
		query = input()
		# Process documents
		processedQuery = self.preprocessQueries([query])[0]

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		# Print the IDs of first five documents
		print("\nTop five document IDs : ")
		for id_ in doc_IDs_ordered[:5]:
			print(id_)



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
  #shipment of gold damaged in a fire.
  #Delivery of silver arrived in a silver truck.
  #Shipment of gold arrived in a truck.
  
  #what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .

#Herbivores typically plant eaters meat eaters
#Carnivores typically meat eaters plant eaters
#Deers eat grass leaves