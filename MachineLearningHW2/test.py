import os,sys
import numpy as np
from operator import itemgetter
import csv
import random
import math
import operator


def getNeighbors(name_dist, k):
	neighbors = []
	for x in range(k):
		neighbors.append(name_dist[x][0])
	return neighbors

def getResponse(neighbors,k):
	classVotes = {}
	for x in range(k):
		response = neighbors[x][0]
		#print"class %s"%response
	if response in classVotes:
			classVotes[response] += 1
	else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(char, predictions):
	correct = 0
	for x in range(len(predictions)):
		if predictions[x]==char:
			correct += 1
	return (correct/len(predictions)) * 100.0

	
def main():

  for k in range(1,10):
    
    fd1=os.open("/home/bo/Desktop/MachineLearningHW2/testDigits",os.O_RDONLY)
    os.fchdir(fd1)
    for filename1 in os.listdir(os.getcwd() ):
	pairwise_dist=[]
	training_data=[]
	name_dist=[]
	distance=[]
	name=[]
	predictions=[]
        f1=open(filename1,'r')
        content1=f1.read()
	fd=os.open("/home/bo/Desktop/MachineLearningHW2/trainingDigits",os.O_RDONLY)
	os.fchdir(fd)
	count=0
	
	for filename in os.listdir(os.getcwd() ):
		
    		if filename.endswith(".txt"):
        		f = open(filename, 'r')
			content = f.read()
			dist=0
        		count=count+1
                        u=zip(content,content1)
			for i,j in u: 
    				if i!=j:
        				dist+=1
				else:
					dist+=0
			training_data.append(content)	 
    			pairwise_dist.append(dist)
			name_dist.append((filename,dist))
			
	name_dist.sort(key=itemgetter(1))
	neighbors=getNeighbors(name_dist,k)
	#print neighbors
	result=getResponse(neighbors,k)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(filename1[0]))
	os.close(fd)
    accuracy = getAccuracy(filename1[0], predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    os.close(fd1)


main()






