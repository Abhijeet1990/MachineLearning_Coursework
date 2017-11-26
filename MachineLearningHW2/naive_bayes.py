import os,sys
import numpy as np
from operator import itemgetter
import csv
import random
import math
import operator



#first find the conditional propability of feature vector given the digit of an image from the training data
#feature vector means the value hold by each co-ordinates in the text file.
#each text file has 32 * 32 matrix holding each value
#P(F0,0 = 1 | image = 0)....P(F0,1=1| image =0).....P(F32,32=1|image=0)...this value becomes zero soon....
#similarly find for all the images 0...to ....9

#if we consider all the pixel as the feature vector then the conditional probability of these feature vectors 
#given the picture get reduced profusely...P(Feature vector|no is 0) = P(f(0,0)|o is 0)*

def main():

    

    fd1=os.open("/home/bo/Desktop/MachineLearningHW2/testDigits",os.O_RDONLY)
    os.fchdir(fd1)
    total=0
    success=0
    

    for filename1 in os.listdir(os.getcwd() ):
	f1=open(filename1,'r')
        content1=f1.read()
	fd=os.open("/home/bo/Desktop/MachineLearningHW2/trainingDigits",os.O_RDONLY)
	os.fchdir(fd)
	total=total+1
        accum0=np.zeros(1088) # 32 * 32
    	accum1=np.zeros(1088) # 32 * 32
    	accum2=np.zeros(1088) # 32 * 32
    	accum3=np.zeros(1088) # 32 * 32
    	accum4=np.zeros(1088) # 32 * 32
    	accum5=np.zeros(1088) # 32 * 32
    	accum6=np.zeros(1088) # 32 * 32
    	accum7=np.zeros(1088) # 32 * 32
    	accum8=np.zeros(1088) # 32 * 32
    	accum9=np.zeros(1088) # 32 * 32

    	Prob_cont_0=np.zeros(1088) # 32 * 32
    	Prob_cont_1=np.zeros(1088) # 32 * 32
    	Prob_cont_2=np.zeros(1088) # 32 * 32
    	Prob_cont_3=np.zeros(1088) # 32 * 32
    	Prob_cont_4=np.zeros(1088) # 32 * 32
    	Prob_cont_5=np.zeros(1088) # 32 * 32
    	Prob_cont_6=np.zeros(1088) # 32 * 32
    	Prob_cont_7=np.zeros(1088) # 32 * 32
    	Prob_cont_8=np.zeros(1088) # 32 * 32
    	Prob_cont_9=np.zeros(1088) # 32 * 32

        count_0=0
    	count_1=0
    	count_2=0
    	count_3=0
    	count_4=0
    	count_5=0
    	count_6=0
    	count_7=0
    	count_8=0
    	count_9=0
	
        for filename in os.listdir(os.getcwd() ):
		
		if filename[0]=='0':
			count_0=count_0+1
			#count[0] is the count of number of training test file with 0
			f = open(filename, 'r')
			content = f.read()
			for i in range(0,1087):
			     	if (((i+2)%34)!=0) and (((i+2)%34)!=1) :
					#print content[i]
					#print "hey wassup %d "%i
			     		accum0[i]=accum0[i]+int(content[i])
					#Prob_cont[i] is the prob of that feature being 1, given the picture is 7
			     		Prob_cont_0[i]=float(accum0[i])/float(count_0)
		if filename[0]=='1':
			count_1=count_1+1
			#count[0] is the count of number of training test file with 0
			f = open(filename, 'r')
			content = f.read()
			for i in range(0,1087):
			     	if (((i+2)%34)!=0) and (((i+2)%34)!=1) :
					#print content[i]
					#print "hey wassup %d "%i
			     		accum1[i]=accum1[i]+int(content[i])
					#Prob_cont[i] is the prob of that feature being 1, given the picture is 7
			     		Prob_cont_1[i]=float(accum1[i])/float(count_1)
		if filename[0]=='2':
			count_2=count_2+1
			#count[0] is the count of number of training test file with 0
			f = open(filename, 'r')
			content = f.read()
			for i in range(0,1087):
			     	if (((i+2)%34)!=0) and (((i+2)%34)!=1) :
					#print content[i]
					#print "hey wassup %d "%i
			     		accum2[i]=accum2[i]+int(content[i])
					#Prob_cont[i] is the prob of that feature being 1, given the picture is 7
			     		Prob_cont_2[i]=float(accum2[i])/float(count_2)
		if filename[0]=='3':
			count_3=count_3+1
			#count[0] is the count of number of training test file with 0
			f = open(filename, 'r')
			content = f.read()
			for i in range(0,1087):
			     	if (((i+2)%34)!=0) and (((i+2)%34)!=1) :
					#print content[i]
					#print "hey wassup %d "%i
			     		accum3[i]=accum3[i]+int(content[i])
					#Prob_cont[i] is the prob of that feature being 1, given the picture is 7
			     		Prob_cont_3[i]=float(accum3[i])/float(count_3)
		if filename[0]=='4':
			count_4=count_4+1
			#count[0] is the count of number of training test file with 0
			f = open(filename, 'r')
			content = f.read()
			for i in range(0,1087):
			     	if (((i+2)%34)!=0) and (((i+2)%34)!=1) :
					#print content[i]
					#print "hey wassup %d "%i
			     		accum4[i]=accum4[i]+int(content[i])
					#Prob_cont[i] is the prob of that feature being 1, given the picture is 7
			     		Prob_cont_4[i]=float(accum4[i])/float(count_4)
		if filename[0]=='5':
			count_5=count_5+1
			#count[0] is the count of number of training test file with 0
			f = open(filename, 'r')
			content = f.read()
			for i in range(0,1087):
			     	if (((i+2)%34)!=0) and (((i+2)%34)!=1) :
					#print content[i]
					#print "hey wassup %d "%i
			     		accum5[i]=accum5[i]+int(content[i])
					#Prob_cont[i] is the prob of that feature being 1, given the picture is 7
			     		Prob_cont_5[i]=float(accum5[i])/float(count_5)
		if filename[0]=='6':
			count_6=count_6+1
			#count[0] is the count of number of training test file with 0
			f = open(filename, 'r')
			content = f.read()
			for i in range(0,1087):
			     	if (((i+2)%34)!=0) and (((i+2)%34)!=1) :
					#print content[i]
					#print "hey wassup %d "%i
			     		accum6[i]=accum6[i]+int(content[i])
					#Prob_cont[i] is the prob of that feature being 1, given the picture is 7
			     		Prob_cont_6[i]=float(accum6[i])/float(count_6)
		if filename[0]=='7':
			count_7=count_7+1
			#count[0] is the count of number of training test file with 0
			f = open(filename, 'r')
			content = f.read()
			#print"count of 7 %d"%count_7
			for i in range(0,1087):
				if (((i+2)%34)!=0) and (((i+2)%34)!=1) :
					#print content[i]
					#print "hey wassup %d "%i
			     		accum7[i]=accum7[i]+int(content[i])
					#Prob_cont[i] is the prob of that feature being 1, given the picture is 7
			     		Prob_cont_7[i]=float(accum7[i])/float(count_7)
					#print"accum %d for i %d count %d"%(accum7[i],i,count_7)
					#print "prob %f i %d  count %d"%(Prob_cont_7[i],i,count_7)
		if filename[0]=='8':
			count_8=count_8+1
			#count[0] is the count of number of training test file with 0
			f = open(filename, 'r')
			content = f.read()
			for i in range(0,1087):
			     	if (((i+2)%34)!=0) and (((i+2)%34)!=1) :
					#print content[i]
					#print "hey wassup %d "%i
			     		accum8[i]=accum8[i]+int(content[i])
					#Prob_cont[i] is the prob of that feature being 1, given the picture is 7
			     		Prob_cont_8[i]=float(accum8[i])/float(count_8)
		if filename[0]=='9':
			count_9=count_9+1
			#count[0] is the count of number of training test file with 0
			f = open(filename, 'r')
			content = f.read()
			for i in range(0,1087):
			     	if (((i+2)%34)!=0) and (((i+2)%34)!=1) :
					#print content[i]
					#print "hey wassup %d "%i
			     		accum9[i]=accum9[i]+int(content[i])
					#Prob_cont[i] is the prob of that feature being 1, given the picture is 7
			     		Prob_cont_9[i]=float(accum9[i])/float(count_9)
	
        Post_0=0
	Post_1=0
	Post_2=0
	Post_3=0
	Post_4=0
	Post_5=0
	Post_6=0
	Post_7=0
	Post_8=0
	Post_9=0
	
	for m in range(0,1087):
		if content1[m]=='1':
		    print 'Prob 7 %.20f x i %d'%(Post_7,m)
		    if Prob_cont_0[m]!=0:
			Post_0 = Post_0 + math.log10(Prob_cont_0[m])
 		    if Prob_cont_1[m]!=0:
			Post_1 = Post_1 + math.log10(Prob_cont_1[m])
		    if Prob_cont_2[m]!=0:
			Post_2 = Post_2 + math.log10(Prob_cont_2[m])
		    if Prob_cont_3[m]!=0:
			Post_3 = Post_3 + math.log10(Prob_cont_3[m])
		    if Prob_cont_4[m]!=0:
			Post_4 = Post_4 + math.log10(Prob_cont_4[m])
		    if Prob_cont_5[m]!=0:
			Post_5 = Post_5 + math.log10(Prob_cont_5[m])
		    if Prob_cont_6[m]!=0:
			Post_6 = Post_6 + math.log10(Prob_cont_6[m])
		    if Prob_cont_7[m]!=0:
			Post_7 = Post_7 + math.log10(Prob_cont_7[m])
		    if Prob_cont_8[m]!=0:
			Post_8 = Post_8 + math.log10(Prob_cont_8[m])
		    if Prob_cont_9[m]!=0:
			Post_9 = Post_9 + math.log10(Prob_cont_9[m])
			
		elif content1[m]=='0':
		    print 'Prob 7 %.20f x i %d'%(Post_7,m)
	            print 'prob %.20f'%Prob_cont_6[m]
		    if Prob_cont_0[m]!=1: 
			Post_0 = Post_0 + math.log10(1-Prob_cont_0[m])
		    if Prob_cont_1[m]!=1: 
			Post_1 = Post_1 + math.log10(1-Prob_cont_1[m])
		    if Prob_cont_2[m]!=1:
			Post_2 = Post_2 + math.log10(1-Prob_cont_2[m])
		    if Prob_cont_3[m]!=1:
			Post_3 = Post_3 + math.log10(1-Prob_cont_3[m])
		    if Prob_cont_4[m]!=1:
			Post_4 = Post_4 + math.log10(1-Prob_cont_4[m])
		    if Prob_cont_5[m]!=1:
			Post_5 = Post_5 + math.log10(1-Prob_cont_5[m])
		    if Prob_cont_6[m]!=1:
			Post_6 = Post_6 + math.log10(1-Prob_cont_6[m])
		    if Prob_cont_7[m]!=1:
			Post_7 = Post_7 + math.log10(1-Prob_cont_7[m])
		    if Prob_cont_8[m]!=1:
			Post_8 = Post_8 + math.log10(1-Prob_cont_8[m])
		    if Prob_cont_9[m]!=1:
			Post_9 = Post_9 + math.log10(1-Prob_cont_9[m])
	print 'Prob 0 %.10f'%Post_0
	print 'Prob 1 %.10f'%Post_1
	print 'Prob 2 %.10f'%Post_2
	print 'Prob 3 %.10f'%Post_3
	print 'Prob 4 %.10f'%Post_4
	print 'Prob 5 %.10f'%Post_5
	print 'Prob 6 %.10f'%Post_6
	print 'Prob 7 %.10f'%Post_7
	print 'Prob 8 %.10f'%Post_8
	print 'Prob 9 %.10f'%Post_9
	

        if (max(Post_0,Post_1,Post_2,Post_3,Post_4,Post_5,Post_6,Post_7,Post_8,Post_9)==Post_0 and filename1[0]=='0'):
		print('> predicted=' + repr(0) + ', actual=' + repr(filename1[0]))
		success=success+1
        if (max(Post_0,Post_1,Post_2,Post_3,Post_4,Post_5,Post_6,Post_7,Post_8,Post_9)==Post_1 and filename1[0]=='1'):
		print('> predicted=' + repr(1) + ', actual=' + repr(filename1[0]))
		success=success+1
        if (max(Post_0,Post_1,Post_2,Post_3,Post_4,Post_5,Post_6,Post_7,Post_8,Post_9)==Post_2 and filename1[0]=='2'):
		print('> predicted=' + repr(2) + ', actual=' + repr(filename1[0]))
		success=success+1
        if (max(Post_0,Post_1,Post_2,Post_3,Post_4,Post_5,Post_6,Post_7,Post_8,Post_9)==Post_3 and filename1[0]=='3'):
		print('> predicted=' + repr(3) + ', actual=' + repr(filename1[0]))
		success=success+1
        if (max(Post_0,Post_1,Post_2,Post_3,Post_4,Post_5,Post_6,Post_7,Post_8,Post_9)==Post_4 and filename1[0]=='4'):
		print('> predicted=' + repr(4) + ', actual=' + repr(filename1[0]))
		success=success+1
        if (max(Post_0,Post_1,Post_2,Post_3,Post_4,Post_5,Post_6,Post_7,Post_8,Post_9)==Post_5 and filename1[0]=='5'):
		print('> predicted=' + repr(5) + ', actual=' + repr(filename1[0]))
		success=success+1
        if (max(Post_0,Post_1,Post_2,Post_3,Post_4,Post_5,Post_6,Post_7,Post_8,Post_9)==Post_6 and filename1[0]=='6'):
		print('> predicted=' + repr(6) + ', actual=' + repr(filename1[0]))
		success=success+1
        if (max(Post_0,Post_1,Post_2,Post_3,Post_4,Post_5,Post_6,Post_7,Post_8,Post_9)==Post_7 and filename1[0]=='7'):
		print('> predicted=' + repr(7) + ', actual=' + repr(filename1[0]))
		success=success+1
        if (max(Post_0,Post_1,Post_2,Post_3,Post_4,Post_5,Post_6,Post_7,Post_8,Post_9)==Post_8 and filename1[0]=='8'):
		print('> predicted=' + repr(8) + ', actual=' + repr(filename1[0]))
		success=success+1
        if (max(Post_0,Post_1,Post_2,Post_3,Post_4,Post_5,Post_6,Post_7,Post_8,Post_9)==Post_9 and filename1[0]=='9'):
		print('> predicted=' + repr(9) + ', actual=' + repr(filename1[0]))
		success=success+1
    
    accuracy=float(success)/float(total)
    print " Accuracy %f"%accuracy

	
main()
	
		
			


			


