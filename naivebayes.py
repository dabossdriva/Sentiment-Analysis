"""
Odessa Elie
May 1, 2020
Naive Bayes Classifier

The purpose of this project is to implement the Naive Bayes Classifier on text data.
"""

#importing packages 
import numpy as np
import pandas as pd
from sklearn import preprocessing
import operator
import nltk
#nltk.download()
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix

#take the user's input of the dataset and label file and print some samples
filename = input("Enter filename: ")
labelfile = input("Enter label filename: ")
classifier1 = input("Enter the first classification label: ")
classifier2 = input("Enter the second classification label: ")

get_data = pd.read_table(filename,sep=',', header=0, names=['Category', 'Message'], encoding= 'unicode_escape')

# Printing the dataset details  
print ("\nDataset Details (instances, attributes): ", get_data.shape) 
	
# Printing the first 5 instances of the dataset 
print ("\nDataset sample: \n",get_data.head())

#Training label
training_label = open(filename)
    
#Extract values from training labels
lines = training_label.readlines()

for longlines in open(filename,"r"):  
    longlines = longlines.strip()  
    lines = longlines.split(";")

#Get numlines number of lines
numlines = len(lines)	
 
d = dict() 
  
# Loop through each line of the file, remove the leading spaces and newline characters and split the line into words
for line in open(labelfile,"r"):  
    line = line.strip()  
    words = line.split(",") 
  
    # Iterate over each word in line to check if it is in the dictionary. If it is there increment and if it is not insert it with a count of 1
    for word in words:  
        if word in d:  
            d[word] += 1
        else: 
            d[word] = 1

#calculate the probability of the classification 
for key in d:
    d[key]/= numlines
    
print("\nProbability of each classification:")
print("\n".join("{}: {}".format(k, v) for k,v in d.items()))

#convert ham and spam to binary
get_data['Category'] = get_data.Category.map({classifier1: 0, classifier2: 1})

get_data['Message'] = get_data['Message'].apply(nltk.word_tokenize)

stemmer = PorterStemmer()

stemfunction = lambda x: [stemmer.stem(y) for y in x]
 
get_data['Message'] = get_data['Message'].apply(stemfunction)

# This converts the list of words into strings separated by spaces
get_data['Message'] = get_data['Message'].apply(lambda x: ' '.join(x))

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(get_data['Message'])

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)

#split into training and testing (60-40 split)
X_train, X_test, y_train, y_test = train_test_split(counts, get_data['Category'], test_size = 0.4, random_state = 50)

NBmodel = MultinomialNB().fit(X_train, y_train)

prediction = NBmodel.predict(X_test)

print("\nAccuracy: ",np.mean(prediction == y_test)*100, "%")

print("\nConfusion Matrix: \n",confusion_matrix(y_test, prediction))
