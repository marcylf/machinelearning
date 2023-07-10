import sys
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1 : Prepare data
corpus = [
    'We enjoyed our stay so much. The weather was not great, but everyething else was perfect.',
    'Going to think twice before taying here again. The wifi was spotty and the rooms smaller than adversited', 
    'Not that mindblowing',
    'Had worse experiences.',
    'Great place for a quick city gateway.',
    'The pictures looked way different.',
    'I do not like this hotel.',
    'I like this hotel.',
    'This hotel is bad.',
    'This hotel is good.'
]

# Sentimant data : 0 is negative sentiment, 1: positive sentiment
targets = [1,0,0,1,1,0,0,1,0,1]

#Splitting the dataset
train_features, test_features, train_targets, test_targets =  train_test_split(corpus, targets, test_size=0.1, random_state=123)

#Turning the corups into a tf=idf array
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, norm='l1')

train_features = vectorizer.fit_transform(train_features)
test_features = vectorizer.transform(test_features)


#Build the perceptron and fit the data
classifier = Perceptron(random_state=457)
classifier.fit(train_features, train_targets)

predictions = classifier.predict(test_features)
score = np.round(metrics.accuracy_score(test_targets,predictions),2)

print("Mean accuracy of predictions: " + str(score))

#Step 5: Make predicition

while True:
    user_text = input("Enter your review (Press 'e' to exit): ")
    new_sentence_features = vectorizer.transform([user_text])
    prediction = classifier.predict(new_sentence_features)
    print(prediction)

    if user_text.lower() == 'e':
        print("Exiting the program...")
        sys.exit()

    if prediction > 0.5:
        print('Good review')
    else:
        print('Bad review')





