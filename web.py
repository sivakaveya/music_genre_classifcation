from atexit import register
from email import feedparser
from urllib.robotparser import RequestRate
from flask import render_template, Flask,request
from werkzeug.utils import secure_filename
from io import BytesIO
from datetime import datetime
import os

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile
from scipy.io import wavfile
import os
import pickle
import random
import operator
import wavfile
import math

from collections import defaultdict
# from google.colab import files




app=Flask(__name__,template_folder='templates',static_folder='static')
app.secret_key='ikstao'
APP_ROOT=os.path.dirname(os.path.abspath(__file__))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def create_path(filename):
    target = os.path.join(APP_ROOT,'static//images/')
    location = "events/".join([target, filename])
    return location



@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='POST':
        music = request.files['music']
        if music:
            filename = secure_filename(music.filename)
            print(filename)
            target = os.path.join(APP_ROOT,'static/')
            destination = "music/".join([target, filename])
            music.save(destination)
            print(destination)
            (rate, sig) = wav.read(destination)
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, 10)
            
            results = defaultdict(int)

            directory = r"C://Users//91922//Documents//19102A0058//BE//ML//ML Project//genres_original//"

            i = 1
            for folder in os.listdir(directory):
                results[i] = folder
                i += 1
            pred = nearestClass(getNeighbors(dataset, feature, 5))
            print(results[pred])  
            return render_template('stats.html',genre=results[pred])
    return render_template('index.html')

########EXTRA FUNCTIONS#######
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    
    return neighbors

def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)

    return sorter[0][0]

def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    
    return (1.0 * correct) / len(testSet)

# directory that holds the wav files
# directory = r"C://Users//vedan//genres//"
directory = r"C://Users//91922//Documents//19102A0058//BE//ML//ML Project//genres_original//"
# binary file where we will collect all the features extracted using mfcc (Mel Frequency Cepstral Coefficients)
f = open("my.dat", 'wb')

i = 0

for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+folder):        
        try:
            (rate, sig) = wav.read(directory+folder+"/"+file)
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)
        except Exception as e:
            print('Got an exception: ', e, ' in folder: ', folder, ' filename: ', file)        

f.close()

# Split the dataset into training and testing sets respectively
dataset = []

def loadDataset(filename, split, trSet, teSet):
    with open('my.dat', 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])
trainingSet = []
testSet = []
loadDataset('my.dat', 0.66, trainingSet, testSet)


def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance

# making predictions using KNN
leng = len(testSet)
predictions = []
for x in range(leng):
    predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))



