import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import random
import tensorflow as tf
import tflearn
import json
import sys
import pickle
from watson_developer_cloud import ConversationV1
import pyttsx3


def initVoice(glos):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('voice', voices[glos].id)
    engine.setProperty('rate', rate - 20)
    return engine

def say(text, engine):
    engine.say(text)
    engine.runAndWait()

lemmatizer = WordNetLemmatizer()

def importJson(file):
    with open(file)as json_data:
        intents = json.loads(json_data.read())
    return intents

def preprocessData(intents):
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!']
    for cont in intents['context']:
        for pattern in cont['pattern']:
            w = word_tokenize(pattern)
            words.extend(w)
            documents.append((w,cont['tag']))
            if cont['tag'] not in classes:
                classes.append(cont['tag'])
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    print(len(documents), "documents")
    print(len(classes), "classes")
    print(len(words), "words")
    return words, classes, documents


def transformData(words, classes, documents):
    training = []
    output = []
    out_empty = [0]*len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        for w in words:
            if w in pattern_words:
                bag.append(1)
            else:
                bag.append(0)
        output_row = list(out_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    random.shuffle(training)
    training = np.array(training)
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    return train_x, train_y
    
def buildNetworkModel(train_x, train_y):
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 15)
    net = tflearn.fully_connected(net, 15)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)
    mod = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    return mod

def startTrain(model, train_x, train_y):
    model.fit(train_x, train_y, n_epoch=1500, batch_size=10, show_metric=True)
    model.save('model.tflearn')
    return None


def saveAll(words, classes, train_x, train_y):
    pickle.dump({'words':words, 'classes':classes, 'train_x':train_x,
                'train_y':train_y}, open('training_data3','wb'))

def restoreParams(pickleFile, jsonFile):
    data = pickle.load(open(pickleFile, 'rb'))
    words = data['words']
    classes = data['classes']
    train_x = data['train_x']
    train_y = data['train_y']
    intents = importJson(jsonFile)
    return words, classes, train_x, train_y, intents
 
def loadModel(mFile, train_x, train_y):
    model = buildNetworkModel(train_x, train_y)
    model.load(mFile, weights_only=True)
    return model

 
def cleanUserInput(sentence):
    sent_words = word_tokenize(sentence)
    sent_words = [lemmatizer.lemmatize(w.lower()) for w in sent_words]
    return sent_words

def bagUser(sentence, words, show_details=False):
    sent_words = cleanUserInput(sentence)
    bag = [0]*len(words)
    for s in sent_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print('Found in bag: {}'.format(w))
    return(np.array(bag))

def classify(sentence, model, words, classes, intents, error):
    results = model.predict([bagUser(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>error]
    results.sort(key=lambda x: x[1], reverse=True)
    final_res = []
    odp = "Sorry I don't understand"
    for r in results:
        final_res.append((classes[r[0]], r[1]))
    if final_res:
        for i in intents['context']:
            if i['tag'] == str(final_res[0][0]):
                if not 'context-filter' in i or i['context-filter'] == contextP['user']:
                    odp = random.choice(i['response'])
                    print("PowerBot says: " + odp)
                    say(odp, eng1)
                    if 'context-set' in i:
                        contextP['user'] = i['context-set']
                        break
    else:
        print(odp)
        say(odp, eng1)
    return final_res, odp



'''   
ints = importJson('3.json')
w,c,d = preprocessData(ints)
train_x, train_y = transformData(w,c,d)
mod = buildNetworkModel(train_x, train_y)
startTrain(mod, train_x, train_y)
saveAll(w,c,train_x,train_y)
print("all done!")
'''
eng1 = initVoice(1)
params = restoreParams('training_data3', '3.json')
model = loadModel('./model.tflearn', params[2], params[3])
print("Welcome to PowerAIBot app")
print("Say something :)")
contextP = {}
while True:
    a = input(">>> ")
    final_res,odp = classify(a, model,params[0], params[1], params[4], 0.35)
    if final_res and final_res[0][0] == 'bye':
        break
