import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random
import tensorflow as tf
import json
import tflearn
import pickle
import pyttsx3
import numpy as np
from watson_developer_cloud import ConversationV1

lemmatizer = WordNetLemmatizer()

with open('./cred.txt', 'r') as f:
    cred = json.loads(f.read())
conversation = ConversationV1(username=cred['username'],
                              password=cred['password'],
                              version='2018-06-10')
context = {}

def initVoice():
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 20)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[2].id)
    return engine

def setVoice(engine, voice=1):
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[voice].id)
    return engine
    

def say(text, engine):
    engine.say(text)
    engine.runAndWait()

def importJson(file):
    with open(file) as json_data:
        intents = json.loads(json_data.read())
    return intents

def restoreParams(pickleFile, jsonFile):
    data = pickle.load(open(pickleFile, 'rb'))
    words = data['words']
    classes = data['classes']
    train_x = data['train_x']
    train_y = data['train_y']
    intents = importJson(jsonFile)
    return words, classes, train_x, train_y, intents

def loadModel(mFile, train_x, train_y):
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 15)
    net = tflearn.fully_connected(net, 15)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)
    mod = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    mod.load(mFile,weights_only=True)
    return mod


def bagUser(sentence, words, show_details=False):
    sent_words = word_tokenize(sentence)
    sent_words = [lemmatizer.lemmatize(w.lower()) for w in sent_words]
    bag = [0]*len(words)
    for s in sent_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i] = 1
                if show_details:
                    print('Found in bag: {}'.format(w))
    return (np.array(bag))

def classify(bag, model, classes, intents, error):
    eng1 = setVoice(engine, 1)
    results = model.predict([bag])[0]
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
                    print()
                    say(odp, eng1)
                    if 'context-set' in i:
                        contextP['user'] = i['context-set']
                        break
    else:
        print("PowerBot says: " + odp)
        print()
        say(odp, eng1)
    return final_res, odp

engine = initVoice()
words, classes, train_x, train_y, intents = restoreParams('training_data3', '3.json')
model = loadModel('./model.tflearn', train_x, train_y)
print("-----Starting conversation------")
Wbot = "Hello I am IBM Watson Service Bot. Let's talk!"
contextP = {}
print(Wbot)
say(Wbot, engine)

while True:
    print()
    bag = bagUser(Wbot, words)
    final_res, ans = classify(bag, model, classes, intents, 0.35)
    if final_res and final_res[0][0] == 'bye':
        break
    resp = conversation.message(
        workspace_id='d9a652c2-b773-4e68-bf1c-441d6c4d0e40',
        input={'text':ans},
        context=context)
    print("WatsonBot says:" + " " + resp['output']['text'][0])
    eng2 = setVoice(engine, 2)
    say(resp['output']['text'][0], eng2)
    context = resp['context']
    Wbot = resp['output']['text'][0]
