# PowerAIbot
Project of two chatbots. First one is build with Python and TensorFlow and Tflearn libs.
Second uses Python and IBM Cloud tools, particularly Watson Conversation module.
## Getting Started
This is public repo so just clone it!
```
git clone https://github.com/tobiw777/PowerAIbot
```
### Prerequisites
This project was built in Power8 Platform and tested there also:) So probalby need strong pc specs!
Due to fact it was created with Python3, you need Python3 version >= 3.64

If you use some virtual envs like conda, check if you have path to you virtual env's libs in $PYTHONPATH

You also need some common libs like numpy, nltk etc. All have been written in requirements.txt so just use:
```
pip install -r requirements.txt
```
For WatsonBot you also need to get Watson-developer-cloud lib so use:
```
pip install --update watson-developer-cloud
```
All needed credentials are included!
### Running project
If you got all required libs you can simply run chatbots.
To run WatsonBot just:
```
python3 WatsonBot.py
```
In order to run PowerAIBot use:
```
python3 PowerAIBot.py
```
### PowerBot contexts
PowerAIBot's small talk is about simple registration procedure. You can ask him what he could do for you.
### WatsonBot contexts
WatsonBot uses default workspace!
