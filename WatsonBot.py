from watson_developer_cloud import ConversationV1
import json
import sys


def loadCred(cred):
    with open(cred, 'r') as f:
        creds = json.loads(f.read())
    return creds

if __name__ == "__main__":
    creds = loadCred(sys.argv[1])
    conversation = ConversationV1(username=creds['username'],
                                  password=creds['password'],
                                  version='2018-06-15')
    context = {}
    print("Hello I am WatsonBot, How can I help you?")
    while(True):
        ask = input(">> ")
        if ask.lower() == 'bye':
            response = conversation.message(workspace_id=creds['workspace_id'],
                input={'text': ask}, context=context)
            print(response['output']['text'][0])
            break
        resp = conversation.message(
            workspace_id=creds['workspace_id'],
            input={'text':ask},
            context=context)
        print(resp['output']['text'][0])
        context = resp['context']


