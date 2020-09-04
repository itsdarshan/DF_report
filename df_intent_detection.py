import dialogflow
import os
import json
from google.protobuf.json_format import MessageToDict
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from config import project_id, session_id, os


def detect_intent_texts(project_id, session_id, text, language_code):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)

    if text:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)  # getting text input
        query_input = dialogflow.types.QueryInput(text=text_input)  # preparing query input for intent detection
        response = session_client.detect_intent(
            session=session, query_input=query_input)  # preparing response

        return response.query_result  # accessing query


def df_acc_for_curr_phrases(directory):
    os.chdir(directory)
    report = []
    # cr = {}
    intent_name = ''
    y_pred, y_true, y_neg, acc = [], [], [], 0
    for file in os.listdir():
        if '_usersays_en' in file or '__usersays_en' in file:
            f_open = open(file, )
            phrase_list = json.load(f_open)
            if '_usersays_en' in file:
                intent_name = file.replace('_usersays_en.json', '')
            if '__usersays_en' in file:
                intent_name = file.replace('__usersays_en.json', '?')
            for phrases in range(0, len(phrase_list)):
                df_intent = detect_intent_texts(project_id, session_id, phrase_list[phrases]['data'][0]['text'], 'en')
                dict_data = MessageToDict(message=df_intent)
                y_pred.append(dict_data['intent']['displayName'])
                y_true.append(intent_name)
                if dict_data['intent']['displayName'] != intent_name:
                    y_neg.append(phrase_list[phrases]['data'][0]['text'])
            acc = accuracy_score(y_true, y_pred)
            pr = precision_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            rec = recall_score(y_true, y_pred, average='macro')
            # cr = classification_report(y_true, y_pred)
            report.append((intent_name, (len(y_neg), len(phrase_list)),
                           (round(acc*100, 2), round(f1, 2), round(pr, 2), round(rec, 2)), y_neg))
            f_open.close()
            y_pred, y_true, y_neg, acc, pr, rec, f1 = [], [], [], 0, 0, 0, 0
    return report # , cr


def df_acc_for_new_phrases(directory, new_directory):
    curr_phrase_report = df_acc_for_curr_phrases(directory)
    new_phrase_report = df_acc_for_curr_phrases(new_directory)
    return curr_phrase_report, new_phrase_report
