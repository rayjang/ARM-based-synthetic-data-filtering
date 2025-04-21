# -*- coding: utf-8 -*-
# pip install --upgrade openai

import json
import os
import openai
import json
import time
import pickle
import pandas as pd
import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('-chat_gpt_api_key', help=' : Please enter chat gpt API acces key')
parser.add_argument('-start_index', help=' : Please enter start index')
parser.add_argument('-end_index', help=' : Please enter end index')
args = parser.parse_args()

# prompt - 620 tokens
PROMPT = """"The content below is text related to the trend of science and technology, which is extracted by OCR from research reports.
Please refer to the content to create Question-Answering data.
Things to consider are:
- Create a Q&A about the scientific fact itself.
- Question should be answered without the given text.
- Answer must include specific evidences.
- Question & Answer should be deal with facts or trend itself, not what are performed as R&D project.
Answer must not contain vocabulary referring to this report(the given text) such as "이 보고서에서는".
- we only need a Question-Answering pair where the answer consists of at least 5 sentences.
- Write all the questions and answers to be created in Korean, and ask for a conversational tone based on honorifics. 
- Output format is like the below: 
###Question1:{Question}
###Answer1:{Answer}
###Question2:{Question}
###Answer2:{Answer}
###Question3:{Question}
###Answer3:{Answer}
###Question4:{Question}
###Answer4:{Answer}
###Question5:{Question}
###Answer5:{Answer}"""


START_INDEX = int(args.start_index)#0
END_INDEX = int(args.end_index)#1000
CHUNK_SIZE =3000

# input & output path
INPUT_DATA = "./output/trend_xml.json"
OUTPUT_PATH = './raw_qa_data_long/'
OUTPUT_FOR_CHATGPT = './output/qa_data_by_chatgpt_long.json'
OUTPUT_FOR_POLYGLOT = './output/qa_data_for_polyglot_long_v2.json'
OUTPUT_FOR_DF_CSV = './output/qa_data_long_for_df.csv'
OUTPUT_FOR_TXT_Q = './output/q_long_txt.txt'
OUTPUT_FOR_TXT_A = './output/a_long_txt.txt'
OUTPUT_FOR_TXT_S = './output/s_long_txt.txt'
OUTPUT_FOR_DF_PICKLE = './output/qa_data_long_for_df.pickle'


# chatGPT api key
ACCESS_KEY = args.chat_gpt_api_key 
GPT_VERSION = 'gpt-3.5-turbo'

openai.api_key = ACCESS_KEY
completion = openai.Completion()

def get_json_data(json_path: str):
    assert json_path.endswith(".json")
    out = list()
    with open(json_path, 'r', encoding="utf-8") as file:
        for line in file:
            j = json.loads(line.strip())
            out.append(j)
    return out


def generate_raw_qa_data():
    out = get_json_data(INPUT_DATA)

    done_raw_dir = OUTPUT_PATH
    files = os.listdir(done_raw_dir)

    done_list = [os.path.splitext(file)[0] for file in files]


    shot = 0
    for row in out:
        shot +=1
        if (shot > END_INDEX) or (shot < START_INDEX):
            continue
        if row['report_no'] in done_list:
            print(f"{str(row['report_no'])} skipped\n")
            continue
        if row['trend'] != '':
            try:
                raw_txt = row['trend'] #'\n'.join(row['trend'])
                size_raw_txt = len(raw_txt)
                if len(raw_txt) < 100:
                    continue
                print(f"{str(row['report_no'])} in process\n")
                print("@@" + str(len(raw_txt)))

                if size_raw_txt < 3000:
                    question = PROMPT + raw_txt
                    messages = []
                    messages.append({"role": "user", "content": question})
                    chat = openai.ChatCompletion.create(model=GPT_VERSION, messages=messages)
                    answer = chat.choices[0].message.content
                else:
                    answer = ""
                    for i in range(0, size_raw_txt, 3000):
                        text_chunk = raw_txt[i:i + 3000]
                        question = PROMPT + text_chunk
                        messages = []
                        messages.append({"role": "user", "content": question})
                        chat = openai.ChatCompletion.create(model=GPT_VERSION, messages=messages)
                        chunk_answer = chat.choices[0].message.content
                        answer = answer + '\n' + chunk_answer
                        #time.sleep(60)
            except Exception as e:
                print('Error occured.', e)
                print(f"error: {row['report_no']}\n")
                continue
        else:
            continue

        with open(OUTPUT_PATH + str(row['report_no']) +'.txt', 'w', encoding="UTF-8") as file:
            file.write(answer)
        print(f"done with report No. {row['report_no']}\n")
        print(OUTPUT_PATH + str(row['report_no']) + '.txt\n')

def convert_to_dataframe():

    question_list = []
    answer_list = []
    source_list = []

    for file_nm in os.listdir(OUTPUT_PATH):
        if file_nm.endswith('.txt'):
            try:
                input_path = OUTPUT_PATH + file_nm
                with open(input_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except:
                print(f"[Skipped] {file_nm}\n")

            lines = [l for l in lines if len(l.strip()) > 1]

            existing_num = len(question_list)
            turn_flag = 0  #
            turn_num = 0
            for idx, line in enumerate(lines):
                if line.startswith("###Question") and turn_flag == 0:
                    if idx == len(lines) - 1:
                        break
                    if len(line.split(':')) < 2:
                        print(f'error: {line}\n')
                        break
                    question = line.split(':')[1].strip()
                    turn_num = int(line.split(':')[0].strip()[-1])
                    question_list.append(question)
                    turn_flag = 1
                elif line.startswith("###Answer") and turn_flag == 1:
                    if idx + 1 > len(lines) - 1:
                        continue
                    if len(line.split(':')) < 2:
                        print(f'error: {line}\n')
                        turn_flag = 0
                        break
                    answer = line.split(':')[1].strip()
                    a_num = int(line.split(':')[0].strip()[-1])
                    if turn_num == a_num:
                        answer_list.append(answer)
                    turn_flag = 0

            len_q = len(question_list)
            len_a = len(answer_list)

            if len_q != len_a:
                if len_q > len_a:
                    question_list = question_list[:len_a]
                elif len_q < len_a:
                    answer_list = answer_list[:len_q]
            new_num = len(question_list) - existing_num
            # if answer_list[-1].strip()[-1] != '.':
            #    question_list = question_list[:len_q-1]
            #    answer_list = answer_list[:len_q-1]

            source_list.extend([file_nm.split('.')[0] for _ in range(new_num)])


    d = {'question' : question_list, 'answer' : answer_list, 'source' : source_list}
    df = pd.DataFrame(d)

    df.to_csv(OUTPUT_FOR_DF_CSV, encoding='utf-8',sep='¶')

    with open(OUTPUT_FOR_DF_PICKLE, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

def convert_to_txt():
    question_list = []
    answer_list = []
    source_list = []

    for file_nm in os.listdir(OUTPUT_PATH):
        if file_nm.endswith('.txt'):
            try:
                input_path = OUTPUT_PATH + file_nm
                with open(input_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except:
                print(f"[Skipped] {file_nm}\n")

            lines = [l for l in lines if len(l.strip()) > 1]

            existing_num = len(question_list)
            turn_flag = 0  #
            turn_num = 0
            for idx, line in enumerate(lines):
                if line.startswith("###Question") and turn_flag == 0:
                    if idx == len(lines) - 1:
                        break
                    if len(line.split(':')) < 2:
                        print(f'error: {line}\n')
                        break
                    question = line.split(':')[1].strip()
                    try:
                        turn_num = int(line.split(':')[0].strip()[-1])
                    except:
                        continue
                    question_list.append(question)
                    turn_flag = 1
                elif line.startswith("###Answer") and turn_flag == 1:
                    if idx + 1 > len(lines) - 1:
                        continue
                    if len(line.split(':')) < 2:
                        print(f'error: {line}\n')
                        turn_flag = 0
                        break
                    answer = line.split(':')[1].strip()
                    try:
                        a_num = int(line.split(':')[0].strip()[-1])
                    except:
                        continue
                    if turn_num == a_num:
                        answer_list.append(answer)
                    turn_flag = 0

            len_q = len(question_list)
            len_a = len(answer_list)

            if len_q != len_a:
                if len_q > len_a:
                    question_list = question_list[:len_a]
                elif len_q < len_a:
                    answer_list = answer_list[:len_q]
            new_num = len(question_list) - existing_num

            source_list.extend([file_nm.split('.')[0] for _ in range(new_num)])

    with open(OUTPUT_FOR_TXT_Q, 'w+') as lf:
        lf.write('\n'.join(question_list))
    with open(OUTPUT_FOR_TXT_A, 'w+') as lf:
        lf.write('\n'.join(answer_list))
    with open(OUTPUT_FOR_TXT_S, 'w+') as lf:
        lf.write('\n'.join(source_list))

def convert_to_alpaca_format():
    question_list = []
    answer_list = []
    source_list = []

    for file_nm in os.listdir(OUTPUT_PATH):
        if file_nm.endswith('.txt'):
            try:
                input_path = OUTPUT_PATH + file_nm
                with open(input_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except:
                print(f"[Skipped] {file_nm}\n")

            lines = [l for l in lines if len(l.strip()) > 1]

            existing_num = len(question_list)
            turn_flag = 0  #
            turn_num = 0
            for idx, line in enumerate(lines):
                if line.startswith("###Question") and turn_flag == 0:
                    if idx == len(lines) - 1:
                        break
                    if len(line.split(':')) < 2:
                        print(f'error: {line}\n')
                        break
                    question = line.split(':')[1].strip()
                    turn_num = int(line.split(':')[0].strip()[-1])
                    question_list.append(question)
                    turn_flag = 1
                elif line.startswith("###Answer") and turn_flag == 1:
                    if idx + 1 > len(lines) - 1:
                        continue
                    if len(line.split(':')) < 2:
                        print(f'error: {line}\n')
                        turn_flag = 0
                        break
                    answer = line.split(':')[1].strip()
                    a_num = int(line.split(':')[0].strip()[-1])
                    if turn_num == a_num:
                        answer_list.append(answer)
                    turn_flag = 0

            len_q = len(question_list)
            len_a = len(answer_list)

            if len_q != len_a:
                if len_q > len_a:
                    question_list = question_list[:len_a]
                elif len_q < len_a:
                    answer_list = answer_list[:len_q]
            new_num = len(question_list) - existing_num
            # if answer_list[-1].strip()[-1] != '.':
            #    question_list = question_list[:len_q-1]
            #    answer_list = answer_list[:len_q-1]

            source_list.extend([file_nm.split('.')[0] for _ in range(new_num)])

    json_list = []
    for q, a, s in zip(question_list, answer_list, source_list):
        json_list.append({"instruction" : q, "output" : a, "source" : s})


    with open(OUTPUT_FOR_CHATGPT, 'w', encoding="UTF-8") as file:
        for line in json_list:
            file.write(json.dumps(line, ensure_ascii=False) + '\n')



def convert_to_polyglot_format():
    question_list = []
    answer_list = []
    source_list = []

    for file_nm in os.listdir(OUTPUT_PATH):
        if file_nm.endswith('.txt'):
            try:
                input_path = OUTPUT_PATH + file_nm
                with open(input_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except:
                print(f"[Skipped] {file_nm}\n")

            lines = [l for l in lines if len(l.strip()) > 1]

            existing_num = len(question_list)
            turn_flag = 0  #
            turn_num = 0
            for idx, line in enumerate(lines):
                if line.startswith("###Question") and turn_flag == 0:
                    if idx == len(lines) - 1:
                        break
                    if len(line.split(':')) < 2:
                        print(f'error: {line}\n')
                        break
                    question = line.split(':')[1].strip()
                    turn_num = int(line.split(':')[0].strip()[-1])
                    question_list.append(question)
                    turn_flag = 1
                elif line.startswith("###Answer") and turn_flag == 1:
                    if idx + 1 > len(lines) - 1:
                        continue
                    if len(line.split(':')) < 2:
                        print(f'error: {line}\n')
                        turn_flag = 0
                        break
                    answer = line.split(':')[1].strip()
                    a_num = int(line.split(':')[0].strip()[-1])
                    if turn_num == a_num:
                        answer_list.append(answer)
                    turn_flag = 0

            len_q = len(question_list)
            len_a = len(answer_list)

            if len_q != len_a:
                if len_q > len_a:
                    question_list = question_list[:len_a]
                elif len_q < len_a:
                    answer_list = answer_list[:len_q]
            new_num = len(question_list) - existing_num

            source_list.extend([file_nm.split('.')[0] for _ in range(new_num)])

    json_list = []
    for q, a, s in zip(question_list, answer_list, source_list):
        json_list.append({"text" : f"### 명령어: {q}\n\n### 결과: {a}"})

    with open(OUTPUT_FOR_POLYGLOT, 'w', encoding="UTF-8") as file:
        for line in json_list:
            file.write(json.dumps(line, ensure_ascii=False) + '\n')


# common functions
def read_txt(txt_path):
    with open(txt_path) as f:
        lines = [line.rstrip('\n') for line in f]
    return lines

def convert_to_polyglot_format_by_given_lists():
    question_list = read_txt('output/refined_q_list.txt')
    answer_list = read_txt('output/refined_a_list.txt')
    source_list = read_txt('output/refined_s_list.txt')


    json_list = []
    for q, a, s in zip(question_list, answer_list, source_list):
        json_list.append({"text" : f"### 명령어: {q}\n\n### 결과: {a}"})

    with open('./output/qa_data_for_polyglot_v3.json', 'w', encoding="UTF-8") as file:
        for line in json_list:
            file.write(json.dumps(line, ensure_ascii=False) + '\n')


def save_as_csv():
    question_list = read_txt('output/refined_q_list.txt')
    answer_list = read_txt('output/refined_a_list.txt')
    source_list = read_txt('output/refined_s_list.txt')

    d = {'question': question_list, 'answer': answer_list, 'source': source_list}
    df = pd.DataFrame(d)

    df.to_csv('refined_qa_list.csv', encoding='utf-8', sep='¶')


def main(argv, args) :
    #generate_raw_qa_data()
    #convert_to_txt()
    save_as_csv()
    #convert_to_polyglot_format_by_given_lists()
    #convert_to_polyglot_format()

if __name__ == '__main__' :
    argv = sys.argv
    main(argv, args)
