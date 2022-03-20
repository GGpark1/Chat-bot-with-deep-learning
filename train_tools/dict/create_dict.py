import sys
path = "/Users/giyeon/Section4//Project4/project4_chatbot"
sys.path.append(path)

from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle

"""
말뭉치 데이터로 단어 사전 생성
단어 사전을 기반으로 의도 분류 및 개체명 인식 모델 학습 진행
"""


#말뭉치 데이터 read

def read_corpus_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    
    return data

#말뭉치 데이터 가져오기

corpus_data = read_corpus_data('./corpus.txt')

p = Preprocess(word2index_dic='chatbot_dict.bin',
               userdic = '../../utils/user_dic.tsv')

dict = []
for c in corpus_data:
    pos = p.pos(c[1]) #문장을 형태소 분류기에 input
    for k in pos:
        dict.append(k[0]) #형태소 분류기를 통과한 데이터에서 키워드만 추출

#키워드 리스트를 단어-인덱스(사전) 형태로 전환

tokenizer = preprocessing.text.Tokenizer(oov_token = 'OOV')
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index

#breakpoint()

#사전 파일 생성

f = open("chatbot_dict.bin", "wb")
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()



