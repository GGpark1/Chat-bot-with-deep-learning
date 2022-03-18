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
        data = data[1:] #헤더 제거
    
    return data

#말뭉치 데이터 가져오기

corpus_data = read_corpus_data('./corpus.txt')

p = Preprocess()
dict = []
for c in corpus_data:
    pos = p.pos(c[1]) #문장을 형태소 분류기에 input
    

