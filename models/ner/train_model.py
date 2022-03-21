import sys
path = "/Users/giyeon/Section4//Project4/project4_chatbot"
sys.path.append(path)

import tensorflow as tf
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from utils.Preprocess import Preprocess

"""
개체명 분류 사전 불러오는 함수
"""

def read_file(file_name):
    sents = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            if l[0] == ';' and lines[idx + 1][0] == '$': #사전의 첫번째 문장을 읽으면 빈 리스트 생성
                this_sent = []
            elif l[0] == '$' and lines[idx - 1][0] == ';': #사전의 두번째 문장을 읽으면 그냥 통과
                continue
            elif l[0] == '\n': 
                sents.append(this_sent) #화이트 스페이스(bio 사전의 끝)이 나오면 sents에 bio 정보를 저장
            else:
                this_sent.append(tuple(l.split())) #bio 정보가 나오면 각 줄을 화이트스페이스 단위로 this_sent 리스트에 저장
    return sents

"""
전처리 객체 생성
"""

p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict.bin', userdic='../../utils/user_dic.tsv')

#학습용 말뭉치 데이터 불러오기
corpus = read_file('ner_train.txt')

#말뭉치 데이터에서 단어와 BIO 태그만 불러와서 학습용 데이터셋 생성
sentences, tags = [], []
for t in corpus:
    tagged_sentence = []
    sentence, bio_tag = [], []
    for w in t:
        tagged_sentence.append((w[1], w[3])) #단어와 태그만 저장
        sentence.append(w[1]) #단어 시퀀스만 저장
        bio_tag.append(w[3]) #bio_tag만 저장
    
    sentences.append(sentence)
    tags.append(bio_tag)

#breakpoint()

print("샘플 크기 : \n", len(sentence))
print("0번째 샘플 단어 시퀀스 : \n", sentences[0])
print("0번째 샘플 bio 태그 : \n", tags[0])
print("샘플 단어 시퀀스 최대 길이 : ", max(len(l) for l in sentences))
print("샘플 단어 시퀀스 평균 길이 : ", (sum(map(len, sentences))/len(sentences)))

#토크나이저 정의
tag_tokenizer = preprocessing.text.Tokenizer(lower=False)
tag_tokenizer.fit_on_texts(tags)

#단어 사전 및 태그 사전 크기
vocab_size = len(p.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1
print("BIO 태그 사전 크기 :", tag_size)
print("단어 사전 크기 :", vocab_size)

"""
학습용 단어 시퀀스 생성
feature : 단어 토큰
target : 단어 토큰에 해당하는 BIO 정보
"""

X_train = [p.get_wordidx_sequence(sent) for sent in sentences]
y_train = tag_tokenizer.texts_to_sequences(tags)

index_to_ner = tag_tokenizer.index_word
index_to_ner[0] = 'PAD'

#시퀀스 패딩 처리
max_len = 40
X_train = preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=max_len)
y_train = preprocessing.sequence.pad_sequences(y_train, padding='post', maxlen=max_len)

#학습 데이터 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#target 원-핫 인코딩

y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)

print("학습 샘플 시퀀스 형상 : ", X_train.shape)
print("학습 샘플 레이블 형상 : ", y_train.shape)
print("테스트 샘플 시퀀스 형상 : ", X_test.shape)
print("테스트 샘플 레이블 형상 : ", y_test.shape)


"""
양방향 LSTM 모델 구성
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam


#모델 구성
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=30, input_length=max_len, mask_zero=True)) #mask_zero=True : 패딩처리된 값을 임베딩 과정에서 무시함
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))) #return_sequences=True : 매 time-step의 은닉상태를 출력, 과적합 방지를 위해 dropout 적용, 순환신경망 노드 내부의 드롭아웃 적용
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))

#모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])

#학습 데이터 훈련
model.fit(X_train, y_train, batch_size=128, epochs=10)

print("평가 결과 : ", model.evaluate(X_test, y_test)[1])
model.save('ner_model_1.h5')

"""
시퀀스를 NER 태그로 변환

- 예측값을 index_to_ner을 사용하여 태깅 정보로 변환 시킴
"""

def sequences_to_tag(sequences):
    result = []
    for sequence in sequences: #전체 시퀀스로부터 시퀀스를 하나씩 꺼냄
        temp = []
        for pred in sequence: #시퀀스로부터 예측값을 하나씩 꺼냄
            pred_index = np.argmax(pred) #가장 큰 값의 인덱스를 반환
            temp.append(index_to_ner[pred_index].replace("PAD", "0"))
        result.append(temp)
    return result


"""
F1-score 계산
"""

from seqeval.metrics import f1_score, classification_report

y_pred = model.predict(X_test)
pred_tags = sequences_to_tag(y_pred) #예측된 NER
test_tags = sequences_to_tag(y_test) #실제 NER

print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))
