"""
의도 분류 모델 학습
인사 - 0
욕설 - 1
질문 - 2
예약 - 3
기타 - 4
"""

#모듈 import

import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate


"""
학습 데이터 읽기

feature = queries
label = intents
"""

train_file = "total_train_data.csv"
data = pd.read_csv(train_file, delimiter=',')
queries = data['query'].tolist()
intents = data['intent'].tolist()

import sys
path = "/Users/giyeon/Section4//Project4/project4_chatbot"
sys.path.append(path)

from utils.Preprocess import Preprocess
p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict.bin',
                userdic='../../utils/user_dic.tsv')


"""
단어 시퀀스 생성

- 학습 데이터의 쿼리를 문장 단위로 출력(for)
- 출력된 문장을 형태소 단위로 출력(p.pos())
- 형태소 단위의 단어에서 불용어 제외(get_keywords)
- 불용어 제외된 단어의 시퀀스를 index로 출력(get_wordidx_sequence)
"""

sequence = []
for sentence in queries:
    #형태소 토큰화
    pos = p.pos(sentence)
    #토큰화한 단어 추출
    keywords = p.get_keywords(pos, without_tag=True)
    #단어를 인덱스로 변환
    seq = p.get_wordidx_sequence(keywords)
    sequence.append(seq)

#breakpoint()

from config.GlobalParams import MAX_SEQ_LEN
#시퀀스 패딩
padded_seqs = preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding = 'post')

"""
학습 데이터 분리&데이터 파이프라인 구축

대량의 데이터를 메모리에 무리주지 않고 처리하는 dataset 메소드 사용
학습:검증:데스트셋의 비율 = 7:2:1

데이터를 처리하는 batch 크기는 20으로 설정
"""
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents))
ds = ds.shuffle(len(queries))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

"""
하이퍼 파라미터 설정

드롭 아웃 비울 = 0.5
임베딩 벡터의 사이즈 = 128
훈련 횟수  = 5
단어 사전의 크기 = 전체 단어 사전
"""
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(p.word_index) + 1

"""
CNN 모델 정의
"""

input_layer = Input(shape=(MAX_SEQ_LEN,)) #패딩된 길이의 단어시퀀스 입력
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer) #VOCAB_SIZE * MAX_SEQ_LEN * EMB_SIZE의 임베딩 벡터 생성
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer) #과적합 방지를 위한 dropout 적용

#합성곱 신경망 병렬 구성
conv1 = Conv1D(filters=128,
                kernel_size=3,
                padding='valid',
                activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(filters=128,
                kernel_size=4,
                padding='valid',
                activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(filters=128,
                kernel_size=5,
                padding='valid',
                activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

#pooling 결과 concatenate
concat = concatenate([pool1, pool2, pool3])

#fully connected layer 구성
hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(5, name='logits')(dropout_hidden)
predictions = Dense(5, activation=tf.nn.softmax)(logits)

"""
모델 생성
"""

model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

"""
모델 학습
"""
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

"""
모델 평가
"""
loss, accuracy = model.evaluate(test_ds, verbose=1)
print('Accuracy: %f' % (accuracy * 100))
print('loss %f' % (loss))

"""
모델 저장
"""
model.save('intent_model.h5')