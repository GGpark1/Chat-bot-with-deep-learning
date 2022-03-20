import sys
path = "/Users/giyeon/Section4//Project4/project4_chatbot"
sys.path.append(path)

from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel

p = Preprocess(word2index_dic='../train_tools/dict/chatbot_dict.bin', userdic='../utils/user_dic.tsv')

ner = NerModel(model_name='../models/ner/ner_model.h5', preprocess=p)
query = '월세지원 어떻게 받을 수 있나요?'
predicts = ner.predict(query)
tags = ner.predict_tags(query)
print(predicts)
print(tags)