import sys
path = "/Users/giyeon/Section4//Project4/project4_chatbot"
sys.path.append(path)

from utils.Preprocess import Preprocess

sent = "월세지원 받고 싶어"

#전처리 객체 생성
p = Preprocess(userdic='../utils/user_dic.tsv')

#형태소 분석기 실행
pos = p.pos(sent)

#불용어를 제외한 단어토큰 출력
ret = p.get_keywords(pos, without_tag=False)
print(ret)