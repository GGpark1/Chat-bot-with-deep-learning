from konlpy.tag import Komoran

class Preprocess:
    def __init__(self, userdic=None):
        
        #형태소 분석기 초기화
        
        self.komoran = Komoran(userdic=userdic)

        #형태소 분석에 제외할 품사 추가
        #관계언, 기호, 어미, 접미사 제거

        self.exclusion_tags = [
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EX', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
        ]

    #형태소 pos 태깅
    def pos(self, sentence):
        return self.komoran.pos(sentence)

    #불용어 제거 및 필요한 품사 정보만 출력
    def get_keywords(self, pos, without_tag=False):
        #불용어 리스트와 단어의 일치 여부를 확인하는 임시 함수 생성
        f = lambda x: x in self.exclusion_tags
        word_list = []
        #pos 태깅이 불용어에 없다면 리스트에 추가
        for p in pos:
            if f(p[1]) is False:
                word_list.append(p if without_tag is False else p[0])
            return word_list

    def get_wordidx_sequence(self, keywords):
        if self.word_index is None:
            return []
        w2i = []
        for word in keywords:
            try:
                w2i.append(self.word_index[word])
            except KeyError:
                w2i.append(self.word_index['OOV'])
        return w2i
        

