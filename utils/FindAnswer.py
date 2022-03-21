class FindAnswer:
    def __init__(self, db):
        self.db = db

    
    #검색 쿼리 생성
    def _make_query(self, intent_name, ner_tags):
        sql = "SELECT * FROM chatbot_train_data"
        if intent_name != None and ner_tags == None:
            sql = sql + " WHERE intent='{}' ".format(intent_name)

        elif intent_name != None and ner_tags != None:
            where = ' WHRER intent="%s" ' % intent_name
            if (len(ner_tags) > 0):
                where += 'and ('
                for ne in ner_tags:
                    where += " ner LIKE '%{}%' OR ".format(ne)
                where = where[:-3] + ')'
            sql = sql + where

        sql = sql + " ORDER BY rand() LIMIT 1"
        return sql

    #답변 검색

    def search(self, intent_name, ner_tags):
        sql = self._make_query(intent_name, ner_tags)
        answer = self.db.select_one(sql)

        #검색되는 답변이 없으면 의도명만 검색
        if answer is None:
            sql = self._make_query(intent_name, None)
            answer = self.db.select_one(sql)
        
        return (answer['answer'], answer['answer_image'])

    #NER 태그를 실제 입력된 단어로 변환

    def tag_to_word(self, ner_predicts, answer):
        for word, tag in ner_predicts:
            if tag == 'B_POLICY':
                answer = answer.replace(tag, word)

        answer = answer.replace('{', '')
        answer = answer.replace('}', '')
        return answer