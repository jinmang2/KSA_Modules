import math, sys
from collections import defaultdict
from konlpy.tag import Okt

class BayesianFilter:
    """베이지안 필터"""
    def __init__(self, laplas=1):
        self.laplas = laplas
        self.words = set() # 출현한 단어 기록
        self.word_dict = defaultdict(
                    lambda : defaultdict(int)) # 카테고리마다 출현 횟수 기록
        self.category_dict = defaultdict(int) # 카테고리 출현 횟수 기록

    def split(self, text):
        """형태소 분석하기 --- (%2)"""
        results, twitter = [], Okt()
        # 단어의 기본형 사용
        malist = twitter.pos(text, norm=True, stem=True)
        for word in malist:
            # 어미/조사/구두점 등은 대상에서 제외
            if not word[1] in ['Josa', 'Eomi', 'Punctuation']:
                results.append(word[0])
        return results

    def inc_word(self, word, category):
        # 단어를 카테고리에 추가
        self.word_dict[category][word] += 1
        self.words.add(word)

    def inc_category(self, category):
        # 카테고리 계산하기
        self.category_dict[category] += 1

    def fit(self, text, category):
        """텍스트 학습하기 --- (%3)"""
        word_list = self.split(text)
        for word in word_list:
            self.inc_word(word, category)
        self.inc_category(category)

    def score(self, words, category):
        """단어 리스트 점수 매기기 --- (%4)"""
        score = math.log(self.category_prob(category))
        for word in words:
            score += math.log(self.word_prob(word, category))
        return score

    def predict(self, text):
        """예측하기 --- (%5)"""
        best_category = None
        max_score = -sys.maxsize
        words = self.split(text)
        score_list = []
        for category in self.category_dict.keys():
            score = self.score(words, category)
            score_list.append((category, score))
            if score > max_score:
                max_score = score
                best_category = category
        return best_category, score_list

    def get_word_count(self, word, category):
        """카테고리 내부의 단어 출현 횟수 구하기"""
        return self.word_dict[category][word]

    def category_prob(self, category):
        """카테고리 계산"""
        sum_categories = sum(self.category_dict.values())
        category_v = self.category_dict[category]
        return category_v / sum_categories

    def word_prob(self, word, category):
        """카테고리 내부의 단어 출현 비율 계산"""
        n = self.get_word_count(word, category) + self.laplas
        d = sum(self.word_dict[category].values()) + len(self.words)
        return n / d

class BayesianClassifier(BayesianFilter):
    pass
