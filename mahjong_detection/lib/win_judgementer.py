import copy
import numpy as np


class WinJudgementer:
    """
    画像で判定した14要素の配列を受け取り、和了判定をするクラス（七対子と国士無双はここで役計算する）
    """
    def __init__(self, list_pi_name):
        self.list_pi_name = list_pi_name
        self.dict_pi = \
            {'1m':0, '2m':1, '3m':2, '4m':3, '5m':4, '6m':5, '7m':6, '8m':7, '9m':8,
            '1p':9, '2p':10, '3p':11, '4p':12, '5p':13, '6p':14, '7p':15, '8p':16, '9p':17,
            '1s':18, '2s':19, '3s':20, '4s':21, '5s':22, '6s':23, '7s':24, '8s':25, '9s':26,
            'h':27, 'f':28, 'c':29, 'e':30, 's':31, 'w':32, 'n':33}

    def _check_array(self):
        if not len(self.list_pi_name) == 14:
            raise ValueError('input must be 14 tiles.')

    def pi_convert_to_int(self):
        self._check_array()
        if not type(self.list_pi_name[0]) == int:
            list_pi_num = [self.dict_pi[pi_name] for pi_name in self.list_pi_name]
            return list_pi_num
        else:
            return self.list_pi_name

    def countpai(self, tehai): # https://qiita.com/okuzawats/items/5fdd8036f9223e021d87
    # 手牌の枚数を牌種ごとにカウント
    # 0〜33までのインデックスを使う
        counter = []
        for i in range(0,34):
            counter.append(tehai.count(i))
        return counter

    def is_head(self, counter,i):
    # 対子かどうかの判定
        if counter[i] >= 2:
            return True
        else:
            return False

    def is_pung(self, counter,i):
    # 刻子を探す
        if counter[i] >= 3:
            return True
        else:
            return False

    def is_chow(self, counter,i):
    # 順子を探す
        if counter[i]>=1 and counter[i+1]>=1 and counter[i+2]>=1:
            return True
        else:
            return False

    def is_thirteen_orphans(self, counter):
        # 定数。ヤオチュー牌のindex
        index_yaochu = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        index_able_tiles = [index for index in range(len(counter)) if counter[index]>=1]
        if len(index_able_tiles) == 13:
            flg_thirteen_orphans = np.allclose(index_yaochu, index_able_tiles)
            return flg_thirteen_orphans
        else:
            flg_thirteen_orphans = False
            return flg_thirteen_orphans


    def agari(self):
        # あがりの判定
        agarihantei = False
        # 七対子判定用
        seven_pairs_cnt = 0
        # 上がり牌を文字列から数値に変換
        tehai = self.pi_convert_to_int()
        # 手牌の枚数を牌種ごとにカウント
        counter = self.countpai(tehai)
        # 各牌について対子かどうか判定
        for i in range(len(counter)):
            # カウンタを保持
            tmp_counter = copy.deepcopy(counter) # tmp=counterとすると上手くいかない
            # 対子の判定
            if self.is_head(tmp_counter,i):
                tmp_counter[i] += -2 #雀頭を除去
                seven_pairs_cnt += 1
                head = i
            # 刻子の判定
                for j in range(len(counter)):
                    if self.is_pung(tmp_counter,j):
                        tmp_counter[j] += -3 #刻子を除去
                    else:
                        continue
            # 順子の判定
                for num in range(0,4): # 一盃口判定用
                    for k in range(len(counter)-7): #字牌の処理は不要
                        if self.is_chow(tmp_counter,k):
                            tmp_counter[k] += -1
                            tmp_counter[k+1] += -1
                            tmp_counter[k+2] += -1
                        else:
                            continue
                if sum(tmp_counter) == 0:
                    agarihantei = True
                    break
            else: #対子でない場合
                continue
        if agarihantei == True:
            return_txt = 'あがり!!!!!!!!!'
            yaku = None
        elif seven_pairs_cnt == 7:
            return_txt = 'seven_pairs!!!!!!'
            yaku = 'seven_pairs'
        elif self.is_thirteen_orphans(counter):
            return_txt = 'kokushi!!!!!!!!'
            yaku = 'kokushi'
        else:
            return_txt = 'No-ten'
        return return_txt, head, yaku
