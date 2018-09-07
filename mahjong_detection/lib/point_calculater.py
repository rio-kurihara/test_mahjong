import copy
import numpy as np


class PointCalculater:
    """
    画像で判定した牌の配列を受け取り、役を計算して点計算するクラス
    """
    def __init__(self, list_pi_name, wj, index_seat_wind, index_round_wind, dora
                 , flg_reach=False, flg_double_reach=False, flg_one_shot=False, flg_concealed_hand=False
                 , flg_claim=False, flg_haitei=False, flg_rinshan=False, flg_chankan=False, flg_earthly_win=False, flg_heavenly_win=False):

        self.list_pi_name = list_pi_name
        self.index_seat_wind  = [index_seat_wind]  # 自風は北家だとする
        self.index_round_wind = [index_round_wind] # 東場だとする
        self.dora = dora

        # 手牌の枚数を牌種ごとにカウント
        tehai = wj.pi_convert_to_int()
        self.counter = wj.countpai(tehai)
        self.return_txt, self.eye, self.yaku = wj.agari()

        self.flg_reach          = flg_reach # リーチしたか
        self.flg_double_reach   = flg_double_reach# ダブリーか
        self.flg_one_shot       = flg_one_shot # 一発上がりか
        self.flg_concealed_hand = flg_concealed_hand # メンツモか
        self.flg_claim          = flg_claim # 鳴き有無
        self.flg_haitei         = flg_haitei # 河底撈魚 ★TODO：山の情報がないと海底牌判定ができない。余裕があれば対応したい
        self.flg_rinshan        = flg_rinshan # 嶺上開花 ★TODO：同上
        self.flg_chankan        = flg_chankan # 槍槓 ★TODO：同上
        self.flg_earthly_win    = flg_earthly_win # 地和かどうか ※デフォルトFalseにする
        self.flg_heavenly_win   = flg_heavenly_win # 天和かどうか　※デフォルトFalseにする

        # 白發中牌
        self.index_dragons = [27, 28, 29]
        # 役牌計算用
        self.index_yakuhai = self.index_dragons + self.index_seat_wind + self.index_round_wind
        # 風牌
        self.index_winds = [30 ,31, 32, 33]
        # 字牌
        self.index_characters = self.index_dragons + self.index_winds
        # 一九牌
        self.index_one_nine = [0, 8, 9, 17, 18, 26]
        # ヤオチュー牌
        self.index_yaochu = self.index_characters + self.index_one_nine
        # 緑一色用
        self.index_all_green = [19, 20, 21, 23, 25, 28]


    def add_han(self, han_cnt, point):
        # 翻数を追加する関数
        han_cnt += point
        return han_cnt


    def is_characters(self):
        # 一九字牌の有無をチェックする関数
        # 有効な牌のindexを取得
        able_tiles = [i[0] for i in enumerate(self.counter) if i[1] >= 1]
        cnt = 0
        for index in able_tiles:
            if index in self.index_yaochu:
                cnt += 1
        if cnt == 0:
            flg_characters = False
        else:
            flg_characters = True
        return flg_characters

    def is_chow(self, counter, i):
        # 順子を探す関数
        index_chow = [i[0] for i in enumerate(counter) if i[1] >= 3]
        if not len(index_chow) == 0:
            flg_chow = True
        else:
            flg_chow = False
        return flg_chow

    def check_chow(self):
        # 順子の有無をチェック
        index_chow = [i[0] for i in enumerate(self.counter) if i[1] >= 3]
        if not len(index_chow) == 0:
            flg_chow = True
        else:
            flg_chow = False
        return flg_chow

    def is_pung(self, counter,i):
        # 刻子を探す関数
        if counter[i] >= 3:
            return True
        else:
            return False

    def is_kung(self, counter,i):
        # 槓子を探す関数
        if counter[i] == 4:
            return True
        else:
            return False

    def is_yaochu(self, i):
        # indexを受け取り、それが一九字牌(ヤオチュー牌)か判定する関数
        if i in self.index_yaochu:
            return True
        else:
            return False

    def is_one_nine(self, i):
        # indexを受け取り、それが一九か判定する関数
        if i in self.index_one_nine:
            return True
        else:
            return False

    def get_indexs_sanshoku(self):
        # 三色同順判定用の順子のindexの組み合わせリストを取得する関数
        tmp = [0, 9, 18]
        index_sanshoku = []
        index_sanshoku.append(tmp)
        for i in range(0,8):
            index_sanshoku.append([index+1 for index in index_sanshoku[i]])
        return index_sanshoku

    def get_indexs_full_straight(self):
        # 一気通貫用のindex取得関数
        tmp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        index_full_straight = []
        index_full_straight.append(tmp)
        for i in range(0,2):
            index_full_straight.append([index+9 for index in index_full_straight[i]])
        return index_full_straight

    def get_indexs_suit(self):
        # 数牌のindexリストを返す関数
        index_characters = []
        index_circles = []
        index_bamboo = []
        for index in range(0,9):
            index_characters.append(index)
            index_circles.append(index+9)
            index_bamboo.append(index+18)
        return index_characters, index_circles, index_bamboo

    def yakuhai(self):
        # 役牌判定関数
        index_pong = [i[0] for i in enumerate(self.counter) if i[1] >= 3]
        point = 0
        for index in index_pong:
            if index in self.index_yakuhai:
                point += 1
                print('役牌')
        return point

    def all_simples(self):
        # タンヤオ判定関数 ※喰いタン有にしている
        # 字牌の有無をチェック
        if self.is_characters():
            point = 0
        else:
            point = 1
            print('タンヤオ')
        return point

    def pinfu(self):
        # ピンフ判定関数 ★TODO：両面待ちでないと成立しない条件加える
        # 順子の有無と雀頭が字牌でないかチェック
        if self.check_chow() or self.is_characters():
            point = 0
        else:
            point = 1
            print('平和')
        return point

    def double_run(self):
        # 一盃口判定関数
        point = 0
        if not self.flg_claim: # 面前のみ有効
            # カウンタを保持
            tmp_counter = copy.deepcopy(self.counter) # tmp=counterとすると上手くいかない
            index_start_chow = []
            for num in range(0,4):
                for k in range(len(self.counter)-7): #字牌の処理は不要
                    if self.is_chow(tmp_counter,k):
                        tmp_counter[k] += -1
                        tmp_counter[k+1] += -1
                        tmp_counter[k+2] += -1
                        index_start_chow.append(k)
                        list_double_run = [index for index in index_start_chow if k == index]
                        if len(list_double_run) == 2:
                            point = 1
                            print('一盃口')
        else:
            pass
        return point

    def sanshoku_chow(self):
        # 三色同順判定関数
        indexs_sanshoku = self.get_indexs_sanshoku()
        # カウンタを保持
        tmp_counter = copy.deepcopy(self.counter)
        index_start_chow = []
        sanshoku_chow_cnt = 0
        point = 0
        for num in range(0,4):
            for k in range(len(self.counter)-7): #字牌の処理は不要
                if self.is_chow(tmp_counter,k):
                    tmp_counter[k] += -1
                    tmp_counter[k+1] += -1
                    tmp_counter[k+2] += -1
                    index_start_chow.append(k)
                    if len(index_start_chow) == 3:
                        for index_sanshoku in indexs_sanshoku:
                            if np.allclose(index_start_chow, index_sanshoku):
                                sanshoku_chow_cnt += 1

        if sanshoku_chow_cnt == 1:
            print('三色同順')
            point = 2
            if self.flg_claim: # 喰い下がり1翻
                point += -1
        return point

    def sanshoku_pung(self):
        # 三色同刻判定関数　※三色同順とほぼ同じ。判定方法が順子か刻子かの違いと喰い下がりがない
        indexs_sanshoku = self.get_indexs_sanshoku()
        # カウンタを保持
        tmp_counter = copy.deepcopy(self.counter)
        index_start_pung = []
        sanshoku_pung = 0
        point = 0
        for num in range(0,4):
            for k in range(len(self.counter)-7): #字牌の処理は不要
                if self.is_pung(tmp_counter,k):
                    tmp_counter[k] += -3
                    index_start_pung.append(k)
                    if len(index_start_pung) == 3:
                        for index_sanshoku in indexs_sanshoku:
                            if np.allclose(index_start_pung, index_sanshoku):
                                sanshoku_pung += 1

        if sanshoku_pung == 1:
            print('三色同刻')
            point = 2
        return point

    def three_pung(self):
        # 三暗刻判定関数 ※4メンツが刻子の場合1鳴きはokだが、3メンツしか刻子でない場合は鳴くと成立しない
        # ★TODO：上がり牌が刻子のうちの一つで、かつロン上がりの場合は成立しない件余裕があれば組み込む
        # ★TODO；刻子が3つあって、鳴きがあった場合どの牌で鳴いたかの情報がないと判断できない。余裕があれば組み込む
        # ※この関数内で四暗刻判定をして、四暗刻なら三暗刻判定処理は通らない仕様にする（三暗刻＋四暗刻になるのを避けるため）

        cnt = self.four_pung()
        point = cnt

        if cnt == 0: # 四暗刻でなかったら三暗刻判定処理が走る
            # カウンタを保持
            tmp_counter = copy.deepcopy(self.counter)
            cnt = 0
            for num in range(0,4):
                for k in range(len(self.counter)): #字牌の処理は不要
                    if self.is_pung(tmp_counter,k):
                        tmp_counter[k] += -3
                        cnt += 1

            if cnt == 3 and not self.flg_claim:
                point = 2
                print('三暗刻')
            elif cnt == 4: # 4メンツ全てが刻子なら鳴いてても鳴いてなくても有効
                point = 2
                print('三暗刻')
            else:
                point = 0
        return point


    def full_straight(self):
        # 一気通貫判定関数
        indexs_full_straight = self.get_indexs_full_straight()
        able_tiles = [i[0] for i in enumerate(self.counter) if i[1] >= 1]
        for i in range(len(indexs_full_straight)):
            src_set = set(indexs_full_straight[i])
            tag_set = set(able_tiles)
            matched_list = list(src_set & tag_set)
            if len(matched_list) == 9:
                point = 2
                print('一気通貫')
                break;
            else:
                point = 0

        if point == 2 and self.flg_claim: # 喰い下がり1翻
            point = 1
        return point

    def seven_pairs(self):
        # 七対子判定関数★TODO：2翻25符らしい。よくわからんのでTODO
        if self.yaku == 'seven_pairs':
            if not self.flg_claim:
                point = 2
                print('七対子')
        else:
            point = 0
        return point

    def all_three_pung(self):
        # 対々和判定関数 鳴きOK
        tmp_counter = copy.deepcopy(self.counter)
        all_three_pung_cnt = 0
        for num in range(0,4):
            for k in range(len(self.counter)): #字牌の処理は不要
                if self.is_pung(tmp_counter,k):
                    tmp_counter[k] += -3
                    all_three_pung_cnt += 1

        if all_three_pung_cnt == 4: # 4メンツ全てが刻子なら鳴いてても鳴いてなくても有効
            point = 2
            print('対々和')
        else:
            point = 0
        return point

    def three_kong(self):
        # 三槓子判定関数 鳴きOK
        tmp_counter = copy.deepcopy(self.counter)
        kong_cnt = 0
        for num in range(0,4):
            for k in range(len(self.counter)): #字牌の処理は不要
                if self.is_kung(tmp_counter,k):
                    tmp_counter[k] += -4
                    kong_cnt += 1

        if kong_cnt == 3:
            point = 2
            print('三槓子')
        else:
            point = 0
        return point

    def big_three_dragons(self):
        # 大三元 鳴きOK
        # カウンタを保持
        tmp_counter = copy.deepcopy(self.counter)
        index_start_pung = []
        for k in range(27, 30): #三元牌の処理のみ
            if self.is_pung(tmp_counter,k):
                tmp_counter[k] += -3
                index_start_pung.append(k)
        if len(index_start_pung) == 3:
            # 頭が三元牌か判定
            if np.allclose(index_dragons, index_start_pung):
                point = 13
                print('大三元')
            else:
                point = 0
        else:
            point = 0
        return point

    def three_dragons(self):
        # 大三元+小三元判定関数 鳴きOK　※役牌がつくので実質4翻
        # この関数内で大三元判定をして、大三元なら小三元判定は通らない仕様にする（大三元＋小三元になるのを避けるため）
        # ★TODO：刻子と槓子1つずつだった場合も小三元判定されるようにする

        # カウンタを保持
        tmp_counter = copy.deepcopy(self.counter)
        index_start_pung = []
        index_start_kung = []
        cnt = self.big_three_dragons()
        point = cnt

        if cnt == 0:
            for k in range(27, len(self.counter)): #字牌の処理のみ
                if self.is_pung(tmp_counter,k):
                    tmp_counter[k] += -3
                    index_start_pung.append(k)

            for k in range(27, len(self.counter)): #字牌の処理のみ
                if self.is_kung(tmp_counter,k):
                    tmp_counter[k] += -4
                    index_start_kung.append(k)

            # 頭が三元牌か判定
            if self.eye in self.index_dragons and len(index_start_pung) == 2:
                point = 2
                print('小三元')
            elif self.eye in self.index_dragons and len(index_start_kung) == 2:
                point = 2
                print('小三元')
            else:
                point = 0
        return point

    def honroutou(self):
        # 混老頭判定関数 ※対々和か七対子がつくので実質4翻。門前なら三暗刻とも複合。門前自摸or単騎待ちロンなら四暗刻と複合し役満
        able_tiles = [i[0] for i in enumerate(self.counter) if i[1] >= 1]
        # 有効な牌がヤオチュー牌ならdropしていく
        [able_tiles.remove(index) for index in self.index_yaochu if index in able_tiles]

        # 有効な牌全てがヤオチュー牌ならリストは空になる
        if len(able_tiles) == 0:
            point = 2
            print('混老頭')
        else:
            point = 0

        return point

    def two_double_run(self):
        # 二盃口判定関数
        double_run_cnt = self.double_run()
        if double_run_cnt == 2:
            point = 3
            print('二盃口')
        else:
            point = 0
        return point

    def half_flush(self):
        # 混一色判定関数
        able_tiles = [i[0] for i in enumerate(self.counter) if i[1] >= 1]
        point = self.full_flush()
        if point == 0:
            # 数牌1種＋字牌のindexを作成し、有効な牌がそれらで不足ないなら混一色
            indexs_full_straight = self.get_indexs_full_straight()
            indexs_half_flash = [index+self.index_characters for index in indexs_full_straight]
            for i in range(len(indexs_full_straight)):
                tiles = [index for index in able_tiles if index in indexs_half_flash[i]]
                if len(tiles) == len(able_tiles):
                    point = 3
                    print('混一色')
                    break;
                else:
                    point = 0

            if point == 3 and self.flg_claim: # 喰い下がり2翻
                point = 2
        else:
            pass
        return point

    def full_flush(self):
        # 清一色判定関数
        indexs_full_flush = self.get_indexs_full_straight()
        able_tiles = [i[0] for i in enumerate(self.counter) if i[1] >= 1]
        # 有効な牌がある一色のうちの牌ならdropしていく
        for index_full_flush in indexs_full_flush:
            tmp_able_tiles = copy.deepcopy(able_tiles)
            [tmp_able_tiles.remove(index) for index in index_full_flush if index in able_tiles]
            # 有効な牌全てが数牌いずれか一色ならリストは空になる
            if len(tmp_able_tiles) == 0:
                point = 6
                print('清一色')
                if self.flg_claim:
                    point = 5 # 喰い下がり5翻
            else:
                point = 0

        return point

    def kokushi(self):
        # 国士無双判定関数
        if yaku == 'kokushi':
            if not self.flg_claim:
                point = 13
                print('国士無双')
        else:
            point = 0
        return point

    def four_pung(self):
        # 四暗刻判定関数 4メンツが刻子で、鳴きありだと無効
        tmp_counter = copy.deepcopy(self.counter)
        pong_cnt = 0
        for num in range(0,4):
            for k in range(len(self.counter)): #字牌の処理は不要
                if self.is_pung(tmp_counter,k):
                    tmp_counter[k] += -3
                    pong_cnt += 1

        if pong_cnt == 4 and not self.flg_claim:
            point = 13
            print('四暗刻')
        else:
            point = 0
        return point

    def all_green(self):
        # 緑一色判定関数
        able_tiles = [i[0] for i in enumerate(self.counter) if i[1] >= 1]
        # 有効な牌が緑牌ならdropしていく
        [able_tiles.remove(index) for index in self.index_all_green if index in able_tiles]

        # 有効な牌全てが緑牌ならリストは空になる
        if len(able_tiles) == 0:
            point = 13
            print('緑一色')
        else:
            point = 0

        return point

    def all_character(self):
        # 字一色判定関数
        able_tiles = [i[0] for i in enumerate(self.counter) if i[1] >= 1]
        # 有効な牌が字牌ならdropしていく
        [able_tiles.remove(index) for index in self.index_characters if index in able_tiles]

        # 有効な牌全てがヤオチュー牌ならリストは空になる
        if len(able_tiles) == 0:
            point = 13
            print('字一色')
        else:
            point = 0

        return point

    def four_kong(self):
        # 四槓子 鳴きOK
        # カウンタを保持
        tmp_counter = copy.deepcopy(self.counter)
        kong_cnt = 0
        for num in range(0,4):
            for k in range(len(self.counter)): #字牌の処理は不要
                if self.is_kung(tmp_counter,k):
                    tmp_counter[k] += -4
                    kong_cnt += 1

        if kong_cnt == 4: # 4メンツ全てが槓子なら鳴いてても鳴いてなくても有効
            point = 13
            print('四槓子')
        else:
            point = 0
        return point

    def little_four_winds(self):
        # 小四喜判定関数
        tmp_counter = copy.deepcopy(self.counter)
        pong_cnt = 0
        index_pong = []
        # 頭が風牌か判定
        if self.eye in self.index_winds:
            for num in range(0,4):
                for k in range(len(self.counter)):
                    if self.is_pung(tmp_counter,k):
                        tmp_counter[k] += -3
                        pong_cnt += 1
                        index_pong.append(k)
            # 刻子のindexが風牌のどれかのうち3つだったら小四喜
            winds_pong = [i for i in index_pong if i in self.index_winds]
            if len(winds_pong) == 3:
                point = 13
                print('小四喜')
            else:
                point = 0
        else:
            point = 0
        return point

    def big_four_winds(self):
        # 大四喜判定関数
        tmp_counter = copy.deepcopy(self.counter)
        pong_cnt = 0
        index_pong = []
        for num in range(0,4):
            for k in range(len(self.counter)):
                if self.is_pung(tmp_counter,k):
                    tmp_counter[k] += -3
                    pong_cnt += 1
                    index_pong.append(k)
        # 刻子のindexが風牌4つすべてだったら大四喜
        winds_pong = [i for i in index_pong if i in self.index_winds]
        if len(winds_pong) == 4:
            point = 13
            print('大四喜')
        else:
            point = 0
        return point


    def calc_rules_around_characters(self):
        # 字牌絡みの役をまとめて判定する関数
        # 役が高い順に判定していく。字一色→混一色→混老頭（混一色と混老頭は複合する）
        point = self.all_character()
        if not point == 0:
            return point
        else:
            half_flush_point = self.half_flush()
            honroutou_point = self.honroutou()
            point = half_flush_point + honroutou_point
            if not point == 0:
                return point
            else:
                point = 0
                return point

    def calc_rules_around_kong(self):
        # 三槓子と四槓子を判定する関数
        point = self.four_kong()
        if point == 0:
            point = self.three_kong()
            if point == 0:
                point = 0
                return point
            else:
                return point
        else:
            return point

    def heavenly_win(self):
        # 天和判定関数
        if self.flg_heavenly_win():
            point = 13
        else:
            point = 0
        return point

    def earthly_win(self):
        # 地和判定関数
        if self.flg_earthly_win():
            point = 13
        else:
            point = 0
        return point

    def pure_outside_hand(self):
        # ジュンチャン判定関数 全てのメンツが一九牌
        point = 0
        # 雀頭が一九牌か
        if self.is_one_nine(self.eye):
            pass
        else:
            return point

        tmp_counter = copy.deepcopy(self.counter)
        tmp_counter[self.eye] = 0
        cnt = 0

        # 刻子の判定
        for j in range(len(self.counter)):
            if self.is_pung(tmp_counter,j):
                tmp_counter[j] += -3 #刻子を除去
                if self.is_one_nine(j): # ヤオチュー牌か判定
                    cnt += 1
                else:
                    break;
        # 順子の判定
        for num in range(0,3): # 一盃口判定用
            for k in range(len(self.counter)):
                if self.is_chow(tmp_counter,k):
                    tmp_counter[k] += -1
                    tmp_counter[k+1] += -1
                    tmp_counter[k+2] += -1
                    if is_one_nine(k) or is_one_nine(k+2): # ヤオチュー牌か判定
                        cnt += 1
                    else:
                        break

        # すべての処理がおわったときにcnrが4だったら、全てのメンツがヤオチュー牌なのでチャンタ
        if cnt == 4:
            point = 3
            print('ジュンチャン')
            if self.flg_claim:
                point = 2 # 喰い下がり2翻
        else:
            point = 0

        return point

    def mixed_outside_hand(self):
        # チャンタ判定関数　※ジュンチャン判定をしてから（全てのメンツがヤオチュー牌）判定する
        cnt = self.pure_outside_hand()
        if not cnt == 0:
            point = cnt
        else:
            point = 0
            # 雀頭がヤオチュー牌か
            if self.is_yaochu(self.eye):
                pass
            else:
                return point

            # 雀頭のindexを0にする
            tmp_counter = copy.deepcopy(self.counter)
            tmp_counter[self.eye] = 0
            cnt = 0

            # 刻子の判定
            for j in range(len(self.counter)):
                if self.is_pung(tmp_counter, j):
                    tmp_counter[j] += -3 #刻子を除去
                    if self.is_yaochu(j): # ヤオチュー牌か判定
                        cnt += 1
                    else:
                        break;
            # 順子の判定
            for num in range(0,3): # 一盃口判定用
                for k in range(len(self.counter)):
                    if self.is_chow(tmp_counter,k):
                        tmp_counter[k] += -1
                        tmp_counter[k+1] += -1
                        tmp_counter[k+2] += -1
                        if self.is_yaochu(k) or self.is_yaochu(k+2): # ヤオチュー牌か判定
                            cnt += 1
                        else:
                            break

            # すべての処理がおわったときにcnrが4だったら、全てのメンツがヤオチュー牌なのでチャンタ
            if cnt == 4:
                point = 2
                print('チャンタ')
                if self.flg_claim:
                    point = 1 # 喰い下がり1翻
            else:
                point = 0
        return point

    def calc(self):
        # 上記すべての役判定関数を実行して、最終的な翻数を返す関数
        han_cnt = 0
        han_cnt = self.add_han(han_cnt, self.yakuhai())
        han_cnt = self.add_han(han_cnt, self.all_simples())
        han_cnt = self.add_han(han_cnt, self.pinfu())
        han_cnt = self.add_han(han_cnt, self.double_run())
        han_cnt = self.add_han(han_cnt, self.two_double_run())
        han_cnt = self.add_han(han_cnt, self.sanshoku_chow())
        han_cnt = self.add_han(han_cnt, self.sanshoku_pung())
        han_cnt = self.add_han(han_cnt, self.three_pung())
        han_cnt = self.add_han(han_cnt, self.full_straight())
        han_cnt = self.add_han(han_cnt, self.seven_pairs())
        han_cnt = self.add_han(han_cnt, self.all_three_pung())
        han_cnt = self.add_han(han_cnt, self.three_dragons())
        han_cnt = self.add_han(han_cnt, self.calc_rules_around_characters())
        han_cnt = self.add_han(han_cnt, self.all_green())
        han_cnt = self.add_han(han_cnt, self.calc_rules_around_kong())
        han_cnt = self.add_han(han_cnt, self.little_four_winds())
        han_cnt = self.add_han(han_cnt, self.big_four_winds())
        # han_cnt = add_han(han_cnt, nine_gate()) # ★TODO
        han_cnt = self.add_han(han_cnt, self.mixed_outside_hand())
        # ドラ分を追加
        han_cnt = self.add_han(han_cnt, self.dora)
#        if self.flg_reach:
#            han_cnt += 1
#            print('リーチ')
        if not self.dora == 0:
            txt_dora = 'ドラ{0}'.format(self.dora)
        txt_han = '計{0}翻'.format(han_cnt)
        return txt_dora, txt_han, self.return_txt
