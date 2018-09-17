import configparser
import re
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.constants import EAST, SOUTH, WEST, NORTH


class PointCalculater():
    # 点数計算用クラス
    def __init__(self, list_piname, win_pi, dora_pi, path_config, section='point_calculate'):
        self.list_piname = list_piname
        self.win_pi = win_pi
        self.dora_pi = dora_pi
        self.list_man = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m']
        self.list_pin = ['1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p']
        self.list_sou = ['1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s']
        self.list_honors = ['e','s','w','n','c', 'f', 'h']
        self.dict_honors = {'e':1, 's':2, 'w':3, 'n':4, 'c':5, 'f':6, 'h':7}
        self.path_config = path_config
        self.section = section

    def _piname2index(self, piname: str):
        # モデルで使用した牌の名前を一つ受け取り、136配列のインデックスで返す
        if piname in self.list_man:
            man = int(re.sub(r'\D', '', piname))
            pi = TilesConverter.string_to_136_array(man=[man])
        elif piname in self.list_pin:
            pin = int(re.sub(r'\D', '', piname))
            pi = TilesConverter.string_to_136_array(pin=[pin])
        elif piname in self.list_sou:
            sou = int(re.sub(r'\D', '', piname))
            pi = TilesConverter.string_to_136_array(sou=[sou])
        elif piname in self.list_honors:
            idx = self.dict_honors[piname]
            pi = [idx]
        else:
            raise ValueError('Not supported pi name: {}'.format(piname))

        return pi[0]
    
    def _get_list_pinum(self):
        # 検出された結果のリストを受け取り、牌の種類分リストを返す
        man = []
        pin = []
        sou = []
        honors = []

        for piname in self.list_piname:
            if piname in self.list_man:
                _man = int(re.sub(r'\D', '', piname))
                man.append(_man)
            elif piname in self.list_pin:
                _pin = int(re.sub(r'\D', '', piname))
                pin.append(_pin)
            elif piname in self.list_sou:
                _sou = int(re.sub(r'\D', '', piname))
                sou.append(_sou)
            elif piname in self.list_honors:
                idx = self.dict_honors[piname]
                honors.append(idx)
            else:
                raise ValueError('Not supported pi name: {}'.format(piname))

        return man, pin, sou, honors
    
    def _read_config(self):
        config = configparser.ConfigParser()
        config.read(self.path_config)
        handconfig = HandConfig(
            is_tsumo = eval(config.get(self.section, 'is_tsumo'))
            ,is_riichi = eval(config.get(self.section, 'is_riichi'))
            ,is_ippatsu = eval(config.get(self.section, 'is_ippatsu'))
            ,is_rinshan = eval(config.get(self.section, 'is_rinshan'))
            ,is_chankan = eval(config.get(self.section, 'is_chankan'))
            ,is_haitei = eval(config.get(self.section, 'is_haitei'))
            ,is_houtei = eval(config.get(self.section, 'is_houtei'))
            ,is_daburu_riichi = eval(config.get(self.section, 'is_daburu_riichi'))
            ,is_nagashi_mangan = eval(config.get(self.section, 'is_nagashi_mangan'))
            ,is_tenhou = eval(config.get(self.section, 'is_tenhou'))
            ,is_renhou = eval(config.get(self.section, 'is_renhou'))
            ,is_chiihou = eval(config.get(self.section, 'is_chiihou'))
            ,player_wind = eval(config.get(self.section, 'player_wind'))
            ,round_wind = eval(config.get(self.section, 'round_wind'))
            ,has_open_tanyao = eval(config.get(self.section, 'has_open_tanyao'))
            ,has_aka_dora = eval(config.get(self.section, 'has_aka_dora'))
            ,disable_double_yakuman = eval(config.get(self.section, 'disable_double_yakuman'))
            ,kazoe = eval(config.get(self.section, 'kazoe'))
            ,kiriage = eval(config.get(self.section, 'kiriage'))
            ,fu_for_open_pinfu = eval(config.get(self.section, 'fu_for_open_pinfu'))
            ,fu_for_pinfu_tsumo = eval(config.get(self.section, 'fu_for_pinfu_tsumo'))
            )
        return handconfig

    def main(self):
        # read config
        handconfig = self._read_config()
        
        # ライブラリに渡す用の前処理
        dora_idx = self._piname2index(self.dora_pi)
        win_idx = self._piname2index(self.win_pi)
        man, pin, sou, honors = self._get_list_pinum()
        pi_idx = TilesConverter.string_to_136_array(man=man, pin=pin, sou=sou, honors=honors)
        # 計算
        calculator = HandCalculator()
        melds = None
        result = calculator.estimate_hand_value(pi_idx, win_idx, melds=melds, dora_indicators=[dora_idx],
                                                config=handconfig)
        yaku = result.yaku
        han = result.han
        hu = result.fu
        parent_point = result.cost['main']
        child_point = result.cost['additional']

        return yaku, han, hu, parent_point, child_point

def create_return_txt(yaku, han, hu, parent_point, child_point):
    _yaku_txt = ''
    for x in yaku:
        _yaku_txt += str(x) + '\n'
    result_txt = '-----役一覧-----\n{}-----点数結果-----\n{}翻{}符\n親:{}点\n子:{}点'.\
    format(_yaku_txt, han, hu, parent_point, child_point)
    return result_txt