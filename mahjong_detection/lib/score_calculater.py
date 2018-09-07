import pandas as pd


def score_calculate(mark, point, flg_leader, flg_ron):
    """
    和了時の情報を受け取り、点数早見表を使って点数計算する関数
    : params mark       int : 符数
    : params point      int : 翻数
    : params flg_leader bool : 親かどうか
    : params flg_ron    bool : ロンならTrue、自摸ならFalse
    """
    
    result = [mark, point]
    
    # 親か否かで読み込む点数早見表を分岐
    if flg_leader:
        point_table = pd.read_csv('../data/point/point_for_leader.csv')
    else:
        point_table = pd.read_csv('../data/point/point_for_non_leader.csv')
    
    # ロンか自摸か
    if flg_ron:
        ron_point = int(point_table[(point_table['mark'] == result[0]) & (point_table['point'] == result[1])]['ron_point'])
        print('------------------------\nロン {0}点\n------------------------'.format(ron_point))
    else:
        drow_point_leader     = int(point_table[(point_table['mark'] == result[0]) & 
                                                 (point_table['point'] == result[1])]['drow_point_leader'])
        drow_point_non_leader = int(point_table[(point_table['mark'] == result[0]) & 
                                                 (point_table['point'] == result[1])]['drow_point_non_leader'])
        if flg_leader:
            print('------------------------\n自摸 子{1}点\n------------------------'.format(drow_point_leader, drow_point_non_leader))
        else:
            print('------------------------\n自摸 親{0}点 子{1}点\n------------------------'.format(drow_point_leader, drow_point_non_leader))
    