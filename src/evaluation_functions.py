import numpy as np
from collections import defaultdict

# リストの中から最も頻出な要素の数を返す
def most_frequent_element_count(lst):
    unique, counts = np.unique(lst, return_counts=True)  # NumPyで要素とそのカウントを取得
    return counts.max() - 1  # 最大カウントを返す


# グループリストと名前リストからグループ分けを行い、スコアを計算する関数
def calculate_score_details(df, group_list, col_name, col_class, COL_OUTPUT, past_out_n):
    # past_out_n = get_past_out_n(df, COL_OUTPUT)
    name_array = df[col_name].to_numpy()  # 名前をNumPy配列に変換
    group_array = np.array(group_list)  # グループもNumPy配列に変換
    
    grouped_indices = defaultdict(list)
    for idx, group in enumerate(group_array):
        grouped_indices[group].append(idx)  # グループごとのインデックスを記録

    target_cols = [col_class] + [f"{COL_OUTPUT}{i+1}" for i in range(past_out_n)]
    target_arrays = {col: df[col].to_numpy() for col in target_cols}  # 必要な列をNumPy配列化

    score_lists = []
    for target_col in target_cols:
        target_data = target_arrays[target_col]
        group_scores = []
        for indices in grouped_indices.values():
            group_data = target_data[indices]  # グループに属するデータを抽出
            group_data = group_data[group_data != None]  # Noneを除外
            group_scores.append(most_frequent_element_count(group_data))
        score_lists.append(group_scores)
    
    return score_lists

# スコアリストを基に合計スコアを算出する関数
def calculate_total_score(score_lists, weight1, weight2, weight3, weight4):
    score_lists = np.array(score_lists)  # NumPy配列に変換
    total_score = weight1 * score_lists[0].max() + weight2 * score_lists[0].sum()
    for score_list in score_lists[1:]:
        total_score += weight3 * score_list.max() + weight4 * score_list.sum()
    return total_score
