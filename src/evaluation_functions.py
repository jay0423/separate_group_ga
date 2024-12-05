import random
from collections import Counter, defaultdict

# リストの中から最も頻出な要素の数を返す。同じクラスの人数をカウントする関数
def most_frequent_element_count(lst):
    if not lst:
        return 0  # リストが空の場合、0を返す
    element_counts = Counter(lst)  # リスト内の要素の出現回数をカウント
    most_common = element_counts.most_common(1)  # 最も多い要素とそのカウントを取得
    return most_common[0][1] - 1

# dfから過去のグループ分けの数を取得する関数
def get_past_out_n(df, COL_OUTPUT):
    past_out_columns = [
        col for col in df.columns if col.startswith(COL_OUTPUT) and col[len(COL_OUTPUT):].isdigit()
    ]
    past_out_n = len(past_out_columns)
    return past_out_n

# グループリストと名前リストからグループ分けを行い、スコアを計算する関数
def calculate_score_details(df, group_list, col_name, col_class, COL_OUTPUT):
    past_out_n = get_past_out_n(df, COL_OUTPUT)
    name_list = list(df[col_name])
    grouped_names = defaultdict(list)
    for group, person in zip(group_list, name_list):
        grouped_names[group].append(person)
    
    target_cols = [col_class] + [COL_OUTPUT + str(i+1) for i in range(past_out_n)]
    score_lists = []
    
    for target_col in target_cols:
        target_dict = df.set_index(col_name)[target_col].to_dict()
        group_scores = []
        for group in grouped_names.values():
            element_list = [
                target_dict.get(name) for name in group if target_dict.get(name) is not None
            ]
            group_scores.append(most_frequent_element_count(element_list))
        score_lists.append(group_scores)
    
    return score_lists

# スコアリストを基に合計スコアを算出する関数
def calculate_total_score(score_lists, weight1, weight2, weight3, weight4):
    total_score = 0
    total_score += weight1 * max(score_lists[0])
    total_score += weight2 * sum(score_lists[0])
    
    for score_list in score_lists[1:]:
        total_score += weight3 * max(score_list)
        total_score += weight4 * sum(score_list)
    
    return total_score