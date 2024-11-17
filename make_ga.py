# encoding:utf-8

import numpy as np
import pandas as pd
import random
from collections import Counter
from tqdm import tqdm
import sys
from deap import base, creator, tools, algorithms

# ランダムにグループを作成する関数
def get_group_dicts(name_list, group_num, group_people_min):
    group_dict = {i: [] for i in range(group_num)}
    for i in range(group_num):
        random_selection = random.sample(name_list, group_people_min)
        name_list = [item for item in name_list if item not in random_selection]
        group_dict[i] = random_selection
    
    # 余った人の割り当て処理
    for name in name_list:
        random_n = random.randint(0, group_num - 1)
        group_dict[random_n].append(name)
    return group_dict

# スコア計算
# 同じ課（class）の人と重複している人数をカウントするスコア
def get_score(df, group_dict, target_col, weight1, weight2):
    target_dict = df.set_index('氏名')[target_col].to_dict()
    
    score_list = []
    for group_members in group_dict.values():
        element_list = [target_dict.get(name, None) for name in group_members if target_dict.get(name, None) is not None]
        element_counts = Counter(element_list)
        max_count = max(element_counts.values()) if element_counts else 0
        score_list.append(weight1 * (max_count - 1) + weight2 * sum(element_counts.values()))
    return sum(score_list)

# 遺伝的アルゴリズムの初期設定
def genetic_algorithm(df, population_size=100, generations=50, cxpb=0.5, mutpb=0.2):
    name_list = list(df['氏名'])
    total_people = len(name_list)
    group_people_min = 4
    group_num = total_people // group_people_min

    # 適応度を最小化するための設定
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", dict, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # 個体生成関数
    def create_individual():
        group_dict = get_group_dicts(name_list.copy(), group_num, group_people_min)
        individual = creator.Individual(group_dict)
        return individual
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 評価関数
    def evaluate(individual):
        return get_score(df, individual, target_col='class', weight1=1000, weight2=4),

    # 交叉関数の修正（辞書型個体に対応）
    def mate(ind1, ind2):
        keys = list(ind1.keys())
        cxpoint1, cxpoint2 = sorted(random.sample(range(len(keys)), 2))
        for key in keys[cxpoint1:cxpoint2]:
            ind1[key], ind2[key] = ind2[key], ind1[key]
        return ind1, ind2

    toolbox.register("mate", mate)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # 初期個体群生成
    population = toolbox.population(n=population_size)

    # 遺伝的アルゴリズムの実行
    algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, 
                        stats=None, halloffame=None, verbose=True)

    # 最良の個体を返す
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

if __name__ == "__main__":
    # データ読み込み
    EXCEL_NAME = "names.xlsx"
    SHEET_NAME = "名簿"
    df_origin = pd.read_excel(EXCEL_NAME, sheet_name=SHEET_NAME)
    df = df_origin.copy()
    df = df[df['対象'] == 1]
    df['class'] = df['部・室'] + df['課']
    
    # 遺伝的アルゴリズムの実行
    best_group_dict = genetic_algorithm(df)

    # 結果の表示
    print("最適なグループ分け:")
    for group, members in best_group_dict.items():
        print(f"グループ {group + 1}: {', '.join(members)}")
