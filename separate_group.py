import random
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import sys

# リストの中から最も頻出な要素の数を返す。同じクラスの人数をカウントする関数
def most_frequent_element_count(lst):
    if not lst:
        return 0  # リストが空の場合、0を返す
    element_counts = Counter(lst)  # リスト内の要素の出現回数をカウント
    most_common = element_counts.most_common(1)  # 最も多い要素とそのカウントを取得
    return most_common[0][1] - 1

# グループリストと名前リストからグループ分けを行い、スコアを計算する関数
def calculate_score_details(df, group_list):
    name_list = list(df["氏名"])
    grouped_names = defaultdict(list)
    for group, person in zip(group_list, name_list):
        grouped_names[group].append(person)
    
    target_cols = [COL_CLASS] + [COL_OUTPUT + str(i+1) for i in range(PAST_OUT_N)]
    score_lists = []
    
    for target_col in target_cols:
        target_dict = df.set_index("氏名")[target_col].to_dict()
        group_scores = []
        for group in grouped_names.values():
            element_list = [target_dict.get(name) for name in group if target_dict.get(name) is not None]
            group_scores.append(most_frequent_element_count(element_list))
        score_lists.append(group_scores)
    
    return score_lists

# スコアリストを基に合計スコアを算出する関数
def calculate_total_score(score_lists):
    total_score = 0
    total_score += WEIGHT3 * max(score_lists[0])
    total_score += WEIGHT4 * sum(score_lists[0])
    
    for score_list in score_lists[1:]:
        total_score += WEIGHT3 * max(score_list)
        total_score += WEIGHT4 * sum(score_list)
    
    return total_score

class GeneticAlgorithm:
    def __init__(self, names, group_size, population_size, generations, mutation_rate):
        self.names = names
        self.group_size = group_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        # 遺伝的アルゴリズムのセットアップ
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_group", random.randint, 1, len(self.names) // self.group_size)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_group, n=len(self.names))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        self.toolbox.register("select", tools.selBest)
        self.toolbox.register("evaluate", self.evaluate)

    # 個体を評価する関数（評価関数）
    def evaluate(self, individual):
        # スコアリストを計算
        score_lists = calculate_score_details(df, individual)
        # 合計スコアを計算
        total_score = calculate_total_score(score_lists)

        return (total_score,)  # Ensure to return a tuple

    # アルゴリズムの実行
    def run(self):
        random.seed(42)
        population = self.toolbox.population(n=self.population_size)

        # 統計情報の設定
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        # 遺伝的アルゴリズムの実行
        population, logbook = algorithms.eaSimple(population, self.toolbox, cxpb=0.7, mutpb=self.mutation_rate,
                                                  ngen=self.generations, stats=stats, verbose=True)

        # 最良の個体の表示
        best_individual = tools.selBest(population, k=1)[0]
        self.display_result(best_individual)

    # 結果の表示
    def display_result(self, best_individual):
        print("Best Individual:", best_individual)
        print("Groups:")
        groups = {}
        for i, group_number in enumerate(best_individual):
            if group_number not in groups:
                groups[group_number] = []
            groups[group_number].append(self.names[i])

        # グループサイズが4人または3人になるように調整
        final_groups = [group for group in groups.values() if len(group) >= 3]
        ungrouped_members = [member for group in groups.values() if len(group) < 3 for member in group]

        for group in final_groups:
            while len(group) < self.group_size and ungrouped_members:
                group.append(ungrouped_members.pop())

        for group_num, members in enumerate(final_groups, start=1):
            print(f"Group {group_num}: {members}")

if __name__ == "__main__":
    # 定数の定義
    EXCEL_NAME = "names.xlsx"
    SHEET_NAME = "名簿"

    # 列名
    COL_NAME = "氏名"
    COL_TARGET = "対象"
    COL_BU = "部・室"
    COL_KA = "課"
    # 追加列名
    COL_CLASS = "class"
    COL_OUTPUT = "グループ分け"  # グループ分け名

    # グループ分け設定
    GROUP_PEOPLE_MIN = 4  # 1グループあたりの最少人数（割り切れない場合はグループによって+1人）

    # 探索諸設定
    N = 1000

    # 重みづけ
    PAST_OUT_N = 3
    WEIGHT1 = 1000
    WEIGHT2 = 4
    WEIGHT3 = 2
    WEIGHT4 = 1

    # 初期データ処理
    df_origin = pd.read_excel(EXCEL_NAME, sheet_name=SHEET_NAME)
    df = df_origin.copy()
    df = df[df[COL_TARGET] == 1]
    print(len(df), len(set(list(df[COL_NAME]))))
    if len(df) != len(set(list(df[COL_NAME]))):
        print("名前が重複している人がいます。")
        sys.exit()
    df[COL_CLASS] = df[COL_BU] + df[COL_KA]
    classes = list(set(df[COL_CLASS]))
    numbered_dict = {value: index for index, value in enumerate(classes)}
    df[COL_CLASS] = df[COL_CLASS].replace(numbered_dict)

    ga = GeneticAlgorithm(
        names=list(df[COL_NAME]),
        group_size=4,
        population_size=100,
        generations=50,
        mutation_rate=0.2
    )
    ga.run()
