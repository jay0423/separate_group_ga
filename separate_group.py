import random
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import sys
import collections

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
        self.toolbox.register("attr_group", self.init_individual)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_group)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", self.crossover_preserve_elements)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.9)

        self.toolbox.register("select_best", tools.selBest, k=5)  # 最も適応度が高い個体を5つ選択
        self.toolbox.register("select_tournament", tools.selTournament, tournsize=3)  # トーナメント選択を実行
        self.toolbox.register("select", self.combined_selection)
        
        self.toolbox.register("evaluate", self.evaluate)

    # 個体を初期化する関数（リストをシャッフル）
    def init_individual(self):
        genome = list(range(1, len(self.names) // self.group_size + 1)) * self.group_size
        genome = genome + [1]
        random.shuffle(genome)
        print(genome)
        original_dict = dict(collections.Counter(genome))
        sorted_dict = dict(sorted(original_dict.items()))
        # print(sorted_dict)
        return genome[:len(self.names)]


    def combined_selection(self, population, k):
        # 最も適応度が高い個体をいくつか選択
        best_individuals = self.toolbox.select_best(population)
        # 残りの個体をトーナメント選択で選択
        rest_individuals = self.toolbox.select_tournament(population, k=len(population) - len(best_individuals))
        # 選択された個体を結合し、全体としてk個体選ぶ
        return best_individuals + rest_individuals[:k - len(best_individuals)]


    # 各要素の出現回数を維持したまま交叉を行う関数
    def crossover_preserve_elements(self, ind1, ind2):
        # 各要素の出現回数を保持
        count1 = Counter(ind1)
        count2 = Counter(ind2)
        
        # 親ゲノムの長さを確認
        length = len(ind1)
        
        # 子供ゲノムを初期化
        child1 = [None] * length
        child2 = [None] * length
        
        # 交叉点をランダムに決定
        crossover_point = random.randint(1, length - 1)

        # 交叉点までを親1、以降を親2からコピー
        child1[:crossover_point] = ind1[:crossover_point]
        child2[:crossover_point] = ind2[:crossover_point]

        # 各要素の出現回数を更新
        count1.subtract(child1[:crossover_point])
        count2.subtract(child2[:crossover_point])

        # 残りの部分に要素を埋める
        for i in range(crossover_point, length):
            available_elements1 = [element for element, count_remaining in count2.items() if count_remaining > 0]
            available_elements2 = [element for element, count_remaining in count1.items() if count_remaining > 0]
            
            child1[i] = random.choice(available_elements1)
            count2[child1[i]] -= 1

            child2[i] = random.choice(available_elements2)
            count1[child2[i]] -= 1

        return creator.Individual(child1), creator.Individual(child2)

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
        population, logbook = algorithms.eaSimple(population, self.toolbox, cxpb=0, mutpb=self.mutation_rate,
                                                  ngen=self.generations, stats=stats, verbose=True)

        # 最良の個体の表示
        best_individual = tools.selBest(population, k=1)[0]
        self.display_result(best_individual)

    # 結果の表示
    def display_result(self, best_individual):
        print("Best Individual:", best_individual)
        print(len(best_individual))
        print("Groups:")
        groups = {}
        for i, group_number in enumerate(best_individual):
            if group_number not in groups:
                groups[group_number] = []
            groups[group_number].append(self.names[i])
        print(self.names)
        print(best_individual)

        # グループサイズが4人または3人になるように調整
        final_groups = [group for group in groups.values() if len(group) >= 1]
        # ungrouped_members = [member for group in groups.values() if len(group) < 3 for member in group]

        # for group in final_groups:
        #     while len(group) < self.group_size and ungrouped_members:
        #         group.append(ungrouped_members.pop())

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
    print(len(df))

    ga = GeneticAlgorithm(
        names=list(df[COL_NAME]),
        group_size=4,
        population_size=1000,
        generations=50,
        mutation_rate=0.5
    )
    ga.run()
