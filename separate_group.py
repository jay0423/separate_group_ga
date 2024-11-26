import random
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import sys
import collections

from src import init_individual


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
    total_score += WEIGHT1 * max(score_lists[0])
    total_score += WEIGHT2 * sum(score_lists[0])
    
    for score_list in score_lists[1:]:
        total_score += WEIGHT3 * max(score_list)
        total_score += WEIGHT4 * sum(score_list)
    
    return total_score

class GeneticAlgorithm:
    def __init__(self, names, group_size, population_size, generations, mutation_rate, mutation_indpb, k_select_best, tournsize, cxpb):
        self.names = names
        self.group_size = group_size
        self.population_size = population_size
        
        # 遺伝的アルゴリズムパラメータ
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_indpb = mutation_indpb
        self.k_select_best = k_select_best
        self.tournsize = tournsize
        self.cxpb = cxpb
        
        self.target_counts = dict()

        # 遺伝的アルゴリズムのセットアップ
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_group", self.init_individual)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_group)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", self.two_point_crossover)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=self.mutation_indpb)

        self.toolbox.register("select_best", tools.selBest, k=self.k_select_best)  # 最も適応度が高い個体を5つ選択
        self.toolbox.register("select_tournament", tools.selTournament, tournsize=self.tournsize)  # トーナメント選択を実行
        self.toolbox.register("select", self.combined_selection)
        # self.toolbox.register("select", tools.selRoulette)
        
        self.toolbox.register("evaluate", self.evaluate)

    # 個体を初期化する関数（リストをシャッフル）
    def init_individual(self):
        genome = init_individual.get_genome(len(self.names), self.group_size, hi_lo="lo")
        random.shuffle(genome)
        self.target_counts = Counter(genome)
        return genome

    def combined_selection(self, population, k):
        # 最も適応度が高い個体をいくつか選択
        best_individuals = self.toolbox.select_best(population)
        # 残りの個体をトーナメント選択で選択
        rest_individuals = self.toolbox.select_tournament(population, k=len(population) - len(best_individuals))
        # 選択された個体を結合し、全体としてk個体選ぶ
        return best_individuals + rest_individuals[:k - len(best_individuals)]

    def two_point_crossover(self, parent1, parent2):
        # 2点交叉
        cx_point1, cx_point2 = sorted(random.sample(range(1, len(parent1)), 2))
        child1 = parent1[:cx_point1] + parent2[cx_point1:cx_point2] + parent1[cx_point2:]
        child2 = parent2[:cx_point1] + parent1[cx_point1:cx_point2] + parent2[cx_point2:]

        # 子供のゲノムを修正
        child1 = self.fix_genome(child1, self.target_counts)
        child2 = self.fix_genome(child2, self.target_counts)

        return creator.Individual(child1), creator.Individual(child2)

    def fix_genome(self, genome, target_counts):
        # 各要素の出現回数をカウント
        count = Counter(genome)

        # 要素が足りない場合は追加、多い場合は削除
        for element, target_count in target_counts.items():
            if count[element] > target_count:
                # 多すぎる要素を減らす
                excess = count[element] - target_count
                for _ in range(excess):
                    genome.remove(element)
            elif count[element] < target_count:
                # 足りない要素を追加
                shortage = target_count - count[element]
                for _ in range(shortage):
                    # 足りない要素を追加する位置をランダムに決定
                    insert_position = random.randint(0, len(genome))
                    genome.insert(insert_position, element)
        return genome

    # 個体を評価する関数（評価関数）
    def evaluate(self, individual):
        # スコアリストを計算
        score_lists = calculate_score_details(df, individual)
        # 合計スコアを計算
        total_score = calculate_total_score(score_lists)

        return (total_score,)

    # アルゴリズムの実行
    def run(self):
        random.seed(42)
        population = self.toolbox.population(n=self.population_size)
        # 統計情報の設定
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        # 遺伝的アルゴリズムの実行
        population, logbook = algorithms.eaSimple(population, self.toolbox, cxpb=self.cxpb, mutpb=self.mutation_rate,
                                                  ngen=self.generations, stats=stats, verbose=True)
        # スコアが0で早期終了
        for gen, record in enumerate(logbook):
            if record["min"] <= 17:  # 最小スコアが0の場合
                print(f"Score reached 0 at generation {gen}. Terminating early.")
                break
        
        # 最良の個体の表示
        best_individual = tools.selBest(population, k=1)[0]
        scores_best_individual, final_groups = self.display_result(best_individual)
        self.export_excel(scores_best_individual, final_groups)

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
        scores_best_individual = calculate_score_details(df, best_individual)
        print(scores_best_individual)

        final_groups = [group for group in groups.values() if len(group) >= 1]
        for group_num, members in enumerate(final_groups, start=1):
            print(f"Group {group_num}: {members}")
        return scores_best_individual, final_groups

    # 結果のエクセルへの出力
    def export_excel(self, scores_best_individual, final_groups):
        df1 = pd.DataFrame(final_groups)
        df2 = pd.DataFrame(scores_best_individual)
        print(df1)
        print(df2)
        print(pd.concat([df1, df2.T], axis=1))

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
        population_size=700,
        generations=25,
        mutation_rate=0.1,
        mutation_indpb=0.2,
        k_select_best=5,
        tournsize=3,
        cxpb=0.8
    )
    ga.run()
