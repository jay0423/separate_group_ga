import random
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import sys

from src import init_individual

from src.settings_xlsx import ExcelTableExtractor

# 評価関数をインポート
from src.evaluation_functions import (
    most_frequent_element_count,
    calculate_score_details,
    calculate_total_score,
)

class GeneticAlgorithm:
    def __init__(
        self,
        names,
        group_size,
        population_size,
        generations,
        mutation_rate,
        mutation_indpb,
        k_select_best,
        tournsize,
        cxpb,
        df,
        col_name,
        col_class,
        col_output,
        past_out_n,
        weight1,
        weight2,
        weight3,
        weight4,
    ):
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

        # 追加のパラメータ
        self.df = df
        self.col_name = col_name
        self.col_class = col_class
        self.col_output = col_output
        self.past_out_n = past_out_n
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        self.weight4 = weight4

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

        self.toolbox.register("select_best", tools.selBest, k=self.k_select_best)  # 最も適応度が高い個体を選択
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
        return best_individuals + rest_individuals[: k - len(best_individuals)]

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
        score_lists = calculate_score_details(
            self.df, individual, self.col_name, self.col_class, self.col_output, self.past_out_n
        )
        # 合計スコアを計算
        total_score = calculate_total_score(
            score_lists, self.weight1, self.weight2, self.weight3, self.weight4
        )

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
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutation_rate,
            ngen=self.generations,
            stats=stats,
            verbose=True,
        )

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
        scores_best_individual = calculate_score_details(
            self.df, best_individual, self.col_name, self.col_class, self.col_output, self.past_out_n
        )
        print(scores_best_individual)

        final_groups = [group for group in groups.values() if len(group) >= 1]
        for group_num, members in enumerate(final_groups, start=1):
            print(f"Group {group_num}: {members}")
        return scores_best_individual, final_groups

    # 結果のエクセルへの出力
    def export_excel(self, scores_best_individual, final_groups):
        df1 = pd.DataFrame(final_groups)
        df2 = pd.DataFrame(scores_best_individual)
        print(pd.concat([df1, df2.T], axis=1))


if __name__ == "__main__":
    # エクセルファイルからの入力値の取得
    # ExcelTableExtractorクラスの使用例
    extractor = ExcelTableExtractor(
        filename="settings.xlsx",
        worksheet_name="settings",
        table_name="テーブル1",
        item_column_name="変数名",  # 「項目」列の名前を指定
        value_column_name="入力値",  # 「入力値」列の名前を指定
    )
    extractor.open_workbook()

    # テーブル1
    EXCEL_NAME = extractor.get_value("EXCEL_NAME")
    SHEET_NAME = extractor.get_value("SHEET_NAME")
    COL_NAME = extractor.get_value("COL_NAME")
    COL_TARGET = extractor.get_value("COL_TARGET")
    COL_CLASS = extractor.get_value("COL_CLASS")

    # テーブル2
    extractor.table = "テーブル2"
    MINIMIZE_DUPLICATE_AFFILIATIONS = extractor.get_value("MINIMIZE_DUPLICATE_AFFILIATIONS")
    GROUP_SIZE = extractor.get_value("GROUP_SIZE")
    ADJUST_GROUP_SIZE_FOR_REMAINDER = extractor.get_value("ADJUST_GROUP_SIZE_FOR_REMAINDER")

    # テーブル3
    extractor.table = "テーブル3"
    WEIGHT1 = extractor.get_value("WEIGHT1")
    WEIGHT2 = extractor.get_value("WEIGHT2")
    WEIGHT3 = extractor.get_value("WEIGHT3")
    WEIGHT4 = extractor.get_value("WEIGHT4")
    population_size = extractor.get_value("population_size")
    generations = extractor.get_value("generations")
    mutation_rate = extractor.get_value("mutation_rate")
    mutation_indpb = extractor.get_value("mutation_indpb")
    k_select_best = extractor.get_value("k_select_best")
    tournsize = extractor.get_value("tournsize")
    cxpb = extractor.get_value("cxpb")

    extractor.close_workbook()

    # 追加列名
    COL_OUTPUT = "グループ分け"  # グループ分け名

    # 重みづけ
    PAST_OUT_N = 3  ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 初期データ処理
    df_origin = pd.read_excel(EXCEL_NAME, sheet_name=SHEET_NAME)
    df = df_origin.copy()
    df = df[df[COL_TARGET] == 1]
    print(len(df), len(set(list(df[COL_NAME]))))
    if len(df) != len(set(list(df[COL_NAME]))):
        print("名前が重複している人がいます。")
        sys.exit()
    classes = list(set(df[COL_CLASS]))
    numbered_dict = {value: index for index, value in enumerate(classes)}
    df[COL_CLASS] = df[COL_CLASS].replace(numbered_dict)
    print(len(df))

    ga = GeneticAlgorithm(
        names=list(df[COL_NAME]),
        group_size=GROUP_SIZE,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        mutation_indpb=mutation_indpb,
        k_select_best=k_select_best,
        tournsize=tournsize,
        cxpb=cxpb,
        df=df,
        col_name=COL_NAME,
        col_class=COL_CLASS,
        col_output=COL_OUTPUT,
        past_out_n=PAST_OUT_N,
        weight1=WEIGHT1,
        weight2=WEIGHT2,
        weight3=WEIGHT3,
        weight4=WEIGHT4,
    )
    ga.run()