import random
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import sys
import collections

from src import init_individual
import openpyxl

from src.settings_xlsx import ExcelTableExtractor

# 評価関数をインポート
from src.evaluation_functions import (
    most_frequent_element_count,
    calculate_score_details,
    calculate_total_score,
    # get_past_out_n,  # 新しく追加
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
        past_out_n,
        col_name,
        col_class,
        weight1,
        weight2,
        weight3,
        weight4,
        COL_OUTPUT,
        COL_HOST,  # 追加: COL_HOST の取得
        custom_crossover_prob,
        shuffle_selection_prob
    ):
        self.names = names
        self.group_size = group_size
        self.population_size = population_size
        self.COL_OUTPUT = COL_OUTPUT
        self.COL_HOST = COL_HOST  # 追加: COL_HOST を保持

        # 遺伝的アルゴリズムパラメータ
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_indpb = mutation_indpb
        self.k_select_best = k_select_best
        self.tournsize = tournsize
        self.cxpb = cxpb

        # 追加のパラメータ
        self.df = df
        self.past_out_n = past_out_n
        self.col_name = col_name
        self.col_class = col_class
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        self.weight4 = weight4

        # 追加: 新しい確率変数
        self.custom_crossover_prob = custom_crossover_prob
        self.shuffle_selection_prob = shuffle_selection_prob

        self.target_counts = dict()

        # 遺伝的アルゴリズムのセットアップ
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_group", self.init_individual)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.attr_group
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 交叉と突然変異の登録
        self.toolbox.register("mate", self.random_crossover)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=self.mutation_indpb)

        self.toolbox.register("select_best", tools.selBest, k=self.k_select_best)  # 最も適応度が高い個体を選択
        self.toolbox.register(
            "select_tournament", tools.selTournament, tournsize=self.tournsize
        )  # トーナメント選択を実行
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
        rest_individuals = self.toolbox.select_tournament(
            population, k=len(population) - len(best_individuals)
        )
        # 選択された個体を結合し、全体としてk個体選ぶ
        return best_individuals + rest_individuals[: k - len(best_individuals)]

    def random_crossover(self, parent1, parent2):
        if random.random() < self.custom_crossover_prob:  # 変更: 0.7 → self.custom_crossover_prob
            return self.custom_crossover(parent1, parent2)
        else:
            return self.two_point_crossover(parent1, parent2)
               
    def custom_crossover(self, parent1, parent2):  
        # parent1とparent2に対してshuffle_selected_elementsを適用  
        child1 = self.shuffle_selected_elements(parent1)  
        child2 = self.shuffle_selected_elements(parent2)  

        # if random.random() < 0.5:
        #     # もう一度shuffle_selected_elementsを適用  
        #     child1 = self.shuffle_selected_elements(child1)  
        #     child2 = self.shuffle_selected_elements(child2)  

        return creator.Individual(child1), creator.Individual(child2)  
    
    def shuffle_selected_elements(self, arr):
        unique_elements = list(set(arr))
        # 最小値と最大値の範囲からランダムに2つの値を選ぶ  
        if random.random() < self.shuffle_selection_prob:  # 変更: 0.9 → self.shuffle_selection_prob
            selected_elements = random.sample(unique_elements, 2)
            val1, val2 = selected_elements[0], selected_elements[1]
        else:
            # 戦略的選択
            score_lists = calculate_score_details(
                self.df, arr, self.col_name, self.col_class, self.COL_OUTPUT, self.past_out_n
            )
            sums = [sum(sublist) for sublist in score_lists]
            top_two_indices = sorted(range(len(sums)), key=lambda i: sums[i], reverse=True)[:2]
            val1, val2 = top_two_indices[0]+1, top_two_indices[1]+1 

        # val1 と val2 のインデックスを取得  
        indices_val1 = [i for i, x in enumerate(arr) if x == val1]
        indices_val2 = [i for i, x in enumerate(arr) if x == val2]

        # val1 と val2 のインデックスを合わせる  
        all_indices = indices_val1 + indices_val2
        random.shuffle(all_indices)

        # シャッフルされたインデックスに基づいて新しい配列を作成  
        shuffled_arr = arr.copy()
        for i, idx in enumerate(indices_val1):
            shuffled_arr[idx] = arr[all_indices[i]]
        for i, idx in enumerate(indices_val2):
            shuffled_arr[idx] = arr[all_indices[len(indices_val1) + i]]

        return shuffled_arr
    
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
            self.df, individual, self.col_name, self.col_class, self.COL_OUTPUT, self.past_out_n
        )
        # 合計スコアを計算
        total_score = calculate_total_score(
            score_lists, self.weight1, self.weight2, self.weight3, self.weight4
        )

        return (total_score,)

    # アルゴリズムの実行
    def run(self):
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
        
        # 各グループ内でCOL_HOSTの値が小さい順にソートし、先頭の人のCOL_HOSTを+1カウント
        for group in final_groups:
            # グループをCOL_HOSTの値でソート
            sorted_group = sorted(
                group, 
                key=lambda name: self.df.loc[self.df[self.col_name] == name, self.COL_HOST].values[0]
            )
            # ソートされたグループを更新
            group[:] = sorted_group
            # 先頭の人のCOL_HOSTを+1カウント
            host_name = sorted_group[0]
            self.df.loc[self.df[self.col_name] == host_name, self.COL_HOST] += 1

        # final_dfの作成
        df_groups = pd.DataFrame(final_groups).transpose()
        df_groups.columns = [f'Group_{i+1}' for i in range(df_groups.shape[1])]
        df_scores = pd.DataFrame(scores_best_individual, index=['Score'])
        final_df = pd.concat([df_groups, df_scores], axis=0)

        # COL_HOSTをfinal_dfに追加
        host_counts = self.df.set_index(self.col_name)[self.COL_HOST]
        final_df = final_df.append(host_counts, ignore_index=True)
        final_df.index = list(final_df.index[:-1]) + ['COL_HOST']
        
        return best_individual, final_df

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
        groups = dict(sorted(groups.items()))
        print(self.names)
        print(best_individual)
        score_lists = calculate_score_details(
            self.df, best_individual, self.col_name, self.col_class, self.COL_OUTPUT, self.past_out_n
        )
        print(score_lists)

        final_groups = [group for group in groups.values() if len(group) >= 1]
        print(groups)
        for group_num, members in enumerate(final_groups, start=1):
            print(f"Group {group_num}: {members}")
        return score_lists, final_groups


if __name__ == "__main__":
    import time
    # 実行前の時間を記録
    start_time = time.time()
    scores = []
    
    for i in range(1):
        # エクセルファイルからの入力値の取得
        COL_OUTPUT = "グループ分け"
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
        COL_HOST = extractor.get_value("COL_HOST")  # 追加: COL_HOST の取得

        # テーブル2
        extractor.table = "テーブル2"
        MINIMIZE_DUPLICATE_AFFILIATIONS = extractor.get_value(
            "MINIMIZE_DUPLICATE_AFFILIATIONS"
        )
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
        custom_crossover_prob = extractor.get_value("custom_crossover_prob")
        shuffle_selection_prob = extractor.get_value("shuffle_selection_prob")

        extractor.close_workbook()

        # 初期データ処理
        # dfから過去のグループ分けの数を取得する関数
        def get_past_out_n(df, COL_OUTPUT):
            past_out_columns = df.filter(like=COL_OUTPUT).columns
            return sum(col[len(COL_OUTPUT):].isdigit() for col in past_out_columns)
        
        df_origin = pd.read_excel(EXCEL_NAME, sheet_name=SHEET_NAME)
        
        # COL_HOST 列の存在確認
        if COL_HOST not in df_origin.columns:
            print(f"エラー: '{COL_HOST}' 列が '{EXCEL_NAME}' のシート '{SHEET_NAME}' に存在しません。")
            sys.exit()
        
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
        past_out_n = get_past_out_n(df, COL_OUTPUT)

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
            past_out_n=past_out_n,
            col_name=COL_NAME,
            col_class=COL_CLASS,
            weight1=WEIGHT1,
            weight2=WEIGHT2,
            weight3=WEIGHT3,
            weight4=WEIGHT4,
            COL_OUTPUT=COL_OUTPUT,
            COL_HOST=COL_HOST,  # 追加: COL_HOST を渡す
            custom_crossover_prob=custom_crossover_prob,       # 変更: extractor から取得した値を使用
            shuffle_selection_prob=shuffle_selection_prob       # 変更: extractor から取得した値を使用
        )
        best_individual, final_df = ga.run()
        
        scores.append(ga.evaluate(best_individual)[0])
        
        # 出力処理
        print(best_individual)
        print(final_df)
        # past_out_n = get_past_out_n(df_origin, COL_OUTPUT)
        # new_col = COL_OUTPUT+str(past_out_n+1)
        # df[new_col] = best_individual
        # group_df = pd.concat([df_origin, df[new_col]],axis=1)

        # # 既存のExcelファイルのシートをすべて読み込む
        # excel_file = pd.ExcelFile(EXCEL_NAME)
        # sheet_names = excel_file.sheet_names
        # sheet_dict = {sheet: excel_file.parse(sheet) for sheet in sheet_names}

        # # 新しいデータを追加
        # sheet_dict[SHEET_NAME] = group_df
        # sheet_dict[new_col] = final_df

        # # すべてのシートを同じExcelファイルに書き込む
        # with pd.ExcelWriter(EXCEL_NAME) as writer:
        #     for sheet_name, dataframe in sheet_dict.items():
        #         dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
    print(scores)
    end_time = time.time()

    # 実行時間を計算
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")
