import itertools
import time
import pandas as pd
import sys
from separate_group import GeneticAlgorithm
from src.settings_xlsx import ExcelTableExtractor
from collections import Counter
from src.evaluation_functions import calculate_score_details, calculate_total_score

def get_past_out_n(df, COL_OUTPUT):
    past_out_columns = df.filter(like=COL_OUTPUT).columns
    return sum(col[len(COL_OUTPUT):].isdigit() for col in past_out_columns)

def read_settings():
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

    # 固定値（最適化する変数以外）
    GENERATIONS = extractor.get_value("generations")
    MUTATION_INDPB = extractor.get_value("mutation_indpb")

    # 取得した値を辞書として返す
    settings = {
        "EXCEL_NAME": EXCEL_NAME,
        "SHEET_NAME": SHEET_NAME,
        "COL_NAME": COL_NAME,
        "COL_TARGET": COL_TARGET,
        "COL_CLASS": COL_CLASS,
        "MINIMIZE_DUPLICATE_AFFILIATIONS": MINIMIZE_DUPLICATE_AFFILIATIONS,
        "GROUP_SIZE": GROUP_SIZE,
        "ADJUST_GROUP_SIZE_FOR_REMAINDER": ADJUST_GROUP_SIZE_FOR_REMAINDER,
        "WEIGHT1": WEIGHT1,
        "WEIGHT2": WEIGHT2,
        "WEIGHT3": WEIGHT3,
        "WEIGHT4": WEIGHT4,
        "GENERATIONS": GENERATIONS,
        "MUTATION_INDPB": MUTATION_INDPB
    }

    extractor.close_workbook()
    return settings

def main():
    # 最適化する変数の候補値を定義
    population_size_options = [150, 300]
    mutation_rate_options = [0.1, 0.2]
    k_select_best_options = [2, 5]
    tournsize_options = [3, 5]
    cxpb_options = [0.7, 0.9]
    custom_crossover_prob_options = [0.2, 0.6]
    # population_size_options = [10]
    # mutation_rate_options = [0.1]
    # k_select_best_options = [2]
    # tournsize_options = [3]
    # cxpb_options = [0.7, 0.9]
    # custom_crossover_prob_options = [0.2, 0.6]

    # 全ての組み合わせを生成
    parameter_combinations = list(itertools.product(
        population_size_options,
        mutation_rate_options,
        k_select_best_options,
        tournsize_options,
        cxpb_options,
        custom_crossover_prob_options
    ))

    settings = read_settings()

    # 結果を保存するリスト
    results = []

    for idx, (population_size, mutation_rate, k_select_best, tournsize, cxpb, custom_crossover_prob) in enumerate(parameter_combinations):
        print(f"試行 {idx+1}/{len(parameter_combinations)}: population_size={population_size}, mutation_rate={mutation_rate}, k_select_best={k_select_best}, tournsize={tournsize}, cxpb={cxpb}, custom_crossover_prob={custom_crossover_prob}")

        start_time = time.time()

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
        generations = settings["GENERATIONS"]
        mutation_indpb = settings["MUTATION_INDPB"]

        extractor.close_workbook()

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
            custom_crossover_prob=custom_crossover_prob,
            shuffle_selection_prob=0.9  # 固定値
        )

        best_individual, final_df = ga.run()

        end_time = time.time()
        execution_time = end_time - start_time

        # 最終スコアの計算
        score = ga.evaluate(best_individual)[0]

        # 結果を保存
        results.append({
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "k_select_best": k_select_best,
            "tournsize": tournsize,
            "cxpb": cxpb,
            "custom_crossover_prob": custom_crossover_prob,
            "execution_time": execution_time,
            "final_score": score
        })

        print(f"実行時間: {execution_time:.6f} 秒, 最終スコア: {score}\n")

    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)

    # 最終スコアが最小の組み合わせを取得
    best_score_row = results_df.loc[results_df["final_score"].idxmin()]

    # 実行時間が最小で、かつ最終スコアが上位50%以内の組み合わせを取得
    median_score = results_df["final_score"].median()
    filtered_df = results_df[results_df["final_score"] <= median_score]
    best_time_row = filtered_df.loc[filtered_df["execution_time"].idxmin()]

    # 結果を表示
    print("最終スコアが最も小さい組み合わせ:")
    print(best_score_row)

    print("\n実行時間が最も少ないかつ、最終スコアが上位50%以内の組み合わせ:")
    print(best_time_row)

    # 結果をExcelに保存
    results_df.to_excel("optimization_results.xlsx", index=False)
    print("\n全ての結果を 'optimization_results.xlsx' に保存しました。")

if __name__ == "__main__":
    main()
