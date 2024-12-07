# data_processor.py
import sys
import pandas as pd

class DataProcessor:
    def __init__(self, EXCEL_NAME, SHEET_NAME, COL_NAME, COL_TARGET, COL_CLASS, COL_OUTPUT, COL_HOST):
        self.EXCEL_NAME = EXCEL_NAME
        self.SHEET_NAME = SHEET_NAME
        self.COL_NAME = COL_NAME
        self.COL_TARGET = COL_TARGET
        self.COL_CLASS = COL_CLASS
        self.COL_OUTPUT = COL_OUTPUT
        self.COL_HOST = COL_HOST

    def get_past_out_n(self, df_origin):
        past_out_columns = df_origin.filter(like=self.COL_OUTPUT).columns
        return sum(col[len(self.COL_OUTPUT):].isdigit() for col in past_out_columns)

    def prepare_data(self):
        df_origin = pd.read_excel(self.EXCEL_NAME, sheet_name=self.SHEET_NAME)
        df = df_origin.copy()
        df = df[df[self.COL_TARGET] == 1]
        if len(df) != len(set(list(df[self.COL_NAME]))):
            print("名前が重複している人がいます。")
            sys.exit()

        classes = list(set(df[self.COL_CLASS]))
        numbered_dict = {value: index for index, value in enumerate(classes)}
        df[self.COL_CLASS] = df[self.COL_CLASS].replace(numbered_dict)
        past_out_n = self.get_past_out_n(df_origin)

        return df, df_origin, past_out_n

    def finalize_data(self, df, df_origin, best_individual, final_df):
        # 更新されたCOL_HOSTをdfに反映済みのため、それをdf_originにも反映する
        for idx, row in df.iterrows():
            name = row[self.COL_NAME]
            host_count = row[self.COL_HOST]
            df_origin.loc[df_origin[self.COL_NAME] == name, self.COL_HOST] = host_count

        past_out_n = self.get_past_out_n(df_origin)
        new_col = self.COL_OUTPUT + str(past_out_n + 1)
        df_origin[new_col] = None
        df_origin.loc[df_origin[self.COL_TARGET] == 1, new_col] = best_individual
        group_df = df_origin.copy()

        # final_dfの加工処理 ---------------------------------------
        # 1. indexを"グループ1"~"グループn"にする
        final_df.index = [f"グループ{i}" for i in range(1, len(final_df) + 1)]

        # カラムリスト取得
        cols = list(final_df.columns)
        col_len = len(cols)

        if col_len == 0:
            # カラムがない場合は特になにもしない
            pass
        else:
            # 2. 一番左の列を"リーダ"にする
            final_df.rename(columns={cols[0]: "リーダ"}, inplace=True)
            
            # リネーム後のカラム再取得
            cols = list(final_df.columns)

            # 右端からpast_out_n列を "重/グループi" に、(past_out_n+1)列目を "課重なり" にする
            # past_out_n列分の重/グループ が存在するためには col_len >= past_out_n + 2 (リーダ + 課重なり + 重/グループn)
            # 課重なり分を含め最低 past_out_n + 1 列は右側に確保されていることが前提
            # 仮にデータがそれに満たない場合は要件上不整合だが、ここではpast_out_nが適切に計算されることを前提とする。

            # 課重なり列のインデックス取得(右からpast_out_n+1番目)
            課重なり_idx = col_len - (past_out_n + 1)
            if 課重なり_idx <= 0:
                # リーダ以外に十分な列が無い場合は要件に合わないのでそのまま終了
                pass
            else:
                # 一旦リーダ以降は全て"メンバ"にする
                for i in range(1, col_len):
                    final_df.columns.values[i] = "メンバ"

                # 課重なり列をリネーム
                final_df.columns.values[課重なり_idx] = "課重なり"

                # 重/グループ列を右端からpast_out_n列分リネーム
                # 右端から1,2,...の順で重/グループiを割り当てる
                for i in range(1, past_out_n + 1):
                    # 右からi番目 => インデックスは col_len - i
                    final_df.columns.values[col_len - i] = f"重/グループ{past_out_n-i+1}"

        # -------------------------------------------------------

        excel_file = pd.ExcelFile(self.EXCEL_NAME)
        sheet_names = excel_file.sheet_names
        sheet_dict = {sheet: excel_file.parse(sheet) for sheet in sheet_names}

        sheet_dict[self.SHEET_NAME] = group_df
        sheet_dict[new_col] = final_df

        with pd.ExcelWriter(self.EXCEL_NAME) as writer:
            for sheet_name, dataframe in sheet_dict.items():
                dataframe.to_excel(writer, sheet_name=sheet_name, index=True)
