import openpyxl

class ExcelTableExtractor:
    def __init__(self, filename, worksheet_name, table_name, item_column_name='変数名', value_column_name='入力値'):
        """
        クラスの初期化

        :param filename: 読み込むExcelファイルのパス
        :param worksheet_name: 対象のワークシート名
        :param table_name: 対象のテーブル名
        :param item_column_name: 「項目」列の名前
        :param value_column_name: 「入力値」列の名前
        """
        self.filename = filename
        self.worksheet_name = worksheet_name
        self.table_name = table_name
        self.item_column_name = item_column_name
        self.value_column_name = value_column_name
        self.workbook = None
        self.worksheet = None
        self._table = None  # プロパティの衝突を避けるために _table として定義
        self.headers = []
        self.data = []

    def open_workbook(self):
        """ワークブックを開き、ワークシートとテーブルを取得する"""
        self.workbook = openpyxl.load_workbook(self.filename)
        self.worksheet = self.workbook[self.worksheet_name]
        self.table = self.table_name  # プロパティを使用してテーブルを設定

    @property
    def table(self):
        """テーブルオブジェクトを取得"""
        return self._table

    @table.setter
    def table(self, value):
        """テーブルを設定し、データを再抽出"""
        if self.worksheet is None:
            raise ValueError("ワークシートが初期化されていません。先に open_workbook() を呼び出してください。")
        if isinstance(value, str):
            self.table_name = value
            self._table = self.worksheet.tables[self.table_name]
        else:
            self._table = value
            self.table_name = self._table.name
        self._extract_table_data()

    def _extract_table_data(self):
        """テーブルのデータを抽出してヘッダーとデータに分ける"""
        ref = self.table.ref
        data = list(self.worksheet[ref])
        self.headers = [cell.value for cell in data[0]]  # ヘッダー行
        self.data = data[1:]  # データ行

    def get_value(self, search_string):
        """
        指定した「項目」の「入力値」を取得する

        :param search_string: 検索する「項目」の文字列
        :return: 対応する「入力値」、見つからない場合は None
        """
        try:
            item_col_idx = self.headers.index(self.item_column_name)
            value_col_idx = self.headers.index(self.value_column_name)
        except ValueError as e:
            print('指定した列が見つかりません:', e)
            return None

        for row in self.data:
            item_cell = row[item_col_idx]
            if item_cell.value == search_string:
                value_cell = row[value_col_idx]
                return value_cell.value
        print('指定した項目が見つかりませんでした。')
        return None

    def close_workbook(self):
        """ワークブックを閉じる"""
        if self.workbook:
            self.workbook.close()