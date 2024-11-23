# このプログラムは、方程式 A = p * m + (p-1) * n の解のうち
# m が最大となる解を探索します。p は固定値として指定され、
# A は自然数として与えられます。n の範囲を探索し、条件を満たす m を求めます。
# 最終的に m が最大の組み合わせを出力します。

def find_max_m_solution(A, p):
    max_m = 0
    best_solution = None

    for n in range(1, A):
        if (A - (p - 1) * n) % p == 0:
            m = (A - (p - 1) * n) // p
            if m > 0 and m > max_m:
                max_m = m
                best_solution = (p, m, n)

    return best_solution

A = 57  # 例えば A = 20 の場合
p = 4   # p の固定値
solution = find_max_m_solution(A, p)
if solution:
    print(f"m が最大の解: p = {solution[0]}, m = {solution[1]}, n = {solution[2]}")
else:
    print("解が見つかりませんでした。")
