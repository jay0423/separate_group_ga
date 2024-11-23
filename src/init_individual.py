# このプログラムは、方程式 A = p * m + (p-1) * n の解のうち
# m が最大となる解を探索します。p は固定値として指定され、
# A は自然数として与えられます。n の範囲を探索し、条件を満たす m を求めます。
# 最終的に m が最大の組み合わせを出力します。
# p:1グループあたりの人数 , m: グループ数1, n: グループ数2

def _find_max_m_solution(A, p):
    max_m = 0
    best_solution = None

    for n in range(1, A):
        if (A - (p - 1) * n) % p == 0:
            m = (A - (p - 1) * n) // p
            if m > 0 and m > max_m:
                max_m = m
                best_solution = (p, m, n)

    return best_solution


def get_genome(all_people, n_per_group, hi_lo="lo"):
    if hi_lo == "lo":
        _, m, n = _find_max_m_solution(all_people, n_per_group) # 総グループ数
        genome = []
        for i in range(1, m+1):
            genome = genome + ([i] * n_per_group)
        for i in range(m+1, m+n+1):
            genome = genome + ([i] * (n_per_group-1))
    elif hi_lo == "hi":
        genome = list(range(1, all_people // n_per_group + 1)) * n_per_group
        genome = genome + list(range(1, all_people - (all_people // n_per_group) * n_per_group + 1))
    return genome


if __name__ == "__main__":
    A = 45  # 例えば A = 20 の場合
    p = 4   # p の固定値
    solution = _find_max_m_solution(A, p)
    if solution:
        print(f"m が最大の解: p = {solution[0]}, m = {solution[1]}, n = {solution[2]}")
    else:
        print("解が見つかりませんでした。")
        
    genome = get_genome(A, p, hi_lo="hi")
    print(genome)
    print(len(genome))