def _find_max_m_solution(A, p, extension_factor):
    max_m = 0
    best_solution = None

    # extension_factor を加味した A の調整
    # A' = A - (p - 1) * extension_factor * p
    A_adjusted = A - (p - 1) * extension_factor * p

    for n in range(0, A):
        if (A_adjusted - (p - 1) * n) % p == 0:
            m = (A_adjusted - (p - 1) * n) // p
            if m > 0 and m > max_m:
                max_m = m
                best_solution = (p, m, n, extension_factor)

    return best_solution


def get_genome(all_people, n_per_group, hi_lo, extension_factor=0):
    if hi_lo == 0:
        solution = _find_max_m_solution(all_people, n_per_group, extension_factor)
        if solution is None:
            return []
        _, m, n, extension_factor = solution

        genome = []
        # m 個の p グループ
        for i in range(1, m + 1):
            genome += ([i] * n_per_group)

        # n 個の (p-1) グループ
        for i in range(m + 1, m + n + 1):
            genome += ([i] * (n_per_group - 1))

        # extension_factor * p 個の (p-1) グループを追加
        start_index = m + n + 1
        for i in range(start_index, start_index + extension_factor * n_per_group):
            genome += ([i] * (n_per_group - 1))

    elif hi_lo == 1:
        genome = list(range(1, all_people // n_per_group + 1)) * n_per_group
        genome += list(range(1, all_people - (all_people // n_per_group) * n_per_group + 1))

    return genome


if __name__ == "__main__":
    A = 45
    p = 4
    extension_factor = 1  # 0～2程度で設定可能

    solution = _find_max_m_solution(A, p, extension_factor)
    if solution:
        print(f"m が最大の解: p = {solution[0]}, m = {solution[1]}, n = {solution[2]}, extension_factor = {solution[3]}")
    else:
        print("解が見つかりませんでした。")

    genome = get_genome(A, p, hi_lo=0, extension_factor=extension_factor)
    print(genome)
    print(len(genome))
