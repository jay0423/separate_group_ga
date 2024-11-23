import random
from collections import Counter

def crossover_preserve_elements(parent1, parent2):
    # 各要素の出現回数を保持
    count1 = Counter(parent1)
    count2 = Counter(parent2)
    
    # 親ゲノムの長さを確認
    length = len(parent1)
    
    # 子供ゲノムを初期化
    child1 = [None] * length
    child2 = [None] * length
    
    # 交叉点をランダムに決定
    crossover_point = random.randint(1, length - 1)

    # 交叉点までを親1、以降を親2からコピー
    child1[:crossover_point] = parent1[:crossover_point]
    child2[:crossover_point] = parent2[:crossover_point]

    # 各要素の出現回数を更新
    count1.subtract(child1[:crossover_point])
    count2.subtract(child2[:crossover_point])

    # 残りの部分に要素を埋める
    for i in range(crossover_point, length):
        for count, child, remaining_counts in zip([count2, count1], [child1, child2], [count1, count2]):
            available_elements = [element for element, count_remaining in count.items() if count_remaining > 0]
            if available_elements:
                chosen_element = random.choice(available_elements)
                child[i] = chosen_element
                count[chosen_element] -= 1

    return child1, child2

# サンプルテスト
parent1 = [1, 1, 2, 2, 2, 3, 3, 3]
parent2 = [2, 3, 2, 1, 1, 2, 3, 3]
child1, child2 = crossover_preserve_elements(parent1, parent2)
print("Child 1:", child1)
print("Child 2:", child2)
