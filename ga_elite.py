import numpy as np
import random
import matplotlib.pyplot as plt

object_w = [5,7,8,10,10,3,8,9,1,4] # 持ち物の重量を定義
object_c = [5,7,5,2,8,13,5,10,1,3] # 持ち物の価値を定義
max_weight = 30 # 重さの上限
epoch = 10 # 世代数

def roulette(fitness_list): # 適応度比例選択の関数
    print("---適応度比例選択---")
    total_fitness = np.sum(fitness_list)
    roulette = np.zeros(len(fitness_list))
    for i in range(len(fitness_list)):
        roulette[i] = fitness_list[i]/total_fitness # 適応度関数
    choiced = np.random.choice(len(roulette), 2, replace=False, p=roulette)
    return choiced

def crossvar(parent1,parent2): # 交叉の関数
    print("---交叉---")
    cross_point = random.randrange(len(parent1)) # 交叉点をランダムに決定
    print(f"交叉点は{cross_point}（添え字が0からなので注意）")
    child1 = parent1[:cross_point] # 親1の交叉点以降を分離
    child2 = parent2[:cross_point] # 親2の交叉点以降を分離
    parent1_b = parent1[cross_point:] # 親１の交叉点までを分離
    parent2_b = parent2[cross_point:] # 親2の交叉点までを分離
    child1 = np.insert(child1,cross_point,parent2_b)
    child2 = np.insert(child2,cross_point,parent1_b)
    # 決定した交叉点の前後で遺伝子座を分離し２つの親の遺伝子座を交換する
    print(f"交叉後の子1は=>{child1}")
    print(f"交叉後の子2は=>{child2}")
    return child1, child2

def mutation(child1,child2,prob):
    print("---突然変異---")
    mutation_c1 = np.random.binomial(1,prob,1)
    print("子2について突然変異を行うか(1なら行う，0なら行わない)=>",mutation_c1[0])
    mutation_c2 = np.random.binomial(1,prob,1)
    print("子2について突然変異を行うか(1なら行う，0なら行わない)=>",mutation_c2[0])
    if mutation_c1[0]:
        mutation_point = random.randrange(10)
        print("子1の突然変異させる場所（添え字が０からなので注意）",mutation_point)
        if child1[mutation_point]==1:
            child1[mutation_point] = 0
        else:
            child1[mutation_point] = 1
        print("突然変異後の子1=>",child1)
    elif mutation_c2[0]:
        mutation_point = random.randrange(10)
        print("子2の突然変異させる場所（添え字が０からなので注意）",mutation_point)
        if child2[mutation_point]==1:
            child2[mutation_point] = 0
        else:
            child2[mutation_point] = 1
        print("突然変異後の子2=>",child2)

sum_w_list = [] # 個体ごとの重量の総和を格納するリスト
sum_c_list = [] # 個体ごとの価値の総和を格納するリスト
fitness_list = [] # 適応度を格納するリスト
gene_list = [] # 生成した遺伝子座を格納するリスト
plt_mean_fitness = [] # 平均適応度保存用リスト（棒グラフ作成に使用）
plt_max_fitness = [] # 各世代の最大適応度保存用リスト（棒グラフ作成に使用）
max_fitness = -9999 # エリート保存戦略用．適応度の最大値が更新されるたびに値が更新される
elite_gene = None # エリート用の遺伝子座格納変数
elite_index = None # エリートの個体インデックス

# 初期集団の生成
print("========第１世代========")
for n in range(10):
    gene = np.random.binomial(1,0.5,10) # 二項分布から遺伝子座となる二値変数生成する
    gene_list.append(gene)
    index = np.where(gene==1) # 遺伝子座の1のインデックスを取り出す
    sum_w = 0 # １個体での重量の総和
    sum_c = 0 # １個体での価値の総和
    fitness = 1 # 適応度を初期化
    for i in index[0]: 
        sum_w = sum_w + object_w[i] # 遺伝子座が１の持ち物の重量の総和を計算
        sum_c = sum_c + object_c[i] # 遺伝子座が１の持ち物の価値の総和を計算
    sum_w_list.append(sum_w)
    sum_c_list.append(sum_c)
    if sum_w <= max_weight: # 重量の上限を超えない場合は持ち物の価値を適応度にする
        fitness = sum_c   
    if max_fitness <= fitness: # エリート保存戦略
        max_fitness = fitness # 最大適応度の更新
        elite_gene = gene # の最大適応度を持つ個体を格納
        elite_index = n # 最大適応度が何番目(n)の個体か格納
    fitness_list.append(fitness)

    print(f"{n}番目の個体の遺伝子座=>{gene},重みの総和=>{sum_w},価値の総和=>{sum_c},適応度=>{fitness}")
print(f"第1世代でのエリートは{elite_index}番目の個体=>{elite_gene},適応度=>{max_fitness}")
print(f"平均適応度:{sum(fitness_list)/len(fitness_list)},最大適応度:{max(fitness_list)}")
plt_mean_fitness.append(sum(fitness_list)/len(fitness_list)) # １世代目の平均適応度をグラフ用のリストに格納
plt_max_fitness.append(max(fitness_list))
parent_index = roulette(fitness_list) # 親の個体インデックスをルーレット選択により決める
parent1 = gene_list[parent_index[0]] # 親１
parent2 = gene_list[parent_index[1]] # 親２
print(f"親に選ばれたのは個体{parent_index[0]}=>{parent1},適応度=>{fitness_list[parent_index[0]]}")
print(f"親に選ばれたのは個体{parent_index[1]}=>{parent2},適応度=>{fitness_list[parent_index[1]]}")
child1, child2 = crossvar(parent1,parent2) # 交叉後の子１子２
mutation(child1,child2,0.2) # 突然変異（突然変異率は0.2）


# 第２世代以降
for generation in range(2,epoch):
    print(f"========第{generation}世代========")
    sum_w_list = [] # 個体ごとの重量の総和を格納するリスト
    sum_c_list = [] # 個体ごとの価値の総和を格納するリスト
    fitness_list = [] # 適応度を格納するリスト
    gene_list = [] # 生成した遺伝子座を格納するリスト
    parent_index=[]
    for n in range(10):
        if n==0: # 2世代目以降は0番目の個体に子1を指定
            gene = child1
            gene_list.append(gene)
        elif n == 1: # 2世代目以降は1番目の個体に子1を指定
            gene = child2
            gene_list.append(gene)
        elif n == 2: # 2世代目以降は3番目の個体はエリート個体を指定
            gene = elite_gene
            gene_list.append(gene)
        else:
            gene = np.random.binomial(1,0.5,10) # 二項分布から遺伝子座となる二値変数生成する
            gene_list.append(gene)
        index = np.where(gene==1) # 遺伝子座の1のインデックスを取り出す
        sum_w = 0 # １個体での重量の総和
        sum_c = 0 # １個体での価値の総和
        fitness = 1 # 適応度を初期化
        for i in index[0]: 
            sum_w = sum_w + object_w[i] # 持ち物の重量の総和を計算
            sum_c = sum_c + object_c[i] # 持ち物の価値の総和を計算
        sum_w_list.append(sum_w)
        sum_c_list.append(sum_c)
        if sum_w <= max_weight: # 重量の上限を超えない場合は持ち物の価値を適応度にする
            fitness = sum_c  
        if max_fitness <= fitness:
            max_fitness = fitness
            elite_gene = gene
            elite_index = n 
        fitness_list.append(fitness)
        print(f"{n}番目の個体の遺伝子座=>{gene},重みの総和=>{sum_w},価値の総和=>{sum_c},適応度=>{fitness}")
    print(f"第{generation}世代でのエリートは{elite_index}番目の個体=>{elite_gene},適応度=>{max_fitness}")
    print(f"平均適応度:{sum(fitness_list)/len(fitness_list)},最大適応度:{max(fitness_list)}")
    plt_mean_fitness.append(sum(fitness_list)/len(fitness_list))
    plt_max_fitness.append(max(fitness_list))
    parent_index = roulette(fitness_list) # 親の個体インデックスをルーレット選択により決める
    parent1 = gene_list[parent_index[0]]
    parent2 = gene_list[parent_index[1]]
    print(f"親に選ばれたのは個体{parent_index[0]}=>{parent1},適応度=>{fitness_list[parent_index[0]]}")
    print(f"親に選ばれたのは個体{parent_index[1]}=>{parent2},適応度=>{fitness_list[parent_index[1]]}")
    child1, child2 = crossvar(parent1,parent2)
    mutation(child1,child2,0.2)

plt_generation = range(epoch-1) # 棒グラフ用（横軸：世代）
fig = plt.figure()
plt.xlabel('Generation')
plt.ylabel('Fitness_Mean')
plt.plot(plt_generation, plt_mean_fitness, marker="o", color = "red")
plt.show()
plt.ylabel('Fitness_Max')
plt.plot(plt_generation, plt_max_fitness, marker="o", color = "Blue")
plt.show()
#np.save("elite_mean.npy",plt_mean_fitness)
#np.save("elite_max.npy",plt_max_fitness)
#np.save("generation.npy",plt_generation)
