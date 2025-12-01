from __future__ import print_function
import sys
import itertools
import math
import gurobipy
from gurobipy import GRB, quicksum
import numpy as np
import random
import time
import sensor_data
import heapq
import networkx as nx
from scipy.spatial import distance_matrix

def generate_k_nearest_graph(num_nodes, max_k, max_edges, omega):
    np.random.seed(2026+omega)
    random.seed(2026+omega)
    # 生成随机点
    points = np.random.rand(num_nodes, 2)

    # 计算距离矩阵
    dist_matrix = distance_matrix(points, points)

    # 创建图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # 为每个节点选择k个最近的邻居并添加边
    for node in range(num_nodes):
        # 随机选择k值
        k = random.randint(1, max_k)

        # 获取距离当前节点最近的k个节点的索引
        nearest_neighbors = np.argsort(dist_matrix[node])[1:k + 1]

        # 添加边
        for neighbor in nearest_neighbors:
            if not G.has_edge(node, neighbor):
                G.add_edge(node, neighbor)

    # 确保连通性
    while not nx.is_connected(G):
        # 找到两个不连通的节点并添加边
        connected_components = list(nx.connected_components(G))
        u = random.choice(list(connected_components[0]))
        v = random.choice(list(connected_components[1]))
        G.add_edge(u, v)

    # 调整边的数量
    while G.number_of_edges() < max_edges:
        # 随机选择两个不相邻的节点并添加边
        u, v = random.sample(range(num_nodes), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    while G.number_of_edges() > max_edges:
        # 随机移除一条边，确保移除后图仍然连通
        edges = list(G.edges)
        random.shuffle(edges)
        for u, v in edges:
            G.remove_edge(u, v)
            if nx.is_connected(G):
                break
            else:
                G.add_edge(u, v)

    return G, points
def generate_random_paths(G, selected_nodes, max_k, num_paths, omega):
    random.seed(2025+omega)
    combination = [(u,v) for u in selected_nodes for v in selected_nodes if u != v]
    n = len(combination)
    paths = []
    k = [random.randint(1, max_k) for _ in range(n)]
    diff = num_paths - sum(k)
    # print(diff, k)
    while 1:
        if diff < 0:
            while diff != 0:
                i = random.randint(0,n-1)
                if k[i] > 0:
                    k[i] -= 1
                    diff += 1
        elif diff > 0:
            while diff != 0:
                i = random.randint(0,n-1)
                k[i] += 1
                diff -= 1
        # print(sum(k), k)
        i=0
        for u, v in combination:
            if k[i] > 0:
                path = list(nx.shortest_simple_paths(G, u, v))
                # print(path[:k[i]])
                paths.extend(path[:k[i]])
                k[i] = len(path[:k[i]])
            i+=1
        # print(len(paths))
        if len(paths) == num_paths:
            return paths
        else:
            diff = num_paths - sum(k)
            paths = []


Data = sensor_data.data()
net = 'RD'
if net == 'SF' or net == 'EM':
    if net == 'SF':
        W0 = Data.W_SF
    else:
        W0 = Data.W_EM
    random.seed(2024)
    number = 300
    Omega = 5
    ran = [[random.random() for i in range(len(W0[0]))] for omega in range(Omega)]
    index = [heapq.nlargest(number, ran[omega]) for omega in range(Omega)]
    for omega in range(Omega):
        print(len(index[omega]),index[omega])
    Womega = [[[W0[j][i] for i in range(len(W0[0])) if ran[omega][i] in index[omega]]
               for j in range(len(W0))] for omega in range(Omega)]
    N = len(Womega[0])
    R = len(Womega[0][0])
elif net == 'ND':
    W = [[1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
[1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0],
[1,1,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0],
[0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0],
[1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0],
[0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0],
[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
[0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0],
[0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    Omega = 1
    N = len(W)
    R = number = len(W[0])
    thmbound = {17:1, 16:1, 15:3, 14:5, 13:7, 12:9, 11:13, 10:17, 9:22}
elif net == 'PH':
    W = [[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
     [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
     [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
     [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]]
    Omega = 1
    N = len(W)
    R = number = len(W[0])
    thmbound = {7: 1, 8: 0}
elif net == 'RD':
    node = 10
    max_k = 5
    N = 20
    R = 60
    selected_OD = int(node / 2)
    Omega = 5
    Womega = [[[0 for r in range(R)] for i in range(N * 2)] for omega in range(Omega)]
    for omega in range(Omega):
        random.seed(2024 + omega)
        G, points = generate_k_nearest_graph(node, max_k, N, omega)

        # 随机选择N个节点
        selected_nodes = random.sample(list(G.nodes), selected_OD)

        # 生成R条路径
        paths = generate_random_paths(G, selected_nodes, max_k, R, omega)
        # 输出节点集合、边集合和路径
        nodes = list(G.nodes)
        edges = list(G.edges)
        for i in range(N):
            edges.append((edges[i][1], edges[i][0]))
        print("节点集合:", nodes)
        print("边集合:", edges)
        print("边的数量:", G.number_of_edges())
        print("选中的节点:", selected_nodes)
        print("生成的路径:", len(paths), paths)

        for r in range(R):
            for a in range(len(paths[r]) - 1):
                i = edges.index((paths[r][a], paths[r][a + 1]))
                Womega[omega][i][r] = 1
    N *= 2
# c = [0.5, 1]
# c = [10, 1]
M = N * 10
timeLimit = 3600*12
# softlimit = 3600*2
bestub = 100000

c = [2,1]
# 参数设置
num_locations = N  # 候选位置数量
# budget = 50  # 预算限制
# costs = np.random.randint(5, 20, num_locations)  # 每个位置的成本
# coverages = np.random.randint(10, 50, num_locations)  # 每个位置的覆盖范围
population_size = 50  # 种群大小
num_generations = 200  # 迭代次数
mutation_rate = 0.01  # 变异率


def generate_vectors(base_vector, num_samples, n, m):
    """
    基于一个N维0-1向量，生成S个新的N维0-1向量，保留部分为1的分量，
    保留的个数是随机的，且总的1的数量不超过m个。

    :param base_vector: 基础向量，其中某些分量为1，其他为0
    :param num_samples: 需要生成的新向量的数量
    :param n: 向量的维度
    :param m: 生成向量中1的最大数量
    :return: 生成的向量列表
    """
    # 找出基础向量中为1的分量的索引
    one_indices = np.where(base_vector == 1)[0]
    num_ones = len(one_indices)

    # 初始化一个空列表来存储生成的向量
    generated_vectors = []

    for _ in range(num_samples):
        # 随机确定保留的个数，范围从0到min(num_ones, m)
        num_retain = np.random.randint(0, min(num_ones, m[1]) + 1)

        # 随机选择num_retain个索引保留为1
        if num_retain > 0:
            retain_indices = np.random.choice(one_indices, size=num_retain, replace=False)
        else:
            retain_indices = []

        # 随机生成一个N维向量，初始为0
        new_vector = np.zeros(n, dtype=int)

        # 将随机选择的分量设置为1
        new_vector[retain_indices] = 1

        # 随机生成剩余的1，使得总的1的数量不超过m
        remaining_ones = m[1] - num_retain
        remaining_ones_low = max(m[0] - num_retain,0)
        num_ones_to_keep = np.random.randint(remaining_ones_low, remaining_ones + 1)
        if num_ones_to_keep > 0:
            remaining_indices = np.random.choice(np.setdiff1d(np.arange(n), retain_indices), size=num_ones_to_keep, replace=False)
            new_vector[remaining_indices] = 1

        generated_vectors.append(new_vector)
    return np.array(generated_vectors)



# 初始化种群
def initialize_population(base_vector, size, num_locations,m):
    return generate_vectors(base_vector, size, num_locations,m)

# 计算适应度'

def rank(individual, label, id = None):
    if id != None:
        individual[id] = 0
    H = [[W[i][r]* label[r] for r in range(R)] for i in range(N)]
    rw = np.linalg.matrix_rank(H)
    if id != None:
        individual[id] = 1
    return rw

def differentiation(individual, id = None):
    if id != None:
        individual[id] = 0
    label = []
    # print(id,individual)
    # print('Wu = ')
    # print(np.array([[W[i][r] * individual[i] for r in range(R)] for i in range(N)]))
    for r in range(R):
        # print(f'route {r}=',sum(W[i][r] * individual[i] for i in range(N)))
        if sum(W[i][r] * individual[i] for i in range(N)) > 0:
            temp = sum([ min(sum([np.abs(W[i][r]-W[i][s]) * individual[i] for i in range(N)]), 1)
                                for s in range(R) if s!=r])
        else:
            temp = 0
        label.append(temp)
    if id != None:
        individual[id] = 1
    return np.array(label)

def differentiation_index(individual, id=None, cover=0):
    if id != None:
        individual[id] = 0

    label = [[min(1,sum([np.abs(W[i][r] - W[i][s]) * individual[i] for i in range(N)])) for s in range(R)] for r in range(R)]
    # print([sum(label[r]) for r in range(R)])
    if cover == 0:
        z = [0 if sum(label[r]) == R - 1 else 1 for r in range(R)]
        if id != None:
            individual[id] = 1
    else:
        label2 = [min(1,sum([W[i][r] * individual[i] for i in range(N)])) for r in range(R)]
        z = [0 if sum(label[r]) == R - 1 and np.abs(label2[r] - 1) < 0.1 else 1 for r in range(R)]
        # print(z)
    return z

def fao_value(individual,id=None):
    n = differentiation(individual, id)
    # print(n)
    # print('n=',n)
    fao = sum([1/(R-n[r]) for r in range(R) if n[r]!=0])
    # print('fao = ',fao)
    return fao

def fitness(individual):
    # solution = update_solution(individual)
    z = differentiation_index(individual, cover=1)
    # return solution
    # print('z = ', z)
    # sorted_arr = sorted_arr[1:]
    rw = rank(individual, z)
    if np.abs(rw - sum(z)) <= 0.1:
        fao = fao_value(individual)
        total_cost = np.sum(individual) * c[0] + (R-fao) * c[1]
        return total_cost
    else:
        return c[0] * M  # 如果超过预算，适应度为0

def intial_feasible_solution():
    sc, obj = MIPmodel()
    arr =[]
    for i in range(N):
        if sc[i] > 0.9:
            arr.append([i, fao_value(sc,i)])
    arr = np.array(arr)
    sorted_indices = np.argsort(arr[:, 1])[::-1]
    sorted_arr = arr[sorted_indices]
    # print(sorted_arr)
    scnew = np.array([0 for i in range(N)])
    k = 1
    while len(sorted_arr) > 0:
        print(f'iteration {k} begins')
        # print('current sc is ', sorted_arr)
        # print(sc)
        k+=1
        i = int(sorted_arr[0,0])
        # fao = int(sorted_arr[0,1])
        z = differentiation_index(sc,id=i,cover=1)
        # print('z = ', z)
        # sorted_arr = sorted_arr[1:]
        rw = rank(sc, z, id=i)

        # print(i,sum(z),rw,R-fao)
        if np.abs(rw - sum(z)) <= 0.1:
            print(f'sc {i} can be removed as rank is {rw}')
            sc[i] = 0
            arr = np.array([[i, fao_value(sc,i)] for i in range(N) if sc[i]>0.9])
            sorted_indices = np.argsort(arr[:, 1])[::-1]
            sorted_arr = arr[sorted_indices]
            # z = differentiation(sc)
        else:
            print(f'sc {i} cannot be removed as rank is {rw}')
            scnew[i] = 1
            sorted_arr = sorted_arr[1:]
        print()

    # z = differentiation_index(scnew,cover=1)
    # print(scnew, z)
    # print(W)
    # rw = rank(scnew,z)
    fao = fao_value(scnew)
    # xp = passivelocation(scnew)
    # print(xp)
    bestfit = sum(scnew)*c[0] + (R-fao)*c[1]
    return scnew, bestfit

def update_solution(sc):
    z = differentiation_index(sc,cover=1)
    # print('z = ', z)
    # sorted_arr = sorted_arr[1:]
    rw = rank(sc, z)
    if np.abs(rw - sum(z)) <= 0.1:
        pass
    else:
        return np.append(sc, c[0] * M)
    arr = []
    for i in range(N):
        if sc[i] > 0.9:
            arr.append([i, fao_value(sc,i)])
    arr = np.array(arr)
    sorted_indices = np.argsort(arr[:, 1])[::-1]
    sorted_arr = arr[sorted_indices]
    # print(sorted_arr)
    scnew = [0 for i in range(N)]
    k = 1
    while len(sorted_arr) > 0:
        # print(f'iteration {k} begins')
        # print('current sc is ', sorted_arr)
        # print(sc)
        k+=1
        i = int(sorted_arr[0,0])
        # fao = int(sorted_arr[0,1])
        z = differentiation_index(sc,id=i,cover=1)
        # print('z = ', z)
        # sorted_arr = sorted_arr[1:]
        rw = rank(sc, z, id=i)

        # print(i,sum(z),rw,R-fao)
        if np.abs(rw - sum(z)) <= 0.1:
            sc[i] = 0
            arr = np.array([[i, fao_value(sc,i)] for i in range(N) if sc[i]>0.9])
            sorted_indices = np.argsort(arr[:, 1])[::-1]
            sorted_arr = arr[sorted_indices]
            # z = differentiation(sc)
        else:
            scnew[i] = 1
            sorted_arr = sorted_arr[1:]
        # print()

    # z = differentiation_index(scnew,cover=1)
    # print(scnew, z)
    # print(W)
    # rw = rank(scnew,z)
    fao = fao_value(scnew)
    # xp = passivelocation(scnew)
    # print(xp)
    bestfit = sum(scnew)*c[0] + (R-fao)*c[1]
    scnew.append(bestfit)
    return np.array(scnew)

def MIPmodel():
    model = gurobipy.Model()
    var_x = {}
    for i in range(N):
        var_x[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x.' + str(i))

    for r in range(R):
        for s in range(R):
            if r != s:
                model.addConstr(quicksum(W[j][r] * var_x[j] for j in range(N)) >= 1)
    for r in range(R):
        for s in range(R):
            if r != s:
                model.addConstr(
                    quicksum(math.fabs(W[i][r] - W[i][s]) * var_x[i] for i in range(N))
                    >= 1, 'c1.' + str(r) + '.' + str(s))
    model.setObjective(quicksum(var_x[i] for i in range(N)), sense=GRB.MINIMIZE)
    model.optimize()
    obj = int(model.objVal)
    print('obj=', obj)
    return np.array([var_x[i].X for i in range(N)]), obj

def passivelocation(xa):
    # if net == 'SF' or net == 'EM':
    #     W = Womega[omega]
    model = gurobipy.Model()
    model.setAttr("ModelSense", GRB.MINIMIZE)
    var_x = {}
    for i in range(N):
        for j in range(2):
            if j==0:
                var_x[j, i] = model.addVar(lb=xa[i], ub=xa[i], name='x.' + str(j) + '.' + str(i))
            else:
                var_x[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
        # if i in xa:
        #     var_x[0,i].Start = 1
        # elif i in xp:
        #     var_x[1,i].Start = 1

    var_z = {}
    for r in range(R):
        var_z[r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z.' + str(r))
    var_u = {}
    var_h = {}
    # var_h2 = {}
    var_r = {}
    var_v = {}
    # var_v2 = {}
    for r in range(R):
        for i in range(N):
            var_h[r, i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                           vtype=GRB.CONTINUOUS, name='h.' + str(r) + '.' + str(i))

            # var_h2[r, i] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='h2.'+str(r)+'.'+str(i))
    for i in range(N):
        for r in range(R):
            var_u[i, r] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS,
                                       name='u.' + str(i) + '.' + str(r))

    for i in range(R):
        for r in range(R):
            for j in range(N):
                var_v[i, j, r] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                              name='v.' + str(i) + '.' + str(j) + '.' + str(r))

    model.update()
    model.setObjective(quicksum(model.getVarByName('x.' + str(0) + '.' + str(i)) for i in range(N)) * c[0]
                       + quicksum(model.getVarByName('x.' + str(1) + '.' + str(i)) for i in range(N)) * c[1])
    '''valid inequations'''
    model.addConstr(
        quicksum(var_x[j, i] for i in range(N) for j in range(2)) >= R - quicksum(var_z[r] for r in range(R)))

    for r in range(R):
        for s in range(R):
            if r != s:
                model.addConstr(quicksum(math.fabs(W[j][r] - W[j][s]) * (var_x[0, j] + var_x[1, j])
                                         for j in range(N)) >= 1 - var_z[r] - var_z[s])
    '''========================'''
    for r in range(R):
        for s in range(R):
            if r != s:
                model.addConstr(
                    quicksum(
                        math.fabs(W[i][r] - W[i][s]) * model.getVarByName('x.0.' + str(i)) for i in range(N))
                    >= model.getVarByName('z.' + str(r)), 'c1.' + str(r) + '.' + str(s))
    for r in range(R):
        model.addConstr(quicksum(
            W[i][r] * (model.getVarByName('x.0.' + str(i)) + model.getVarByName('x.1.' + str(i))) for i in
            range(N)) >= 1, 'c2-1.' + str(r))
        model.addConstr(quicksum(W[i][r] * model.getVarByName('x.0.' + str(i)) for i in range(N)) >= var_z[r],
                        'c2-2.' + str(r))
    for i in range(N):
        model.addConstr(quicksum(model.getVarByName('x.' + str(j) + '.' + str(i)) for j in range(2)) <= 1,
                        'c3.' + str(i))

    model.addConstr(
        quicksum(var_x[j, i] for i in range(N) for j in range(2)) >= R - quicksum(var_z[r] for r in range(R)))
    for r in range(R):
        for s in range(R):
            if r != s:
                # model.addConstr(quicksum(math.fabs(W[j][r] - W[j][s]) * (var_x[0,j] + var_x[1,j])
                #                          for j in range(N)) >= 1 - var_z[r] - var_z[s])
                model.addConstr(quicksum(math.fabs(W[j][r] - W[j][s]) * (var_x[0, j] + var_x[1, j])
                                         for j in range(N)) >= 1)

    '''elementary transformation'''
    for i in range(R):
        for r in range(R):
            if i == r:
                model.addConstr(quicksum(var_v[i, j, r] * W[j][r] for j in range(N)) == 1 - var_z[r],
                                'c5-1.' + str(i) + '.' + str(r))
            else:
                model.addConstr(quicksum(var_v[i, j, r] * W[j][r] for j in range(N)) == 0,
                                'c5-1.' + str(i) + '.' + str(r))

    '''definition of variable v'''
    for i in range(R):
        for j in range(N):
            for r in range(R):
                model.addConstr(var_v[i, j, r] <= var_h[i, j] + M * (1 - var_u[j, r]),
                                'c7-1.' + str(i) + '.' + str(j) + '.' + str(r))
                model.addConstr(var_v[i, j, r] >= var_h[i, j] - M * (1 - var_u[j, r]),
                                'c7-2.' + str(i) + '.' + str(j) + '.' + str(r))
                model.addConstr(var_v[i, j, r] <= M * var_u[j, r],
                                'c7-3.' + str(i) + '.' + str(j) + '.' + str(r))
                model.addConstr(var_v[i, j, r] >= -M * var_u[j, r],
                                'c7-3.' + str(i) + '.' + str(j) + '.' + str(r))

    '''definition of variable u'''
    for i in range(N):
        for r in range(R):
            model.addConstr(model.getVarByName('u.' + str(i) + '.' + str(r))
                            <= 1 - model.getVarByName('z.' + str(r)), 'c6-1.' + str(i) + '.' + str(r))
            model.addConstr(model.getVarByName('u.' + str(i) + '.' + str(r))
                            <= quicksum(model.getVarByName('x.' + str(j) + '.' + str(i)) for j in range(2)),
                            'c6-2.' + str(i) + '.' + str(r))
            model.addConstr(model.getVarByName('u.' + str(i) + '.' + str(r))
                            >= quicksum(model.getVarByName('x.' + str(j) + '.' + str(i)) for j in range(2))
                            - model.getVarByName('z.' + str(r)),
                            'c6-3.' + str(i) + '.' + str(r))

    '''solve model'''
    # model.setParam(GRB.Param.LogFile, f'EM_logging.{omega}.{c}.log')
    model.setParam('TimeLimit', timeLimit)
    model.optimize()
    sol_xp = [round(model.getVarByName("x.1." + str(i)).X, 0) for i in range(N)]
    return sol_xp

# 选择操作
def selection(population, fitnesses):
    total_fitness = np.sum(1/fitnesses)
    probabilities = (1 / fitnesses) / total_fitness
    # print(fitnesses)
    # print(probabilities)
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    # print(selected_indices)
    return population[selected_indices]

# 交叉操作
def crossover(parent1, parent2):
    crossover_point = random.randint(1, num_locations - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# 变异操作
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

# 遗传算法主程序
def genetic_algorithm(base_vector, globalobj, m):
    best_individual = np.array([])
    population = initialize_population(base_vector, population_size, num_locations, m)
    gast = time.perf_counter()
    for generation in range(num_generations):
        # fitnesses = []
        # for ind in population:
        #     fitnesses.append(list(fitness(ind)))
        #     # print(len(fitnesses[-1]))
        #     # print(fitness(ind), fitnesses)
        # fitnesses = np.array(fitnesses)
        # population = fitnesses[:,:-1]
        # fitnesses = fitnesses[:, -1]
        stt = time.perf_counter()
        fitnesses = np.array([fitness(ind) for ind in population])
        id = np.argmin(fitnesses)
        print(f"Generation {generation}: current best solution", np.nonzero(population[id]))
        best_fitness = np.min(fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
        # if best_fitness < c[0] * N:
        #     xp = passivelocation(population[id])
        #     print(sum(population[id]) * c[0] + sum(xp)*c[1])
        if best_fitness < globalobj:
            best_individual = population[id]
            globalobj = best_fitness
        new_population = []
        population = selection(population, fitnesses)
        for i in range(0, population_size, 2):
            parent1, parent2 = population[i], population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = np.array(new_population)
        ett = time.perf_counter()
        print(f"Generation {generation}: Time spend = {ett-stt}")
        print('====')
        gaet = time.perf_counter()
        if gaet - gast > timeLimit:
            break
    # best_individual = population[np.argmin(fitnesses)]
    return best_individual, globalobj

# 运行遗传算法
# n = 10  # 向量的维度
# base_vector = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])  # 基础向量
# num_samples = 10  # 需要生成的新向量的数量
# m = [2,5]  # 生成向量中1的最大数量
#
# generated_vectors = generate_vectors(base_vector, num_samples, n, m)
#
# for i, vec in enumerate(generated_vectors):
#     print(f"Generated Vector {i+1}: {vec}")
C = [[2,1],[5,1],[10, 1]]
# C = [[10,1]]

for c in C:
    for ii in range(Omega):
        if net == 'SF' or net == 'EM' or net == 'RD':
            W = Womega[ii]

        starttime = time.perf_counter()
        sc, obj = MIPmodel()
        # feasible_solution, feasible_fitness = intial_feasible_solution()
        # ett = time.perf_counter()
        # print(f'time for initial solution is {ett-starttime}')
        # print(feasible_solution)
        # print(feasible_fitness)
        # fao = fao_value(feasible_solution)
        # total_cost = np.sum(feasible_solution) * c[0] + (R - fao) * c[1]
        #
        # # print(feasible_solution)
        # # print(feasible_fitness)
        # best_solution, best_fitness = genetic_algorithm(feasible_solution, feasible_fitness, [int(obj/2), obj])
        # if best_solution.size == 0:
        #     best_solution = feasible_solution
        #     best_fitness = feasible_fitness
        # # xp = passivelocation(best_solution)
        # endtime = time.perf_counter()
        # print(ii, c)
        # print("Best Solution (active):", sum(best_solution), best_solution)
        # # print("Best Solution (passive):", xp)
        # print("Best Fitness:", best_fitness)
        # print("computational time:", endtime-starttime)
        # print('====')
