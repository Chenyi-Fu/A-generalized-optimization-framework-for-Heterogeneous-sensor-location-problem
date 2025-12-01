from __future__ import print_function

import sys
import itertools
import math
import gurobipy
from gurobipy import GRB, quicksum
import numpy as np
import random
import time
import heapq
import sensor_data
Data = sensor_data.data()
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
            print(u,v)
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


net = 'RD'
mtype = 'exact'
if net == 'SF' or net == 'EM':
    if net == 'SF':
        W0 = Data.W_SF
    else:
        W0 = Data.W_EM
    random.seed(2024)
    number = 100
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
    node = 20
    max_k = 5
    N = 30
    R = number = 60
    selected_OD = int(node / 4)
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
        print("边的数量:", G.number_of_edges()*2)
        print("选中的节点:", selected_nodes)
        print("生成的路径:", len(paths), paths)
        print()
        for r in range(R):
            # print(r, paths[r])
            for a in range(len(paths[r]) - 1):
                i = edges.index((paths[r][a], paths[r][a + 1]))
                Womega[omega][i][r] = 1
    N *= 2
# c = [0.5, 1]
# c = [10, 1]
M = 10
timeLimit = 3600*12
softlimit = 3600*2
starttime = 0
bestub = 100000

def substitution(W, xa, xp, zlabel):
    What = [[W[i][r] * (xa[i]+xp[i]) if r in zlabel else 0 for r in range(R)] for i in range(N)]
    # for i in range(N):
    #     print(What[i])
    Wsum = [sum(What[i][r] for i in range(N)) for r in range(R)]
    # print()
    # print([[r,Wsum[r]] for r in range(R) if Wsum[r] > 1])
    zlabel_redu = [r for r in range(R) if Wsum[r] > 1]
    # for i in range(N):
    #     if xa[i]+xp[i]>=1 and sum([What[i][r] for r in range(R) if Wsum[r]>1]) > 0:
    #         print(i, [What[i][r] for r in range(R) if Wsum[r]>1])
    linkset = []
    pathset = []
    useless_sensor = []
    for r in zlabel_redu:
        path = []
        link = []
        if r not in pathset:
            link = [i for i in range(N) if What[i][r] and i not in linkset]
            for i in link:
                for s in zlabel_redu:
                    if s not in path and What[i][s] == 1:
                        path.append(s)
            pathset.extend(path)
            for s in path:
                for i in range(N):
                    if i not in link and What[i][s] == 1:
                        link.append(i)
            linkset.extend(link)
            # print(link, path)
            xp_re = [i for i in link if xp[i]]
            xa_re = [i for i in link if xa[i]]
            # print('x = ', xa_re, xp_re)
            # print('W temp = ')
            W_temp = [[What[i][r] for r in path] for i in link]
            # for i in range(len(W_temp)):
            #     print(W_temp[i])
            W_temp_xa = [[What[i][r] for r in path] for i in xa_re]
            rank_xa = np.linalg.matrix_rank(W_temp_xa)
            rank_all = np.linalg.matrix_rank(W_temp)
            rank_need = rank_all - rank_xa
            redundant = len(xp_re) - rank_need
            if redundant > 0:
                useless_sensor.extend([xp_re[i] for i in range(redundant)])
    return useless_sensor

def softtime(model, where):
    global starttime, bestub
    if where == GRB.Callback.MIP:
        runtime = model.cbGet(GRB.Callback.RUNTIME) - starttime
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        # objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        # gap = abs((objbst - objbnd) / objbst)
        if bestub > objbst:
            bestub = objbst
            starttime = model.cbGet(GRB.Callback.RUNTIME)
            print('Find new feasible solution ', starttime, bestub)
        if runtime > softlimit:
            model.terminate()

def benders(model, where):
    if where == GRB.Callback.MIP:
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        if objbnd + c[1] >= objbst:
            model.terminate()
    if where == GRB.Callback.MIPSOL:
        z_vals = model.cbGetSolution(model._zvars)
        x_vals = model.cbGetSolution(model._xvars)
        zval = [round(z_vals[i],0) for i in z_vals.keys()]
        zsum = sum(zval)
        u = [[(x_vals[(0,i)] + x_vals[(1,i)]) * (1-zval[r]) for r in range(R)] for i in range(N)]
        Wu = [[W[j][r] * u[j][r] for r in range(R)] for j in range(N)]
        rw = np.linalg.matrix_rank(Wu)
        if rw == R - zsum:
            pass
        else:
            # cut (18)
            if rr[3]>0.9:
                pi, phi = subproblem_update(u,zval)
                model.cbLazy(quicksum((1-model._zvars[r]) * pi[r] for r in range(R) if pi[r] != 0)
                             - M * quicksum((1-model._uvars[j,r]) * (phi[0][s][j][r] + phi[1][s][j][r])
                                          for j in range(N) for r in range(R) for s in range(R))
                             - M * quicksum(model._uvars[j,r] * (phi[2][s][j][r] + phi[3][s][j][r])
                                          for j in range(N) for r in range(R) for s in range(R))
                             <= 0)
            if rr[0]>0.9:
                # cut (12)
                model.cbLazy(quicksum(model._xvars[(0,i)]for i in range(N) if round(x_vals[(0,i)],0) >= 0.9)
                             + quicksum(model._xvars[(1,i)]for i in range(N) if round(x_vals[(1,i)],0) >= 0.9)
                             + quicksum(1-model._xvars[(0,i)]for i in range(N) if round(x_vals[(0,i)],0) <= 0.1)
                             + quicksum(1-model._xvars[(1,i)]for i in range(N) if round(x_vals[(1,i)],0) <= 0.1)
                             <= 2*N - 1)

            # model.cbLazy(quicksum(model._xvars[(0,i)] + model._xvars[(1,i)] for i in range(N) if round(x_vals[(0,i)] + x_vals[(1,i)]) <= 0.1)
            #              + quicksum(model._zvars[r] for r in range(R) if round(z_vals[r],0) <= 0.1) >= 1)
            if rr[1] > 0.9:
                # cut (13)
                model.cbLazy(quicksum(model._xvars[(0, i)] + model._xvars[(1, i)] for i in range(N) if
                                      round(x_vals[(0, i)] + x_vals[(1, i)]) <= 0.1)
                             + quicksum(model._zvars[r] for r in range(R) if round(z_vals[r], 0) <= 0.1) >= R-zsum-rw)
            if rr[2] > 0.9:
                # cut (14)
                Wy = [[W[j][r] * (1 - zval[r]) for r in range(R)] for j in range(N)]
                rwy = np.linalg.matrix_rank(Wy)
                if rwy < R - zsum:
                    model.cbLazy(quicksum(model._zvars[r] for r in range(R) if z_vals[r] < 0.1) >= R-zsum-rwy)

def subproblem():
    # Wu = [[W[j][r] * u[j,r] for r in range(R)] for j in range(N)]
    u = [[1 for r in range(R)] for j in range(N)]
    z = [1 for r in range(R)]
    model = gurobipy.Model()

    pi = {}
    for i in range(R):
        for r in range(R):
            pi[i, r] = model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=f'pi_{i}_{r}')
    phi = {}
    for s in range(4):
        for j in range(R):
            for i in range(N):
                for r in range(R):
                    phi[s,j,i,r] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'phi_{j}_{i}_{r}')

    model.setObjective(quicksum(pi[r, r] * (1-z[r]) for r in range(R))
                       - M * quicksum((1-u[j][r]) * phi[0,s,j,r] + (1-u[j][r]) * phi[1,s,j,r] + u[j][r] * phi[2,s,j,r] + u[j][r] * phi[3,s,j,r]
                                      for j in range(N) for r in range(R) for s in range(R)),
                       GRB.MAXIMIZE)

    # model.addConstr(quicksum((1-u[j][r]) * phi[0,s,j,r] + (1-u[j][r]) * phi[1,s,j,r] + u[j][r] * phi[2,s,j,r] + u[j][r] * phi[3,s,j,r]
    #                                   for j in range(N) for r in range(R) for s in range(R)) == 0)
    for j in range(N):
        for s in range(R):
            model.addConstr(quicksum(phi[0,s,j,r] - phi[1,s,j,r] for r in range(R)) == 0)
            for r in range(R):
                model.addConstr(W[j][r] * pi[s,r]-phi[0,s,j,r] + phi[1,s,j,r] - phi[2,s,j,r] + phi[3,s,j,r] == 0)
    model.setParam('InfUnbdInfo', 1)
    model.setParam('OutputFlag', 0)

    model._pivars = pi
    model._phivars = phi
    return model


# def subproblem2(u, z):
#     # Wu = [[W[j][r] * u[j,r] for r in range(R)] for j in range(N)]
#     # u = [[1 for r in range(R)] for j in range(N)]
#     # z = [1 for r in range(R)]
#     model = gurobipy.Model()
#
#     pi = {}
#     for i in range(R):
#         for r in range(R):
#             pi[i, r] = model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=f'pi_{i}_{r}')
#     phi = {}
#     for s in range(4):
#         for j in range(R):
#             for i in range(N):
#                 for r in range(R):
#                     phi[s,j,i,r] = model.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name=f'phi_{j}_{i}_{r}')
#
#     model.setObjective(quicksum(pi[r, r] * (1-z[r]) for r in range(R))
#                        - M * quicksum((1-u[j][r]) * phi[0,s,j,r] + (1-u[j][r]) * phi[1,s,j,r] + u[j][r] * phi[2,s,j,r] + u[j][r] * phi[3,s,j,r]
#                                       for j in range(N) for r in range(R) for s in range(R)),
#                        GRB.MAXIMIZE)
#
#     model.addConstr(quicksum((1-u[j][r]) * phi[0,s,j,r] + (1-u[j][r]) * phi[1,s,j,r] + u[j][r] * phi[2,s,j,r] + u[j][r] * phi[3,s,j,r]
#                                       for j in range(N) for r in range(R) for s in range(R)) == 0)
#     for j in range(N):
#         for s in range(R):
#             model.addConstr(quicksum(phi[0,s,j,r] - phi[1,s,j,r] for r in range(R)) == 0)
#             for r in range(R):
#                 model.addConstr(W[j][r] * pi[s,r]-phi[0,s,j,r] + phi[1,s,j,r] - phi[2,s,j,r] + phi[3,s,j,r] == 0)
#     model.setParam('InfUnbdInfo', 1)
#     model.setParam('OutputFlag', 0)
#
#     model.optimize()
#     val_pi = [SP._pivars[r,r].UnbdRay for r in range(R)]
#     val_phi = [[[[SP._phivars[i,s,j,r].UnbdRay for r in range(R)] for j in range(N)] for s in range(R)] for i in range(4)]
#     print('obj',
#         sum(-M * (1 - u[j][r]) * (val_phi[0][s][j][r] + val_phi[1][s][j][r]) - M * u[j][r] * (val_phi[2][s][j][r] + val_phi[3][s][j][r]) for s in range(R) for j in range(N) for r in range(R))
#     )
#     return val_pi, val_phi

def subproblem_update(u, z):
    # Wu = [[W[j][r] * u[j,r] for r in range(R)] for j in range(N)]

    for r in range(R):
        SP._pivars[r,r].Obj = 1-z[r]
    for s in range(R):
        for j in range(N):
            for r in range(R):
                SP._phivars[0,s,j,r].Obj = -M * (1-u[j][r])
                SP._phivars[1,s,j,r].Obj = -M * (1-u[j][r])
                SP._phivars[2,s,j,r].Obj = -M * u[j][r]
                SP._phivars[3,s,j,r].Obj = -M * u[j][r]
    SP.optimize()
    val_pi = [SP._pivars[r,r].UnbdRay for r in range(R)]
    val_phi = [[[[SP._phivars[i,s,j,r].UnbdRay for r in range(R)] for j in range(N)] for s in range(R)] for i in range(4)]
    print('obj',
        sum(-M * (1 - u[j][r]) * (val_phi[0][s][j][r] + val_phi[1][s][j][r]) - M * u[j][r] * (val_phi[2][s][j][r] + val_phi[3][s][j][r]) for s in range(R) for j in range(N) for r in range(R))
    )
    return val_pi, val_phi

def rank(xa):
    model = gurobipy.Model()
    model.setAttr("ModelSense", GRB.MAXIMIZE)
    var_x = {}
    for i in range(N):
        for j in range(1):
            # var_x[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='x.' + str(j) + '.' + str(i))
            if i in xa and j == 0:
                var_x[j, i] = model.addVar(lb=1, ub=1, vtype=GRB.CONTINUOUS, name='x.' + str(j) + '.' + str(i))
            else:
                var_x[j, i] = model.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name='x.' + str(j) + '.' + str(i))

    var_z = {}
    var_zhat = {}
    for r in range(R):
        var_z[r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z.' + str(r))
        var_zhat[r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z.' + str(r))

    model.update()
    model.setParam('OutputFlag', 0)
    model.setObjective(quicksum(var_z[r] for r in range(R)))

    for r in range(R):
        for s in range(R):
            if r != s:
                model.addConstr(
                    quicksum(math.fabs(W[i][r] - W[i][s]) * model.getVarByName('x.0.' + str(i)) for i in range(N))
                    >= var_z[r], 'c1.' + str(r) + '.' + str(s))
    for r in range(R):
        model.addConstr(quicksum(W[i][r] * model.getVarByName('x.0.' + str(i)) for i in range(N)) >= var_z[r],
                        'c2-2.' + str(r))
    model.optimize()
    sol_z = [round(model.getVarByName("z." + str(r)).X, 0) for r in range(R)]
    zlabel1 = [r for r in range(R) if round(sol_z[r], 0) == 1]
    zlabel0 = [r for r in range(R) if round(sol_z[r], 0) == 0]
    # print(sol_z)
    # print('z_true = ', zlabel0, zlabel1)
    return sol_z

k = 0
C = [[5, 1]]
# C=[[10,1]]
# xxa = [1, 2, 3, 5, 8, 9, 11, 16, 17, 18, 19, 20, 28, 29, 34 ]
# xxp = [6, 22, 30, 36]
dim = 3
possible_vectors = set(itertools.product([0, 1], repeat=dim))
possible_vectors = list(possible_vectors)
possible_vectors.sort(reverse=True)
possible_vectors = [[0,1,1,0]]
repeat = 10
np.random.seed(2025)
ranc = np.random.uniform(1,5,(N, repeat))
ran = np.random.uniform(0,0.5,(N, repeat))


for c in C:
    for rr in possible_vectors:
        print('cut combination: ', c, rr)
        for omega in range(Omega):
            if net == 'SF' or net == 'EM' or net == 'RD':
                W = Womega[omega]
            # k += 1
            model = gurobipy.Model()
            model.setAttr("ModelSense", GRB.MINIMIZE)
            var_x = {}
            for i in range(N):
                for j in range(2):
                    # if j == 0 and i+1 in xxa:
                    #     var_x[j, i] = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                    # elif j == 1 and i+1 in xxp:
                    #     var_x[j, i] = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                    # else:
                    #     var_x[j, i] = model.addVar(lb=0, ub=0, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                    var_x[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                # var_x[0,i].Start = xa[omega][i]
                # var_x[1,i].Start = xp[omega][i]


            if c != [1,1]:
                var_z = {}
                var_y = {}
                for r in range(R):
                    var_z[r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z.' + str(r))
                    for s in range(R):
                        var_y[r, s] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y.' + str(r) + '.' + str(s))

                var_u = {}
                for i in range(N):
                    for r in range(R):
                        var_u[i, r] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS,
                                                   name='u.' + str(i) + '.' + str(r))
                model.update()
                model.setObjective(quicksum(model.getVarByName('x.' + str(0) + '.' + str(i)) for i in range(N)) * c[0]
                                   + quicksum(model.getVarByName('x.' + str(1) + '.' + str(i)) for i in range(N)) * c[
                                       1])
                for r in range(R):
                    for s in range(R):
                        if r != s:
                            model.addConstr(
                                quicksum(math.fabs(W[i][r] - W[i][s]) * var_x[0,i] for i in range(N))
                                >= var_y[r,s], 'c1-1.' + str(r) + '.' + str(s))
                            model.addConstr(
                                quicksum(math.fabs(W[i][r] - W[i][s]) * var_x[0,i] for i in range(N))
                                <= N*var_y[r,s], 'c1-2.' + str(r) + '.' + str(s))
                    model.addConstr(quicksum(var_y[r,s] for s in range(R) ) >= R * var_z[r])
                    model.addConstr(quicksum(var_y[r,s] for s in range(R) ) <= (R-1) + var_z[r])
                #             for i in range(N):
                #                 model.addConstr(
                #                     math.fabs(W[i][r] - W[i][s]) * model.getVarByName('x.0.' + str(i))
                #                     <= model.getVarByName('z.' + str(r)), 'c1-2.' + str(r) + '.' + str(s) + '.' + str(i))
                # for r in [27,31,33,43,49]:
                #     for s in range(R):
                #         if r!=s:
                #             print(r,s, [(i,math.fabs(W[i][r] - W[i][s]) * model.getVarByName('x.0.' + str(i)).UB) for i in range(N)])
                for r in range(R):
                    model.addConstr(quicksum(
                        W[i][r] * (model.getVarByName('x.0.' + str(i)) + model.getVarByName('x.1.' + str(i))) for i in
                        range(N)) >= 1, 'c2-1.' + str(r))
                    model.addConstr(quicksum(W[i][r] * model.getVarByName('x.0.' + str(i)) for i in range(N)) >= var_y[r,r],
                                    'c2-2.' + str(r))
                    model.addConstr(quicksum(W[i][r] * model.getVarByName('x.0.' + str(i)) for i in range(N)) <= N * var_y[r,r],
                                    'c2-3.' + str(r))
                for i in range(N):
                    model.addConstr(quicksum(model.getVarByName('x.' + str(j) + '.' + str(i)) for j in range(2)) <= 1,
                                    'c3.' + str(i))

                '''valid inequality'''
                model.addConstr(quicksum(var_x[j,i] for i in range(N) for j in range(2)) >= R - quicksum(var_z[r] for r in range(R)))
                for r in range(R):
                    for s in range(R):
                        if r != s:
                            model.addConstr(quicksum(math.fabs(W[j][r] - W[j][s]) * (var_x[0,j] + var_x[1,j])
                                                     for j in range(N)) >= 1 - var_z[r] - var_z[s])

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
            else:
                for r in range(R):
                    for s in range(R):
                        if r != s:
                            model.addConstr(
                                quicksum(math.fabs(W[i][r] - W[i][s]) * var_x[0, i] for i in range(N))
                                >= 1, 'c1-1.' + str(r) + '.' + str(s))
                    model.addConstr(quicksum(W[i][r] * var_x[0, i] for i in range(N)) >= 1, 'c2-1.' + str(r))
                model.update()
                model.setObjective(quicksum(model.getVarByName('x.' + str(0) + '.' + str(i)) for i in range(N)) * c[0])

            '''solve model'''
            SP = subproblem()
            if net == 'RD':
                model.setParam(GRB.Param.LogFile, net+f'_benders_logging.{omega}.{c}.{node}.{N}.{number}.log')
            else:
                model.setParam(GRB.Param.LogFile, net+f'_benders_logging.{omega}.{c}.{N}.{number}.log')
            model.setParam('TimeLimit', timeLimit)
            if c == [1,1]:
                model.optimize()
            else:
                model._xvars = var_x
                model._zvars = var_z
                model._uvars = var_u
                model.Params.LazyConstraints = 1
                model.Params.Heuristics = 0
                model.optimize(benders)
            try:
                # sol_u = [[model.getVarByName("u." + str(i) + '.' + str(r)).X for r in range(R)] for i in range(N)]
                sol_xa = [round(model.getVarByName("x.0." + str(i)).X, 0) for i in range(N)]
                sol_xp = [round(model.getVarByName("x.1." + str(i)).X, 0) for i in range(N)]
                # sol_h = [[model.getVarByName("h." + str(i) + '.' + str(r)).X for r in range(N)] for i in range(R)]
                # sol_r = [[model.getVarByName("r." + str(i) + '.' + str(r)).X for r in range(R)] for i in range(R)]
                xalabel = [i for i in range(N) if round(sol_xa[i], 0) == 1]
                xplabel = [i for i in range(N) if round(sol_xp[i], 0) == 1]
                print('xa', xalabel)
                print('xp', xplabel)
                if c != [1,1]:
                    sol_z = [round(model.getVarByName("z." + str(r)).X, 0) for r in range(R)]
                    zlabel1 = [r for r in range(R) if round(sol_z[r], 0) == 1]
                    zlabel0 = [r for r in range(R) if round(sol_z[r], 0) == 0]
                    print('z', zlabel0, zlabel1)
                # sub = substitution(W, sol_xa, sol_xp,zlabel0)
                # print('useless sensors = ', sub)
                if net == "RD":
                    with open(net+f'benders_results.{omega}.{c}.{node}.{N}.{number}.txt', 'w') as file:
                        file.write(str(xalabel) + '\n')
                        file.write(str(xplabel) + '\n')
                        # file.write(str(sub))
                    file.close()
                else:
                    with open(net+f'benders_results.{omega}.{c}.{N}.{number}.txt', 'w') as file:
                        file.write(str(xalabel) + '\n')
                        file.write(str(xplabel) + '\n')
                        # file.write(str(sub))
                    file.close()
            except:
                pass

            # with open(net + '_' + str(number) +'.txt', 'a') as f:
            #     f.write('obj ' + str(model.getAttr('ObjVal')) + '\n')
            #     f.write('time and gap ' + str(model.getAttr('Runtime')) + '  ' + str(model.getAttr('MIPGap')) + '\n')
            #     f.write('xa '+str(xalabel) + '\n')
            #     f.write('xp '+str(xplabel) + '\n')
            #     f.write('z_ture '+str(zlabel0) + '\n')
            #     f.write(str(sub) + '\n\n')
            # f.close()


