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
import networkx as nx
from scipy.spatial import distance_matrix


net = 'RD'
if net != 'RD':
    import sensor_data
    Data = sensor_data.data()
mtype = 'exact'

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
        ii=0
        for u, v in combination:
            if k[i] > 0:
                if k[i] > 1:
                    path = list(nx.shortest_simple_paths(G, u, v))
                    paths.extend(path[:k[i]])
                    k[i] = len(path[:k[i]])
                # print(path[:k[i]])
                else:
                    path = list(nx.dijkstra_path(G, source=u, target=v))
                    print(ii, path)
                    paths.append(path)
                    k[i] = 1
                    ii+=1
            i+=1
        # print(len(paths))
        if len(paths) == num_paths:
            return paths
        else:
            diff = num_paths - sum(k)
            paths = []


if net == 'SF' or net == 'EM':
    if net == 'SF':
        W0 = Data.W_SF
    else:
        W0 = Data.W_EM
    random.seed(2024)
    number = 200
    Omega = 5
    ran = [[random.random() for i in range(len(W0[0]))] for omega in range(Omega)]
    index = [heapq.nlargest(number, ran[omega]) for omega in range(Omega)]
    # for omega in range(Omega):
    #     print(len(index[omega]),index[omega])
    Womega = [[[W0[j][i] for i in range(len(W0[0])) if ran[omega][i] in index[omega]]
               for j in range(len(W0))] for omega in range(Omega)]
    N = len(Womega[0])
    R = len(Womega[0][0])
elif net == 'ND':
    W = Data.W_ND
    Omega = 1
    N = len(W)
    R = number = len(W[0])
elif net == 'PH':
    W = Data.W_PH
    Omega = 1
    N = len(W)
    R = number = len(W[0])
elif net == 'RD':
    node = 10
    max_k = 1
    N = 20
    R = 60
    selected_OD = int(node/2)
    Omega = 200

    Womega = [[[0 for r in range(R)] for i in range(N * 2)] for omega in range(Omega)]
    for omega in range(Omega):
        random.seed(214 + omega)
        G, points = generate_k_nearest_graph(node, 5, N, omega)

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
M = 10
timeLimit = 3600*12
softlimit = 3600*2
starttime = 0
bestub = 100000

def substitution(W, xa, xp, zlabel):
    # print([i for i in range(N) if xa[i]])
    # print([i for i in range(N) if xp[i]])
    What = [[W[i][r] * xa[i] if r in zlabel else 0 for r in range(R)] for i in range(N)]
    Whatp = [[W[i][r] * xp[i] if r in zlabel else 0 for r in range(R)] for i in range(N)]
    # for i in range(N):
    #     print(What[i])
    # print()
    # for i in range(N):
    #     print(Whatp[i])
    xplabel = [i for i in range(N) if xp[i]>0.5]
    xalabel = [i for i in range(N) if xa[i] > 0.5]
    no_active_path_xp = {}
    with_active_path_xa = {}
    with_active_path_xp = {}
    useless = []
    for r in zlabel:
        if sum([What[i][r] for i in range(N)]) < 0.5:
            no_active_path_xp[r] = [i for i in xplabel if Whatp[i][r]>0.5]
        else:
            with_active_path_xp[r] = [i for i in xplabel if Whatp[i][r]>0.5]
            with_active_path_xa[r] = [i for i in xalabel if What[i][r] >0.5]
    pathset = []
    linkset_xp = []
    # linkset_all = []
    usedpath = []


    for r in with_active_path_xa.keys():
        if r not in usedpath:
            usedpath.append(r)
            path = [r]
            link = [rr for rr in with_active_path_xp[r]]
            # linkall = [rr for rr in with_active_path_xa[r]]
            # linkall.extend(link)
            for s in with_active_path_xa.keys():
                if s != r and with_active_path_xa[s] == with_active_path_xa[r]:
                    usedpath.append(s)
                    path.append(s)
                    link.extend(with_active_path_xp[s])
                    # linkall.extend(with_active_path_xp[s])
            pathset.append(path)
            linkset_xp.append(link)
            # linkset_all.append(linkall)

    for ii in no_active_path_xp.keys():
        pathset.append([ii])
        linkset_xp.append(no_active_path_xp[ii])
    # W_temp = [[1 if i in linkset_xp else 0 for i in range(N)] for r in range(len(linkset_xp))]
    # for r in range(len(linkset_xp)):
    #     print(W_temp[r])
    print('ooo')
    print(pathset)
    print(linkset_xp)
    model = gurobipy.Model()
    model.setAttr("ModelSense", GRB.MINIMIZE)
    var_x = {}
    for r in range(len(pathset)):
        for i in xplabel:
            var_x[r,i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY,name=f'x.{r}.{i}')
    for r in range(len(pathset)):
        if len(pathset[r]) == 1:
            # print([[W_temp[r][i], var_x[i]] for i in xplabel])
            model.addConstr(quicksum( var_x[r,i] for i in linkset_xp[r]) == 1)
        else:
            # print([[W_temp[r][i], var_x[i]] for i in xplabel])
            model.addConstr(quicksum( var_x[r,i] for i in linkset_xp[r]) == len(pathset[r]) - 1)
    for i in xplabel:
        model.addConstr(quicksum(var_x[r,i] for r in range(len(pathset)) if i in linkset_xp[r]) <= 1, name=f'cons.{i}')
    # model.write('test1.lp')
    model.setObjective(quicksum(var_x[r, i] for i in xplabel for r in range(len(pathset))))

    model.optimize()
    useless = [i for i in xplabel if sum(model.getVarByName(f"x.{r}.{i}").X for r in range(len(pathset))) < 0.5]


    return useless

def differentiation_index(individual, id=None):
    if id != None:
        individual[id] = 0

    label = [[min(1,sum([np.abs(W[i][r] - W[i][s]) * individual[i] for i in range(N)])) for s in range(R)] for r in range(R)]
    label2 = [min(1,sum([W[i][r] * individual[i] for i in range(N)])) for r in range(R)]
    z = [0 if sum(label[r]) == R - 1 and np.abs(label2[r] - 1) < 0.1 else 1 for r in range(R)]
    # print(id)
    # print([[r, [s for s in range(R) if label[r][s] < 1 and r!=s]] for r in range(R) if z[r]])
    # print([sum(label[r]) for r in range(R) if z[r]])
    # print([label2[r] for r in range(R) if z[r]])
    if id != None:
        individual[id] = 1
    return z
# xp 6 [12, 19, 21, 23, 28, 36]

def rank(individual, label, id = None):
    if id != None:
        individual[id] = 0
    H = [[W[i][r]* label[r] for r in range(R)] for i in range(N)]
    rw = np.linalg.matrix_rank(H)
    if id != None:
        individual[id] = 1
    return rw

def substitution_active(W, xa, xp, cost):
    arr =[]
    arrp = []
    for i in range(N):
        if xa[i] > 0.9:
            arr.append([i, cost[0][i]])
        arrp.append([i, cost[1][i]])
    arr = np.array(arr)
    arrp = np.array(arrp)
    sorted_indices = np.argsort(arr[:, 1])[::-1]
    sorted_arr = arr[sorted_indices]
    sorted_indices = np.argsort(arrp[:, 1])
    sorted_arrp = arrp[sorted_indices]
    active_remove = []
    passive_add = []
    xp_ori = [xp[ii] for ii in range(N)]
    # print(sorted_arr)
    # print(sorted_arrp)
    for j in range(len(sorted_arr)):
        i = sorted_arr[j][0]
        ca = sorted_arr[j][1]
        z = differentiation_index(xa, id=i)
        rw = rank(xa, z, id=i)
        # print(f'test {i} with cost {ca} rank is {rw} and number of undifferentiated routes is {sum(z)}')
        if np.abs(rw - sum(z)) <= 0.1:
            xa[i] = 0
            What = [[W[ii][r] * (xa[ii]+xp[ii]) * z[r] for r in range(R)] for ii in range(N)]
            # print([r for r in range(R) if z[r] > 0.5])
            # for ii in range(N):
            #     if sum(What[ii]) > 0:
            #         print(ii, [What[ii][r] for r in range(R) if z[r] > 0.5])
            # print()
            rw1 = np.linalg.matrix_rank(What)
            tempcost = 0
            for jj in range(len(sorted_arrp)):
                if xa[jj] + xp[jj] < 0.5:
                    What = [[W[ii][r] * (xa[ii]+xp[ii]) * z[r] for r in range(R)]
                            if ii != jj else [W[ii][r] * z[r] for r in range(R)] for ii in range(N)]
                    rw2 = np.linalg.matrix_rank(What)
                    if rw2 > rw1:
                        # for ii in range(N):
                        #     if sum(What[ii]) > 0:
                        #         print(ii, [What[ii][r] for r in range(R) if z[r] > 0.5])
                        # print()
                        # z = differentiation_index(xa)
                        # print('rank1=',rw2,sum(z),rw)
                        rw1 = rw2
                        xp[jj] = 1
                        tempcost += sorted_arrp[jj][1]
                    if rw2 == rw:
                        z = differentiation_index(xa)
                        # print('rank2=',rw2,sum(z),rw)
                        break
            if tempcost >= ca:
                print(f'active sensor {i} cannot removed with cost {sum(xp)-sum(xp_ori)}')
                xa[i] = 1
                xp = [xp_ori[ii] for ii in range(N)]
                # print(xp_ori)
                # print('xa=', [ii for ii in range(N) if xa[ii] > 0.5])
                # print('xp=', [ii for ii in range(N) if xp[ii] > 0.5])
            else:
                print(f'active sensor {i} is removed with cost {sum(xp)-sum(xp_ori)}')
                active_remove.append(i)
                # print(xp_ori)
                # print('xa=', [ii for ii in range(N) if xa[ii] > 0.5])
                # print('xp=', [ii for ii in range(N) if xp[ii] > 0.5])
                temp = [jj for jj in range(N) if xp_ori[jj]==0 and xp[jj]==1]
                passive_add.extend(temp)
                xp_ori = [xp[ii] for ii in range(N)]

    print('remove active sensors:', active_remove)

    print('add new passive sensors', passive_add)

    z = differentiation_index(xa)
    rw = rank(xa, z)
    print('rank=',rw, sum(z))
    print(np.linalg.matrix_rank([[z[r] * (xa[i]+xp[i]) * W[i][r] for r in range(R)] for i in range(N)]))
    zlabel = [r for r in range(R) if z[r] >0.5]
    print(zlabel,len(zlabel))
    sub = substitution(W, xa, xp, zlabel)
    # check([i for i in range(N) if xa[i]>0.5], [i for i in range(N) if xp[i]>0.5], W)
    return sub, [i for i in range(N) if xa[i]>0.5], [i for i in range(N) if xp[i]>0.5]



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

# def rank(xa, B=0):
#     model = gurobipy.Model()
#     model.setAttr("ModelSense", GRB.MAXIMIZE)
#     var_x = {}
#     if xa != []:
#         for i in range(N):
#             if i in xa:
#                 var_x[0, i] = model.addVar(lb=1, ub=1, vtype=GRB.CONTINUOUS, name='x.' + str(0) + '.' + str(i))
#             else:
#                 var_x[0, i] = model.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name='x.' + str(0) + '.' + str(i))
#     else:
#         for i in range(N):
#             var_x[0, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x.' + str(0) + '.' + str(i))
#         model.addConstr(quicksum(var_x[0, i] for i in range(N)) <= B)
#
#     var_z = {}
#     # var_zhat = {}
#     for r in range(R):
#         var_z[r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z.' + str(r))
#         # var_zhat[r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z.' + str(r))
#
#     model.update()
#     model.setObjective(quicksum(var_z[r] for r in range(R)))
#
#     for r in range(R):
#         for s in range(R):
#             if r != s:
#                 model.addConstr(
#                     quicksum(math.fabs(W[i][r] - W[i][s]) * model.getVarByName('x.0.' + str(i)) for i in range(N))
#                     >= var_z[r], 'c1.' + str(r) + '.' + str(s))
#     for r in range(R):
#         model.addConstr(quicksum(W[i][r] * model.getVarByName('x.0.' + str(i)) for i in range(N)) >= var_z[r],
#                         'c2-2.' + str(r))
#     model.optimize()
#     sol_z = [round(var_z[r].X, 0) for r in range(R)]
#     zlabel1 = [r for r in range(R) if round(sol_z[r], 0) == 1]
#     zlabel0 = [r for r in range(R) if round(sol_z[r], 0) == 0]
#     print(sol_z)
#     print('z_true = ', zlabel0, zlabel1)
#     # return zlabel0


def differentiation(xp):
    var_x = {}
    for i in range(N):
        if i in xp:
            var_x[i] = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name='x.' + str(i))
        else:
            var_x[i] = model.addVar(lb=0, ub=0, vtype=GRB.BINARY, name='x.' + str(i))

    var_z = {}
    for r in range(R):
        var_z[r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z.' + str(r))
    for r in range(R):
        for s in range(R):
            if r != s:
                model.addConstr(quicksum(W[j][r] * var_x[j] for j in range(N)) >= var_z[r])
    for r in range(R):
        for s in range(R):
            if r != s:
                model.addConstr(
                    quicksum(math.fabs(W[i][r] - W[i][s]) * var_x[i] for i in range(N))
                    >= var_z[r], 'c1.' + str(r) + '.' + str(s))
    model.setObjective(quicksum(var_z[r] for r in range(R)), sense=GRB.MAXIMIZE)
    model.optimize()
    return [r for r in range(R) if var_z[r].X < 0.5]


def check(xa, xp, W):
    model = gurobipy.Model()
    model.setAttr("ModelSense", GRB.MINIMIZE)
    var_x = {}
    for i in range(N):
        for j in range(2):
            if j == 0:
                if i in xa:
                    var_x[j, i] = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                else:
                    var_x[j, i] = model.addVar(lb=0, ub=0, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
            else:
                if i in xp:
                    var_x[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                else:
                    var_x[j, i] = model.addVar(lb=0, ub=0, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))

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
    # for i in range(R):
    #     for r in range(R):
    #         if i == r:
    #             var_r[i,r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='r.'+str(i)+'.'+str(r))
    #         else:
    #             var_r[i,r] = model.addVar(lb=0, ub=0, vtype=GRB.BINARY, name='r.'+str(i)+'.'+str(r))

    for i in range(R):
        for r in range(R):
            for j in range(N):
                var_v[i, j, r] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                                  name='v.' + str(i) + '.' + str(j) + '.' + str(r))

                # var_v2[i, j, r] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS,
                #                               name='v2.'+str(i)+'.'+str(j)+'.'+str(r))

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
                    quicksum(math.fabs(W[i][r] - W[i][s]) * model.getVarByName('x.0.' + str(i)) for i in range(N))
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

    # model.addConstr(quicksum(var_x[j,i] for i in range(N) for j in range(2)) >= R - quicksum(var_z[r] for r in range(R)))
    # for r in range(R):
    #     for s in range(R):
    #         if r != s:
    #             model.addConstr(quicksum(math.fabs(W[j][r] - W[j][s]) * (var_x[0,j] + var_x[1,j])
    #                                      for j in range(N)) >= 1 - var_z[r] - var_z[s])

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
                model.addConstr(var_v[i, j, r] <= M * var_u[j, r], 'c7-3.' + str(i) + '.' + str(j) + '.' + str(r))
                model.addConstr(var_v[i, j, r] >= -M * var_u[j, r], 'c7-3.' + str(i) + '.' + str(j) + '.' + str(r))

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
    # print('haha')
    model.setParam('TimeLimit', timeLimit)
    model.optimize()
    # print('hehe')


k = 0
C = [[1, 1],[2, 1], [5, 1], [10, 1]]
# C = [[5,1]]
for omega in range(Omega):
    for c in C:
        if net == 'SF' or net == 'EM' or net == 'RD':
            W = Womega[omega]
        k += 1
        model = gurobipy.Model()
        model.setAttr("ModelSense", GRB.MINIMIZE)
        var_x = {}
        for i in range(N):
            for j in range(2):
                if j == 0:
                    var_x[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                else:
                    var_x[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
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
                if mtype == 'appro':
                    var_h[r, i] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='h.'+str(r)+'.'+str(i))
                else:
                    var_h[r, i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                               vtype=GRB.CONTINUOUS, name='h.'+str(r)+'.'+str(i))

                # var_h2[r, i] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='h2.'+str(r)+'.'+str(i))
        for i in range(N):
            for r in range(R):
                var_u[i, r] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS,
                                           name='u.' + str(i) + '.' + str(r))
        # for i in range(R):
        #     for r in range(R):
        #         if i == r:
        #             var_r[i,r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='r.'+str(i)+'.'+str(r))
        #         else:
        #             var_r[i,r] = model.addVar(lb=0, ub=0, vtype=GRB.BINARY, name='r.'+str(i)+'.'+str(r))

        for i in range(R):
            for r in range(R):
                for j in range(N):
                    if mtype == 'appro':
                        var_v[i, j, r] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS,
                                                      name='v.'+str(i)+'.'+str(j)+'.'+str(r))
                    else:
                        var_v[i, j, r] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                                      name='v.'+str(i)+'.'+str(j)+'.'+str(r))

                    # var_v2[i, j, r] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS,
                    #                               name='v2.'+str(i)+'.'+str(j)+'.'+str(r))

        model.update()
        model.setObjective(quicksum(model.getVarByName('x.' + str(0) + '.' + str(i)) for i in range(N)) * c[0]
                           + quicksum(model.getVarByName('x.' + str(1) + '.' + str(i)) for i in range(N)) * c[1])
        '''valid inequations'''
        model.addConstr(quicksum(var_x[j,i] for i in range(N) for j in range(2)) >= R - quicksum(var_z[r] for r in range(R)))
        for r in range(R):
            for s in range(R):
                if r != s:
                    model.addConstr(quicksum(math.fabs(W[j][r] - W[j][s]) * (var_x[0,j] + var_x[1,j])
                                             for j in range(N)) >= 1 - var_z[r] - var_z[s])
        '''========================'''
        for r in range(R):
            for s in range(R):
                if r != s:
                    model.addConstr(
                        quicksum(math.fabs(W[i][r] - W[i][s]) * model.getVarByName('x.0.' + str(i)) for i in range(N))
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

        # model.addConstr(quicksum(var_x[j,i] for i in range(N) for j in range(2)) >= R - quicksum(var_z[r] for r in range(R)))
        # for r in range(R):
        #     for s in range(R):
        #         if r != s:
        #             model.addConstr(quicksum(math.fabs(W[j][r] - W[j][s]) * (var_x[0,j] + var_x[1,j])
        #                                      for j in range(N)) >= 1 - var_z[r] - var_z[s])

        model.addConstr(quicksum(var_x[j,i] for i in range(N) for j in range(2)) >= R - quicksum(var_z[r] for r in range(R)))
        for r in range(R):
            for s in range(R):
                if r != s:
                    # model.addConstr(quicksum(math.fabs(W[j][r] - W[j][s]) * (var_x[0,j] + var_x[1,j])
                    #                          for j in range(N)) >= 1 - var_z[r] - var_z[s])
                    model.addConstr(quicksum(math.fabs(W[j][r] - W[j][s]) * (var_x[0,j] + var_x[1,j])
                                             for j in range(N)) >= 1)

        '''elementary transformation'''
        for i in range(R):
            for r in range(R):
                if i == r:
                    model.addConstr(quicksum(var_v[i, j, r] * W[j][r] for j in range(N)) == 1-var_z[r],
                                    'c5-1.' + str(i) + '.' + str(r))
                else:
                    model.addConstr(quicksum(var_v[i, j, r] * W[j][r] for j in range(N)) == 0,
                                    'c5-1.' + str(i) + '.' + str(r))

        # for i in range(R):
        #     for r in range(R):
        #         model.addConstr(var_r[i, r] == quicksum((var_v[i, j, r] - var_v2[i, j, r]) * W[j][r] for j in range(N)),
        #                         'c5.' + str(i) + '.' + str(r))
        # # for i in range(R):
        # #     model.addConstr(quicksum(var_r[i, r] for r in range(R)) <= 1, 'c6-1.' + str(i))
        # for r in range(R):
        #     model.addConstr(quicksum(var_r[i, r] for i in range(R)) == 1 - var_z[r], 'c6-2.' + str(r))

        # for i in range(N-1):
        #     model.addConstr(quicksum(var_r[i, r] - var_r[i+1, r] for r in range(R)) >= 0, 'c6-5.' + str(i))
        # for i in range(N):
        #     for j in range(N):
        #         model.addConstr(var_h[j,i] <= quicksum(W[i][r] * var_u[i, r] for r in range(R)),
        #                         'c6-3.'+str(i) +'.'+str(j))
        #         model.addConstr(var_h[i,j] <= quicksum(W[i][r] * var_u[i, r] for r in range(R)),
        #                         'c6-4.'+str(i) +'.'+str(j))


        '''definition of variable v'''
        for i in range(R):
            for j in range(N):
                for r in range(R):
                    if mtype == 'appro':
                        model.addConstr(var_v[i,j,r] <= var_h[i,j] + (1 - var_u[j,r]), 'c7-1.' + str(i)+'.'+str(j)+'.'+str(r))
                        model.addConstr(var_v[i,j,r] >= var_h[i,j] - (1 - var_u[j,r]), 'c7-2.' + str(i)+'.'+str(j)+'.'+str(r))
                        model.addConstr(var_v[i,j,r] <= var_u[j,r], 'c7-3.' + str(i)+'.'+str(j)+'.'+str(r))
                    else:
                        model.addConstr(var_v[i,j,r] <= var_h[i,j] + M*(1 - var_u[j,r]), 'c7-1.' + str(i)+'.'+str(j)+'.'+str(r))
                        model.addConstr(var_v[i,j,r] >= var_h[i,j] - M*(1 - var_u[j,r]), 'c7-2.' + str(i)+'.'+str(j)+'.'+str(r))
                        model.addConstr(var_v[i,j,r] <= M * var_u[j,r], 'c7-3.' + str(i)+'.'+str(j)+'.'+str(r))
                        model.addConstr(var_v[i,j,r] >= -M * var_u[j,r], 'c7-3.' + str(i)+'.'+str(j)+'.'+str(r))

                    # model.addConstr(var_v[i,j,r] >= -var_u[j,r], 'c7-4.' + str(i)+'.'+str(j)+'.'+str(r))

                    # model.addConstr(var_v2[i,j,r] <= var_h2[i,j] + (1 - var_u[j,r]), 'c7-5.' + str(i)+'.'+str(j)+'.'+str(r))
                    # model.addConstr(var_v2[i,j,r] >= var_h2[i,j] - (1 - var_u[j,r]), 'c7-6.' + str(i)+'.'+str(j)+'.'+str(r))
                    # model.addConstr(var_v2[i,j,r] <= var_u[j,r], 'c7-7.' + str(i)+'.'+str(j)+'.'+str(r))
                    # model.addConstr(var_v2[i,j,r] >= -var_u[j,r], 'c7-8.' + str(i)+'.'+str(j)+'.'+str(r))

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

        '''save model'''
        model.update()
        import os
        project_root = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(project_root)
        instance_path = os.path.join(project_root, 'instance')
        if not os.path.exists(instance_path+f'/train/{net}'):
            os.makedirs(instance_path+f'/train/{net}')
        model.write(instance_path+f'/train/{net}/model.{omega}.{c}.{node}.{N}.{R}.mps')
        continue

        '''solve model'''
        model.setParam(GRB.Param.LogFile, net+f'_logging.{omega}.{c}.{node}.{N}.{R}.log')
        model.setParam('TimeLimit', timeLimit)
        st = time.perf_counter()
        model.optimize()
        sol_u = [[model.getVarByName("u." + str(i) + '.' + str(r)).X for r in range(R)] for i in range(N)]
        sol_xa = [round(model.getVarByName("x.0." + str(i)).X, 0) for i in range(N)]
        sol_xp = [round(model.getVarByName("x.1." + str(i)).X, 0) for i in range(N)]
        sol_h = [[model.getVarByName("h." + str(i) + '.' + str(r)).X for r in range(N)] for i in range(R)]
        # sol_r = [[model.getVarByName("r." + str(i) + '.' + str(r)).X for r in range(R)] for i in range(R)]
        xalabel = [i for i in range(N) if round(sol_xa[i], 0) == 1]
        xplabel = [i for i in range(N) if round(sol_xp[i], 0) == 1]
        # zz = differentiation(xalabel)
        sol_z = [round(model.getVarByName("z." + str(r)).X, 0) for r in range(R)]
        zlabel1 = [r for r in range(R) if round(sol_z[r], 0) == 1]
        zlabel0 = [r for r in range(R) if round(sol_z[r], 0) == 0]
        print('xa', len(xalabel), xalabel)
        print('xp', len(xplabel), xplabel)
        print('z', zlabel0, zlabel1)
        # sub, xalabel1, xplabel1 = substitution_active(W, sol_xa, sol_xp, [[c[i] for j in range(N)] for i in range(2)])
        et = time.perf_counter()
        # print('useless sensors = ', sub)
        with open(net+f'_results.{omega}.{c}.{node}.{N}.{R}.txt','w') as file:
            file.write(str(len(xalabel)) + str(xalabel)+'\n')
            file.write(str(len(xplabel)) + str(xplabel)+'\n')
            #file.write(str(len(xalabel1)) + str(xalabel1)+'\n')
            # file.write(str(len(xplabel1)) + str(xplabel1)+'\n')
            # file.write(str(sub)+'\n')
            # file.write(str(c[0]*len(xalabel1) + c[1] * len(xplabel1)-len(sub)) + '\n')
            file.write(str(et-st))
        file.close()
        # for i in xalabel:
        #     print([W[i][j] for j in zz])
        # print()
        #
        # for i in xplabel:
        #     print([W[i][j] for j in zz])





        # rank(xalabel)

        # with open(net + '_' + str(number) +'.txt', 'a') as f:
        #     f.write('obj ' + str(model.getAttr('ObjVal')) + '\n')
        #     f.write('time and gap ' + str(model.getAttr('Runtime')) + '  ' + str(model.getAttr('MIPGap')) + '\n')
        #     f.write('xa '+str(xalabel) + '\n')
        #     f.write('xp '+str(xplabel) + '\n')
        #     f.write('z_ture '+str(zlabel0) + '\n')
        #     f.write(str(sub) + '\n\n')
        # f.close()


