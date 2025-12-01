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

Data = sensor_data.data()
net = 'ND'
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

# c = [0.5, 1]
# c = [10, 1]
M = N * 10
timeLimit = 3600*12
# softlimit = 3600*2
bestub = 100000

c = [10,1]
# 参数设置
num_locations = N  # 候选位置数量
# budget = 50  # 预算限制
# costs = np.random.randint(5, 20, num_locations)  # 每个位置的成本
# coverages = np.random.randint(10, 50, num_locations)  # 每个位置的覆盖范围


def rank(individual, label, id = None):
    if id != None:
        for i in id:
            individual[i] = 0
    H = [[W[i][r]* label[r] for r in range(R)] for i in range(N)]
    rw = np.linalg.matrix_rank(H)
    if id != None:
        for i in id:
            individual[i] = 1
    return rw


def differentiation_index(individual, id=None, cover=0):
    if id != None:
        for i in id:
            individual[i] = 0

    label = [[min(1,sum([np.abs(W[i][r] - W[i][s]) * individual[i] for i in range(N)])) for s in range(R)] for r in range(R)]
    # print([sum(label[r]) for r in range(R)])
    if cover == 0:
        z = [0 if sum(label[r]) == R - 1 else 1 for r in range(R)]
        if id != None:
            for i in id:
                individual[i] = 1
    else:
        label2 = [min(1,sum([W[i][r] * individual[i] for i in range(N)])) for r in range(R)]
        z = [0 if sum(label[r]) == R - 1 and np.abs(label2[r] - 1) < 0.1 else 1 for r in range(R)]
        # print(z)
    return z



def intial_feasible_solution():
    sc, obj = MIPmodel()
    arr =[]
    for i in range(N):
        if sc[i] > 0.9:
            arr.append(i)
    arr = np.array(arr)
    scnew = []
    k = 1
    for i in arr:
        print(f'iteration {k} begins')
        k+=1
        z = differentiation_index(sc,id=[i],cover=1)
        rw = rank(sc, z, id=[i])

        # print(i,sum(z),rw,R-fao)
        if np.abs(rw - sum(z)) <= 0.1:
            print(f'sc {i} can be removed as rank is {rw}')
            scnew.append(i)
            # z = differentiation(sc)
        else:
            print(f'sc {i} cannot be removed as rank is {rw}')
        print()
    print(sc)
    graph = [[0 for j in range(N)] for i in range(N)]
    xa_set = [i for i in range(N) if sc[i]>0.9]
    for i in scnew:
        for j in scnew:
            if i<j:
                z = differentiation_index(sc, id=[i,j], cover=1)
                rw = rank(sc, z, id=[i,j])

                # print(i,sum(z),rw,R-fao)
                if np.abs(rw - sum(z)) <= 0.1:
                    print(f'sc {i} and {j} can be removed as rank is {rw} ({sum(z)})')
                    graph[i][j] = 1
                    graph[j][i] = 1
                    # z = differentiation(sc)
                else:
                    print(f'sc {i} and {j} cannot be removed as rank is {rw} ({sum(z)})')
                print()
    return xa_set, graph


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


def revert(xa, xp, xpnew, route):
    print(route)
    print(xa)
    print(xp)
    print(xpnew)
    print()
    for i in xa:
        if sum( [W[i][r] for r in route])>0:
            print([W[i][r] for r in route])
    print()
    for i in xp:
        if sum( [W[i][r] for r in route])>0:
            print([W[i][r] for r in route])
    print()
    for i in xpnew:
        if sum( [W[i][r] for r in route])>0:
            print([W[i][r] for r in route])
    print()
    WW = []
    for i in range(N):
        WW.append([W[i][r] for r in route])
    RR = len(route)
    # if net == 'SF' or net == 'EM':
    #     W = Womega[omega]
    for r in route:
        for s in route:
            if r!=s:
                if sum([np.abs(W[i][r]-W[i][s]) for i in range(N) if i in xp or i in xpnew]) == 0:
                    print('hahaha=',r,s)
    model = gurobipy.Model()
    model.setAttr("ModelSense", GRB.MINIMIZE)
    var_x = {}
    for i in range(N):
        for j in range(2):
            if j == 0:
                if i in xa:
                    var_x[j, i] = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                elif i in xp or i in xpnew:
                    var_x[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                else:
                    var_x[j, i] = model.addVar(lb=0, ub=0, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
            else:
                if i in xp:
                    var_x[j, i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                else:
                    var_x[j, i] = model.addVar(lb=0, ub=0, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
        # if i in xa:
        #     var_x[0,i].Start = 1
        # elif i in xp:
        #     var_x[1,i].Start = 1

    var_z = {}
    for r in range(RR):
        var_z[r] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z.' + str(r))
    var_u = {}
    var_h = {}
    # var_h2 = {}
    var_r = {}
    var_v = {}
    # var_v2 = {}
    for r in range(RR):
        for i in range(N):
            var_h[r, i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                           vtype=GRB.CONTINUOUS, name='h.' + str(r) + '.' + str(i))

            # var_h2[r, i] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='h2.'+str(r)+'.'+str(i))
    for i in range(N):
        for r in range(RR):
            var_u[i, r] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS,
                                       name='u.' + str(i) + '.' + str(r))

    for i in range(RR):
        for r in range(RR):
            for j in range(N):
                var_v[i, j, r] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                              name='v.' + str(i) + '.' + str(j) + '.' + str(r))

    model.update()
    model.setObjective(quicksum(model.getVarByName('x.' + str(0) + '.' + str(i)) for i in range(N)) * c[0]
                       + quicksum(model.getVarByName('x.' + str(1) + '.' + str(i)) for i in range(N)) * c[1])
    '''valid inequations'''
    model.addConstr(
        quicksum(var_x[j, i] for i in range(N) for j in range(2)) >= RR - quicksum(var_z[r] for r in range(RR)))

    for r in range(RR):
        for s in range(RR):
            if r != s:
                model.addConstr(quicksum(math.fabs(WW[j][r] - WW[j][s]) * (var_x[0, j] + var_x[1, j])
                                         for j in range(N)) >= 1 - var_z[r] - var_z[s])
    '''========================'''
    for r in range(RR):
        for s in range(RR):
            if r != s:
                model.addConstr(
                    quicksum(
                        math.fabs(WW[i][r] - WW[i][s]) * model.getVarByName('x.0.' + str(i)) for i in range(N))
                    >= model.getVarByName('z.' + str(r)), 'c1.' + str(r) + '.' + str(s))
    for r in range(RR):
        model.addConstr(quicksum(
            WW[i][r] * (model.getVarByName('x.0.' + str(i)) + model.getVarByName('x.1.' + str(i))) for i in
            range(N)) >= 1, 'c2-1.' + str(r))
        model.addConstr(quicksum(WW[i][r] * model.getVarByName('x.0.' + str(i)) for i in range(N)) >= var_z[r],
                        'c2-2.' + str(r))
    for i in range(N):
        model.addConstr(quicksum(model.getVarByName('x.' + str(j) + '.' + str(i)) for j in range(2)) <= 1,
                        'c3.' + str(i))

    model.addConstr(
        quicksum(var_x[j, i] for i in range(N) for j in range(2)) >= RR - quicksum(var_z[r] for r in range(RR)))
    for r in range(RR):
        for s in range(RR):
            if r != s:
                # model.addConstr(quicksum(math.fabs(W[j][r] - W[j][s]) * (var_x[0,j] + var_x[1,j])
                #                          for j in range(N)) >= 1 - var_z[r] - var_z[s])
                model.addConstr(quicksum(math.fabs(WW[j][r] - WW[j][s]) * (var_x[0, j] + var_x[1, j])
                                         for j in range(N)) >= 1)

    '''elementary transformation'''
    for i in range(RR):
        for r in range(RR):
            if i == r:
                model.addConstr(quicksum(var_v[i, j, r] * WW[j][r] for j in range(N)) == 1 - var_z[r],
                                'c5-1.' + str(i) + '.' + str(r))
            else:
                model.addConstr(quicksum(var_v[i, j, r] * WW[j][r] for j in range(N)) == 0,
                                'c5-1.' + str(i) + '.' + str(r))

    '''definition of variable v'''
    for i in range(RR):
        for j in range(N):
            for r in range(RR):
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
        for r in range(RR):
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
    sol_xa = [i for i in range(N) if round(model.getVarByName("x.0." + str(i)).X, 0) >0.9 and i not in xa]
    sol_xp = [i for i in range(N) if round(model.getVarByName("x.1." + str(i)).X, 0)>0.9]
    obj = model.objVal
    return sol_xa, sol_xp, obj

def passivelocation(xa_set,xp_set):
    # if net == 'SF' or net == 'EM':
    #     W = Womega[omega]
    model = gurobipy.Model()
    model.setAttr("ModelSense", GRB.MINIMIZE)
    var_x = {}
    for i in range(N):
        for j in range(2):
            if j==0:
                if i in xa_set:
                    var_x[j, i] = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                else:
                    var_x[j, i] = model.addVar(lb=0, ub=0, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
            else:
                if i in xp_set:
                    var_x[j, i] = model.addVar(lb=1, ub=1, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
                else:
                    var_x[j, i] = model.addVar(lb=0, ub=0, vtype=GRB.BINARY, name='x.' + str(j) + '.' + str(i))
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
    sol_xa = [round(model.getVarByName("x.0." + str(i)).X, 0) for i in range(N)]
    sol_xp = [round(model.getVarByName("x.1." + str(i)).X, 0) for i in range(N)]
    obj = model.objVal
    return sol_xa, sol_xp, obj

def find_maximum_clique(graph):
    n = len(graph)
    vertices = list(range(n))

    max_clique = []

    def is_clique(current_clique):
        # 约束函数: 判断给定的顶点集合是否构成一个团(完全子图)
        for i in range(len(current_clique)):
            for j in range(i + 1, len(current_clique)):
                if not graph[current_clique[i]][current_clique[j]]:
                    return False

        return True

    def bound(current_clique, vertices):
        # 限界函数
        return len(current_clique) + len(vertices)

    def backtrack(vertices, current_clique):
        nonlocal max_clique

        if not vertices:
            if len(current_clique) > len(max_clique):
                max_clique.clear()
                max_clique.extend(current_clique)

            return

        vertex = vertices.pop(0)
        current_clique.append(vertex)

        neighbors = []
        for v in vertices:
            if graph[vertex][v]:
                neighbors.append(v)

        # 选择当前顶点并加入团
        if is_clique(current_clique):
            backtrack(neighbors, current_clique)

        # 恢复回溯前状态
        current_clique.pop()

        # 不选择当前顶点
        if bound(current_clique, vertices) > len(max_clique):
            backtrack(vertices, current_clique)

    backtrack(vertices, [])

    return max_clique


for ii in range(Omega):
    if net == 'SF' or net == 'EM':
        W = Womega[ii]
    starttime = time.perf_counter()
    # xa, obj = MIPmodel()
    # for i in [0, 15, 54, 61]:
    #     print(i, [W[i][r] for r in [13, 49, 97, 9]])
    # print()
    xa_set, graph = intial_feasible_solution()


    xp_set = find_maximum_clique(graph)
    print(xa_set)
    print(xp_set)
    # xa_set = [0, 1, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 19, 22, 24, 26, 27, 28, 31, 34, 37, 38, 40, 41, 42, 43, 44, 45, 46, 49,
    #  50, 53, 54, 56, 58, 61, 63, 65, 66, 67, 69, 71, 72]
    # xp_set = [0, 1, 5, 6, 8, 13, 15, 24, 26, 28, 31, 34, 37, 38, 40, 41, 43, 44, 46, 50, 54, 56, 58, 61, 63, 66, 67, 71, 72]
    xa_set_need = []
    xp_set_need = []
    for i in xa_set:
        if i not in xp_set:
            xa_set_need.append(i)

    label3 = []
    while 1:
        label = [[min(1,sum([np.abs(W[i][r] - W[i][s]) for i in xa_set_need])) for s in range(R)] for r in range(R)]
        label2 = [min(1, sum([W[i][r] for i in xa_set_need])) for r in range(R)]
        label4 = []
        undiff_but_cover = []
        for r in range(R):
            if r not in label3 and r not in label4:
                temp = [s for s in range(R) if label[r][s] == 0 and r!=s and label2[r]==1]
                temp.append(r)
                if len(temp) > 1:
                    undiff_but_cover.append(temp)
                    label4.extend(temp)
        if undiff_but_cover == []:
            label_cover = [min(1, sum([W[i][r] for i in xa_set_need])) for r in range(R)]
            undiff_but_cover = [r for r in range(R) if label_cover[r] == 0]
            print(undiff_but_cover)
            xa_new, xp_new, objnew = revert(xa_set_need, xp_set, xp_set_need, undiff_but_cover)
            xa_set_need.extend(xa_new)
            xp_set_need.extend(xp_new)
            xp_set_need = [i for i in xp_set_need if i not in xa_set_need]
            xp_set = [i for i in xp_set if i not in xa_new and i not in xp_new]
            print('new active sensor is ', xa_new)
            print('new passive sensor is ', xp_new)
            print('current active sensor location is ',xa_set_need)
            print('current passive sensor location is', xp_set_need)
            print('not allocated sensors are ', xp_set)
            break
        else:
            label3.extend(undiff_but_cover[0])
            print('haha = ',undiff_but_cover)
            temp = undiff_but_cover[0]
            print(temp)
            for i in xa_set_need:
                if sum([W[i][r] for r in temp]) > 0:
                    print(i, [W[i][r] for r in temp])
            for i in xp_set:
                if sum([W[i][r] for r in temp]) > 0:
                    print(i, [W[i][r] for r in temp])
            for i in xp_set_need:
                if sum([W[i][r] for r in temp]) > 0:
                    print(i, [W[i][r] for r in temp])
            xa_new, xp_new, objnew = revert(xa_set_need, xp_set, xp_set_need, undiff_but_cover[0])
            xa_set_need.extend(xa_new)
            xp_set_need.extend(xp_new)
            xp_set_need = [i for i in xp_set_need if i not in xa_set_need]
            xp_set = [i for i in xp_set if i not in xa_new and i not in xp_new]
            print('new active sensor is ', xa_new)
            print('new passive sensor is ', xp_new)
            print('current active sensor location is ',xa_set_need)
            print('current passive sensor location is', xp_set_need)
            print('not allocated sensors are ', xp_set)
    undiff_but_cover = []
    label3 = []
    for r in range(R):
        if r not in label3:
            temp = [s for s in range(R) if label[r][s] == 0 and r!=s and label2[r] == 1]
            if len(temp) > 0:
                undiff_but_cover.append(temp)
    print(undiff_but_cover)
    label = [sum([min(1, sum([np.abs(W[i][r] - W[i][s]) for i in xa_set_need])) for s in range(R)]) for r in range(R)]
    label_diff = [1 if label[r] == R-1 else 0 for r in range(R)]
    label_cover = [min(1, sum([W[i][r] for i in xa_set_need])) for r in range(R)]
    print([r for r in range(R) if label_diff[r] == 0])
    print([r for r in range(R) if label_cover[r] == 0])
    Wnew = [[W[i][r] for r in range(R) if label_diff[r]==0 or label_cover == 0] for i in range(N) if i in xp_set_need or i in xa_set_need]
    for i in range(len(Wnew)):
        if sum(Wnew[i]) > 0:
            print(Wnew[i])
    # rw = np.linalg.matrix_rank(Wnew)
    # n = len(Wnew[0])
    # print(rw, n)
    # obj = passivelocation(xa_set_need, xp_set)
    endtime = time.perf_counter()

    print("Best Solution (active):", [1 if i in xa_set_need else 0 for i in range(N)])
    print("Best Solution (passive):", [1 if i in xp_set_need else 0 for i in range(N)])
    print("Best Fitness:", len(xa_set_need)*c[0] + len(xp_set_need)*c[1])
    print("computational time:", endtime-starttime)
