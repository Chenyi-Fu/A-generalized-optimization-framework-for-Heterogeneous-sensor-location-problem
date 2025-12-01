import os.path
import pickle
from multiprocessing import Process, Queue
import gurobipy as gp
import numpy as np
import argparse
from helper import get_a_new2, gz2mps
import gzip
import json
import re


def solve_grb(filepath, log_dir, settings):
    gp.setParam('LogToConsole', 0)
    m = gp.read(filepath)

    m.Params.PoolSolutions = settings['maxsol']
    m.Params.PoolSearchMode = settings['mode']
    m.Params.TimeLimit = settings['maxtime']
    m.Params.Threads = settings['threads']
    log_path = os.path.join(log_dir, os.path.basename(filepath) + '.log')
    with open(log_path, 'w'):
        pass

    m.Params.LogFile = log_path
    m.optimize()

    sols = []
    objs = []
    solc = m.getAttr('SolCount')

    mvars = m.getVars()
    # get variable name,
    oriVarNames = [var.varName for var in mvars]

    varInds = np.arange(0, len(oriVarNames))

    for sn in range(solc):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn))
        objs.append(m.PoolObjVal)

    sols = np.array(sols, dtype=np.float32)
    objs = np.array(objs, dtype=np.float32)

    sol_data = {
        'var_names': oriVarNames,
        'sols': sols,
        'objs': objs,
    }

    return sol_data


def collect(ins_dir, q, sol_dir, log_dir, bg_dir, settings):
    while True:
        filename = q.get()
        if not filename:
            break
        filepath = os.path.join(ins_dir, filename)
        sol_data = solve_grb(filepath, log_dir, settings)
        # get bipartite graph , binary variables' indices
        A2, v_map2, v_nodes2, c_nodes2, b_vars2 = get_a_new2(filepath)
        BG_data = [A2, v_map2, v_nodes2, c_nodes2, b_vars2]

        # save data
        pickle.dump(sol_data, open(os.path.join(sol_dir, filename + '.sol'), 'wb'))
        pickle.dump(BG_data, open(os.path.join(bg_dir, filename + '.bg'), 'wb'))


def collect2(ins_dir, q, sol_dir, log_dir, bg_dir, settings):
    while True:
        filename = q.get()
        if not filename:
            break
        filepath = os.path.join(ins_dir, filename)
        # get bipartite graph , binary variables' indices
        A2, v_map2, v_nodes2, c_nodes2, b_vars2 = get_a_new2(filepath)

        if filepath.endswith('.gz'):
            tmp_path = gz2mps(filepath)
            model = gp.read(tmp_path)
            os.remove(tmp_path)
        else:
            model = gp.read(filepath)
        m_vars = model.getVars()
        oriVarNames = [var.varName for var in m_vars]
        VarMap = {var: i for i, var in enumerate(oriVarNames)}
        sols = np.zeros((1, len(oriVarNames)), dtype=np.float32)
        objs = np.zeros((1,), dtype=np.float32)

        filename0 = re.match('(.*)\.mps', filename).group(1)
        try:
            with gzip.open(ins_dir + f'_sol/{filename0}/1/{filename0}.sol.gz', 'rt') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    elif line.startswith('='):
                        objs[0] = float(line.split()[1])
                        continue
                    parts = line.split()
                    if len(parts) == 2:
                        var, val = parts
                        sols[0, VarMap[var]] = float(val)
        except FileNotFoundError:
            print(f"Solution file not found for {filename}, skip")
            return 0
            # sols = np.zeros((len(oriVarNames), 1), dtype=np.float32)

        if max(A2.shape) > 1e5: # drop too large problems
            return 0

        sol_data = {
            'var_names': oriVarNames,
            'sols': sols,
            'objs': objs,
        }
        BG_data = [A2, v_map2, v_nodes2, c_nodes2, b_vars2]

        # save data
        pickle.dump(sol_data, open(os.path.join(sol_dir, filename + '.sol'), 'wb'))
        pickle.dump(BG_data, open(os.path.join(bg_dir, filename + '.bg'), 'wb'))


if __name__ == '__main__':
    # sizes=['small','large']
    # sizes=["IP","WA","IS","CA","NNV"]
    sizes = ['MIPLIP']

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='./')
    parser.add_argument('--nWorkers', type=int, default=32)
    parser.add_argument('--maxTime', type=int, default=3600)
    parser.add_argument('--maxStoredSol', type=int, default=500)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()

    for size in sizes:

        dataDir = args.dataDir

        INS_DIR = os.path.join(dataDir, f'instance/train/{size}')

        if not os.path.isdir(f'./dataset/{size}'):
            os.mkdir(f'./dataset/{size}')
        if not os.path.isdir(f'./dataset/{size}/solution'):
            os.mkdir(f'./dataset/{size}/solution')
        if not os.path.isdir(f'./dataset/{size}/NBP'):
            os.mkdir(f'./dataset/{size}/NBP')
        if not os.path.isdir(f'./dataset/{size}/logs'):
            os.mkdir(f'./dataset/{size}/logs')
        if not os.path.isdir(f'./dataset/{size}/BG'):
            os.mkdir(f'./dataset/{size}/BG')

        SOL_DIR = f'./dataset/{size}/solution'
        LOG_DIR = f'./dataset/{size}/logs'
        BG_DIR = f'./dataset/{size}/BG'
        os.makedirs(SOL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        os.makedirs(BG_DIR, exist_ok=True)

        N_WORKERS = args.nWorkers

        # gurobi settings
        SETTINGS = {
            'maxtime': args.maxTime,
            'mode': 2,
            'maxsol': args.maxStoredSol,
            'threads': args.threads,

        }

        filenames = os.listdir(INS_DIR)

        model_set = []
        pattern = r'^(.+)\.mps'
        with open('binary-v1.test.txt', 'r') as f:
            for line in f:
                line = line.strip()
                # model_set.append(re.match(pattern, line).group(1))
                model_set.append(line)
        # filter of problems with solutions
        filenames_solution = os.listdir(f'./instance/train/{size}_sol')
        filenames_solution = [filename + '.mps.gz' for filename in filenames_solution]
        filenames = set(filenames) & set(model_set) & set(filenames_solution)
        # filenames = [filenames[1]] # only use the first instance for testing
        q = Queue()
        # add ins
        for filename in filenames:
            # if not os.path.exists(os.path.join(BG_DIR,filename+'.bg')):
            q.put(filename)
        # add stop signal
        for i in range(N_WORKERS):
            q.put(None)

        # collect2(INS_DIR, q, SOL_DIR, LOG_DIR, BG_DIR, SETTINGS)

        ps = []
        for i in range(N_WORKERS):
            p = Process(target=collect2, args=(INS_DIR, q, SOL_DIR, LOG_DIR, BG_DIR, SETTINGS))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()

        print('done')


