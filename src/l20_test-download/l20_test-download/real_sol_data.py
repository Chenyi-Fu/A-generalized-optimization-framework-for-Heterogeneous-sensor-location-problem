import os
import pickle
import numpy as np
import re
from typing import Dict, Optional

def load_sol_data(path):
    """
    从指定路径加载 sol_data（pickle 文件）。
    返回一个字典或在出错时返回 None。
    """
    if not os.path.exists(path):
        print(f"file not found: {path}")
        return None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"failed to load pickle: {e}")
        return None
    except Exception as e:
        print(f"unexpected error: {e}")
        return None

    # 验证并标准化字段
    var_names = data.get("var_names")
    sol = data.get("sol")
    obj = data.get("obj", None)

    if sol is not None:
        sol = np.asarray(sol, dtype=np.float32)

    return {"var_names": var_names, "sol": sol, "obj": obj}

# code to load all sols from a directory with specific cost ratio
def load_all_sols(directory: str = "logs/RD_20_90/RD_GRB_Predect&Search", ratio = 1) -> Dict[str, Optional[dict]]:
    """
    扫描 `directory`，匹配以 `SF.` 开头、包含字面子串 `[ratio, 1]` 并以 `.p` 结尾的文件，
    使用已有的 load_sol_data 加载每个文件，返回 {filename: data} 的字典。
    """
    pattern = re.compile(r"^" + re.escape("RD.") + r".*" + re.escape(f"[{ratio}, 1]") + r".*\.p$")
    results: Dict[str, Optional[dict]] = {}

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"directory not found: {directory}")

    for name in os.listdir(directory):
        if pattern.match(name):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                try:
                    results[name] = load_sol_data(path)  # 复用已定义的函数
                except Exception as e:
                    # 保持简短，记录 None 表示加载失败
                    results[name] = None

    return results

sols_1_all = load_all_sols(ratio=10, directory= "logs/RD_20_90/RD_20_90_GRB_Predect&Search")

arcs = 60

pass_all = [0] * len(sols_1_all)
active_all = [0] * len(sols_1_all)
for idx, (filename, sols) in enumerate(sols_1_all.items()):
    if sols is None:
        continue
    for i, name in enumerate(sols['var_names']):
        if name.startswith('x.1.') and sols['sol'][i] > 0.5:
            pass_all[idx] += 1
        elif name.startswith('x.0.') and sols['sol'][i] > 0.5:
            active_all[idx] += 1
print(sum(pass_all))
print(sum(active_all))
np.mean(pass_all)

np.float64(25.6)
np.mean(active_all)
np.float64(39.6)

# costs = [
#     [66,67,65,67,64],
#     [131,120,120,120,119],
#     [279,280,278,278,267],
#     [524,513,544,538,511]
# ]
# costs = np.array(costs)
#
# costs = [
#     [ 52, 56, 61, 55, 60 ],
#     [ 98, 102, 105, 102, 104 ],
#     [ 215, 224, 224, 225, 227 ],
#     [ 407, 417, 434, 423, 427 ]
# ]
# costs = np.array(costs)
#
# cost_save = costs[1:, :] / (np.array([2,5,10]).reshape(-1,1) * costs[0, :]) - 1
# np.mean(cost_save, axis=1)
# array([-0.09847658, -0.21276044, -0.25608637])
