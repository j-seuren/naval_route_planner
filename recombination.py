import numpy as np
import random

from haversine import haversine


def crossover(toolbox, ind1, ind2):
    u_list = [[i, u] for i, u in enumerate(ind1[:, 0:2])][1:-2]
    random.shuffle(u_list)
    v_list = [[i, v] for i, v in enumerate(ind2[:, 0:2])][2:-1]
    random.shuffle(v_list)
    while u_list:
        u_idx, u = u_list.pop()
        shuffled_v = v_list[:]
        while shuffled_v:
            v_idx, v = shuffled_v.pop()
            if toolbox.edge_feasible(u, v):
                child = np.append(ind1[:u_idx+1], ind2[v_idx:], axis=0)
                return child
    print('No crossover performed')
    return False
