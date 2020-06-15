import random


def crossover(toolbox, ind1, ind2):
    u_list = [[i, row[0]] for i, row in enumerate(ind1)][1:-2]
    random.shuffle(u_list)
    v_list = [[i, row[0]] for i, row in enumerate(ind2)][2:-1]
    random.shuffle(v_list)
    while u_list:
        u_idx, u = u_list.pop()
        shuffled_v = v_list[:]
        while shuffled_v:
            v_idx, v = shuffled_v.pop()
            if toolbox.edge_feasible(u, v):
                child = ind1[:u_idx+1] + ind2[v_idx:]
                return child
    print('No crossover performed')
    return False
