import main
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import skopt
import time

from datetime import datetime
from pathlib import Path
# from skopt import plots
from skopt.callbacks import CheckpointSaver
from support import locations

DIR = Path('D:/')
N_CALLs = 200
N_POINTS = 10
DEPART = datetime(2016, 1, 1)
CURRENT = False
BATHYMETRY = False
inputParameters = {'gen': 130,
                   'segLengthF': 100,
                   'mutationOperators': ['speed', 'insert', 'move', 'delete'],
                   'mutpb': 0.7}
timestamp = datetime.now().strftime('%m%d%H%M')
tuningDir = DIR / 'output/tuning' / '{}_i{}_calls_{}points'.format(timestamp, N_CALLs, N_POINTS)
figDir = tuningDir / 'figures'
os.makedirs(figDir)
excelFP = tuningDir / '{}_best_parameters_{}iterations.xlsx'.format(timestamp, 4)
writer = pd.ExcelWriter(excelFP)
DF = None

for iteration in range(4):
    SPACE = [
        # skopt.space.Integer(50, 100, name='gen'),  # Minimal nr. generations
        skopt.space.Integer(5, 45, name='maxGDs'), skopt.space.Real(1e-6, 1e-4, name='minVar'),
        # Minimal variance generational distance
        skopt.space.Integer(2, 15, name='nMutations'),
        skopt.space.Categorical([4 * i for i in range(1, 101)], name='n'), skopt.space.Real(0.5, 1.0, name='cxpb'),
        skopt.space.Real(0.2, 0.7, name='mutpb'), # skopt.space.Integer(5, 500, name='nBar', prior='log-uniform'),
        # skopt.space.Integer(1, 10, name='recomb'),
        # skopt.space.Integer(1, 20, name='fails'),
        # skopt.space.Integer(1, 20, name='moves'),
        # skopt.space.Real(0.001, 10., name='widthRatio'),
        # skopt.space.Real(0.001, 10., name='radius'),
        # skopt.space.Real(0.1, 5.0, name='delFactor')
    ]



    def process_results(res, oldDF):
        # Save best parameters
        best_params = {par.name: res.x[i] for i, par in enumerate(SPACE)}
        print('best parameters: ', best_params)

        newDF = pd.DataFrame(best_params, index=[iteration])
        df = oldDF.append(newDF) if iteration > 0 else newDF

        # # Save tuning results
        # resFP = tuningDir / '{}.gz'.format(iteration)
        # skopt.dump(_res, resFP, compress=9, store_objective=False)
        # print('Saved tuning results to', resFP)

        # # Plot results
        # figFP = figDir / '{}_'.format(timestamp)
        # plots.plot_evaluations(res)
        # plt.savefig(figFP.as_posix() + 'eval.pdf')
        # plots.plot_objective(res)
        # plt.savefig(figFP.as_posix() + 'obj.pdf')
        # plots.plot_convergence(res)
        # plt.savefig(figFP.as_posix() + 'conv.pdf')
        # print('Saved figures to', figDir)

        return df

    res = skopt.load('D:/JobS/Downloads/drive-download-20201005T123221Z-001/{}.gz'.format(iteration))

    DF = process_results(res, DF)
    DF.to_excel(writer)
print('written df to', writer.path)
writer.close()