import indicators
import os
import pickle
import main
import pandas as pd

from datetime import datetime
from deap import tools
from pathlib import Path

planner = main.RoutePlanner(bathymetry=False, ecaFactor=1)


def create_raw_dicts():
    loadDir = Path('C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/Current results/Gulf/download_6-10/')
    gulfDir = loadDir / 'var'
    os.chdir(loadDir)

    refFiles = [file for file in os.listdir(gulfDir) if 'R' in file]
    refFronts2014, refFronts2015 = {}, {}
    for d, date in enumerate([datetime(2014, 11, 15), datetime(2015, 5, 15)]):
        refFrontsDict = refFronts2014 if d == 0 else refFronts2015
        planner.evaluator.set_classes(inclCurr=True, inclWeather=False, nDays=7, startDate=date)
        evaluate = planner.evaluator.evaluate

        for refFile in refFiles:
            split = refFile.split('_')
            pair = split[-1]
            with open(gulfDir / refFile, 'rb') as fh:
                refRawList = pickle.load(fh)
            refFronts = [refRaw['fronts'][0][0] for refRaw in refRawList]
            newRefFronts = []
            for oldFront in refFronts:
                fits = [evaluate(ind, revert=False, includePenalty=False) for ind in oldFront]
                for fit, ind in zip(fits, oldFront.items):
                    ind.fitness.values = fit
                newFront = tools.ParetoFront()
                newFront.update(oldFront.items)
                newRefFronts.append(newFront)
            refFrontsDict[pair] = newRefFronts

    files = [file for file in os.listdir(gulfDir) if 'R' not in file]

    fronts14, fronts15 = {}, {}
    for file in files:
        split = file.split('_')
        pair = split[-1]
        with open(gulfDir / file, 'rb') as fh:
            rawList = pickle.load(fh)
        fronts = [raw['fronts'][0][0] for raw in rawList]
        if '2014' in file:
            fronts14[pair] = (fronts, refFronts2014[pair])
        else:
            fronts15[pair] = (fronts, refFronts2015[pair])

    return fronts14, fronts15


writer = pd.ExcelWriter('output.xlsx')


def compute_metrics(name, frontsDict):
    pairs = list(frontsDict.keys())
    dfBinaryHV = pd.DataFrame(columns=pairs, index=range(5))
    dfCoverage = pd.DataFrame(columns=pairs, index=range(5))

    for pair, frontTup in frontsDict.items():
        print('\r', pair, end='')
        fronts, refFronts = frontTup

        for run, (front, refFront) in enumerate(zip(fronts, refFronts)):
            biHV = indicators.binary_hypervolume(front, refFront)
            coverage = indicators.two_sets_coverage(front, refFront)
            dfBinaryHV.iloc[run][pair] = biHV
            dfCoverage.iloc[run][pair] = coverage

    for df in [dfBinaryHV, dfCoverage]:
        mean, std, minn, maxx = df.mean(), df.std(), df.min(), df.max()
        df.loc['mean'] = mean
        df.loc['std'] = std
        df.loc['min'] = minn
        df.loc['max'] = maxx

    dfCoverage.to_excel(writer, sheet_name='{}_C'.format(name))
    dfBinaryHV.to_excel(writer, sheet_name='{}_B'.format(name))


fronts2014, fronts2015 = create_raw_dicts()

for key, _fronts in {'2014': fronts2014, '2015': fronts2015}.items():
    print('\r', key, end='\n')
    compute_metrics(key, _fronts)

writer.close()
