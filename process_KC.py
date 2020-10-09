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
    loadDir = Path('C:/Users/JobS/Dropbox/EUR/Afstuderen/Ortec - Jumbo/5. Thesis/Current results/KC')
    rawDir = loadDir / 'raws_8_10'
    os.chdir(loadDir)

    refFiles = [file for file in os.listdir(rawDir) if 'R' in file]
    refFronts2014, refFronts2015 = {}, {}
    for d, date in enumerate([datetime(2014, 11, 15), datetime(2015, 5, 15)]):
        refFrontsDict = refFronts2014 if d == 0 else refFronts2015
        planner.evaluator.set_classes(inclCurr=True, inclWeather=False, nDays=7, startDate=date)
        evaluate = planner.evaluator.evaluate

        for refFile in refFiles:
            split = refFile.split('_')
            pair = split[-1]
            with open(rawDir / refFile, 'rb') as fh:
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

    files = [file for file in os.listdir(rawDir) if 'R' not in file]

    fronts14, fronts15 = {}, {}
    for file in files:
        split = file.split('_')
        pair = split[-1]
        with open(rawDir / file, 'rb') as fh:
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
    dfBinaryHV = pd.DataFrame(columns=pairs)
    dfCoverage = pd.DataFrame(columns=pairs)

    for pair, frontTup in frontsDict.items():
        print('\r', pair, end='')
        fronts, refFronts = frontTup

        for front, refFront in zip(fronts, refFronts):
            biHV = indicators.binary_hypervolume(front, refFront)
            coverage = indicators.two_sets_coverage(front, refFront)
            dfBinaryHV = dfBinaryHV.append({pair: biHV}, ignore_index=True)
            dfCoverage = dfCoverage.append({pair: coverage}, ignore_index=True)

    for df in [dfBinaryHV, dfCoverage]:
        mean, std, minn, maxx = df.mean(), df.std(), df.min(), df.max()
        df.loc['mean'] = mean
        df.loc['std'] = std
        df.loc['min'] = minn
        df.loc['max'] = maxx

    dfCoverage.to_excel(writer, sheet_name='{}_C'.format(name))
    dfBinaryHV.to_excel(writer, sheet_name='{}_B'.format(name))


def save_fronts(name, frontsDict):

    pairs = list(frontsDict.keys())

    index = ['TT', 'FC']
    dfFronts = pd.DataFrame(columns=pairs)
    dfRefFronts = pd.DataFrame(columns=pairs)
    df = pd.DataFrame(np.random.randn(3, 8), index=['A', 'B', 'C'], columns=index)

    for pair, frontTup in frontsDict.items():
        print('\r', pair, end='')
        fronts, refFronts = frontTup

        for front, refFront in zip(fronts, refFronts):
            biHV = indicators.binary_hypervolume(front, refFront)
            coverage = indicators.two_sets_coverage(front, refFront)
            dfFronts = dfFronts.append({pair: biHV}, ignore_index=True)
            dfRefFronts = dfRefFronts.append({pair: coverage}, ignore_index=True)

    for df in [dfFronts, dfRefFronts]:
        mean, std, minn, maxx = df.mean(), df.std(), df.min(), df.max()
        df.loc['mean'] = mean
        df.loc['std'] = std
        df.loc['min'] = minn
        df.loc['max'] = maxx

    dfRefFronts.to_excel(writer, sheet_name='{}_C'.format(name))
    dfFronts.to_excel(writer, sheet_name='{}_B'.format(name))


fronts2014, fronts2015 = create_raw_dicts()

inputDict = {'2014': fronts2014, '2015': fronts2015} if len(fronts2015) > 0 else {'2014': fronts2014}

for key, _fronts in inputDict.items():
    print('\r', key, end='\n')
    compute_metrics(key, _fronts)
    save_front(key, _fronts)

writer.close()
