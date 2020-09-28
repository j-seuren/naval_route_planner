import main
import skopt

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from skopt import plots, load


DIR = 'D:/'

SPACE = [
    skopt.space.Integer(50, 300, name='gen'),  # Minimal nr. generations
    skopt.space.Integer(10, 50, name='maxGDs'),
    skopt.space.Real(1e-7, 1e-4, name='minVar'),  # Minimal variance generational distance
    skopt.space.Integer(2, 20, name='nMutations'),
    skopt.space.Categorical([4 * i for i in range(1, 101)], name='n'),
    skopt.space.Real(0.5, 1.0, name='cxpb'),
    skopt.space.Real(0.2, 0.7, name='mutpb'),
    # skopt.space.Integer(5, 500, name='nBar', prior='log-uniform'),
    # skopt.space.Integer(1, 10, name='recomb'),
    # skopt.space.Integer(1, 20, name='fails'),
    # skopt.space.Integer(1, 20, name='moves'),
    # skopt.space.Real(0.0001, 10.0, name='widthRatio', prior='log-uniform'),
    # skopt.space.Real(0.0001, 10.0, name='radius', prior='log-uniform'),
    # skopt.space.Real(0.1, 5.0, name='delFactor')
    ]


def show_results(fp, timestamp):
    res = load(fp)
    print(res)

    # Print results
    best_params = {par.name: res.x[i] for i, par in enumerate(SPACE)}
    print('best result: ', res.fun)
    print('best parameters: ', best_params)

    # Plot results
    with PdfPages('/output/tuning/figures/{}.pdf'.format(timestamp)) as pdf:
        plots.plot_evaluations(res)
        pdf.savefig()
        plots.plot_objective(res)
        pdf.savefig()
        plots.plot_convergence(res)
        pdf.savefig()


show_results('D:/JobS/Downloads/checkpoint_2.pkl', datetime.now().strftime('%m%d%H%M'))
