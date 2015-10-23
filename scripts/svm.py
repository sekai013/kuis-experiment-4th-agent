# Modules
import sys
import click
import numpy
import cvxopt
from math import exp
import matplotlib.pyplot as pyplot

# Main Class
class SVM(object):

    DELTA = 1.0e-6

    MIN_PARAM = {
            'poly': 1,
            'gauss': 1.0
            }

    MAX_PARAM = {
            'poly': 10,
            'gauss': 15.0
            }

    def __init__(self, options):
        self.kernel_mode = options['kernel']
        self.datafile = options['datafile']
        self.do_plot = options['plot']
        self.debug = options['debug']
        self.fold = options['fold']

        cvxopt.solvers.options['show_progress'] = self.debug

        self.kernel = self.get_kernel(self.kernel_mode)

        try:
            data = numpy.loadtxt(self.datafile)
            self.clas = map(lambda d : d[-1], data)
            self.data = map(lambda d : d[:-1], data)
        except IOError:
            click.echo('Failed to open file %s' % self.datafile)
            exit()

    def run(self):
        if self.kernel_mode == 'linear' or len(self.data) < 10:
            self.calc_classifier(self.kernel_mode, self.data, self.clas, self.kernel, self.debug, self.do_plot, show_result=True)
        else:
            self.decide_kernel_parameter(
                    self.MIN_PARAM[self.kernel_mode],
                    self.MAX_PARAM[self.kernel_mode]
                    )
            optimal_kernel = lambda x, y : self.kernel(x, y, self.parameter)
            self.calc_classifier(self.kernel_mode, self.data, self.clas, optimal_kernel, self.debug, self.do_plot, show_result=True)

    def decide_kernel_parameter(self, min_param, max_param):
        if self.kernel_mode == 'poly':
            click.echo('Searching Optimal Parameter for Polynomial Kernel')
            candidates = range(min_param, max_param+1)
            trial_msg_tpl = 'Trying d = %d ...'
            result_msg_tpl = 'Optimal Parameter: d = %d'
        else:
            click.echo('Searching Optimal Parameter for Gauss Kernel')
            trial_msg_tpl = 'Trying sigma = %f ...'
            candidates = numpy.linspace(min_param, max_param, 50)
            result_msg_tpl = 'Optimal Parameter: sigma = %f'

        max_accuracy = 0
        optimal_param = -1

        for c in candidates:
            click.echo(trial_msg_tpl % c)

            kernel = lambda x, y : self.kernel(x, y, c)
            accuracy = self.cross_validate(self.kernel_mode, self.data, self.clas, kernel, debug=self.debug, fold=self.fold)

            click.echo('Accuracy: %f' % accuracy)
            click.echo('Done.')

            if max_accuracy < accuracy:
                max_accuracy = accuracy
                optimal_param = c

        click.echo()
        click.echo(result_msg_tpl % optimal_param)
        click.echo('Estimated Accuracy: %f' % max_accuracy)
        click.echo()
        self.parameter = optimal_param

    def print_debug(self, item):
        if self.debug:
            click.echo(item)

    @classmethod
    def calc_classifier(cls, kernel_mode, data, clas, kernel, debug=False, do_plot=False, show_result=False):

        try:
            solution = cls.solve_qp(data, clas, kernel, debug)
            if solution['status'] == 'unknown':
                raise BaseException, 'Couldn\'t solve program'
            lagrangians = solution['x']
         
            if debug:
                click.echo('Lagrangians:')
                click.echo(lagrangians)
         
            if kernel_mode == 'linear':
                weight = [0] * len(data[0])
         
                for i in range(len(lagrangians)):
                    if lagrangians[i] > cls.DELTA:
                        weight = map(lambda j : weight[j] + lagrangians[i] * clas[i] * data[i][j], range(len(data[0])))
         
                threshold = cls.threshold_function(kernel_mode, data, clas, lagrangians, kernel, weight=weight)
         
                if show_result:
                    click.echo('Weight: %s' % weight)
                    click.echo('Threshold: %f' % threshold)
         
                if do_plot and len(data[0]):
                    cls.plot(kernel_mode, data, clas, kernel, threshold, weight=weight)

                class_func = cls.get_classifier_function(kernel_mode)
                return lambda x : class_func(x, weight, kernel, threshold)
         
            else:
                threshold = cls.threshold_function(kernel_mode, data, clas, lagrangians, kernel)
         
                if show_result:
                     click.echo('p:')
                     click.echo(map(lambda i : lagrangians[i] * clas[i] if lagrangians[i] > cls.DELTA else 0, range(len(lagrangians))))
                     click.echo('Threshold t: %f' % threshold)
                     click.echo('Classifier: f(x, p, t) = sign ( sum(p_i * Kernel(x, x_i)) - t )')
         
                if do_plot and len(data[0]):
                    cls.plot(kernel_mode, data, clas, kernel, threshold, lagrangians=lagrangians)
         
                class_func = cls.get_classifier_function(kernel_mode)
                return lambda x : class_func(x, data, clas, lagrangians, kernel, threshold)

        except BaseException:
            return None


    @classmethod
    def get_kernel(cls, kernel_mode):
        error_msg_tpl = 'Size of x is %d but size of y is %d'

        def linear(x, y):
            if len(x) != len(y):
                raise BaseException, error_msg_tpl % (len(x), len(y))

            inner_product = 0

            for i in range(len(x)):
                inner_product += x[i] * y[i]

            return inner_product

        def poly(x, y, dimension=2):
            return (1 + linear(x, y)) ** dimension

        def gauss(x, y, sigma=10):
            if len(x) != len(y):
                raise BaseException, error_msg_tpl % (len(x), len(y))

            norm = sum(map(lambda i : (x[i] - y[i]) ** 2, range(len(x))))
            return exp(- norm / (2 * (sigma ** 2)))

        KERNELS = {
                'linear': linear,
                'poly': poly,
                'gauss': gauss
                }

        try:
            if kernel_mode.lower() in KERNELS:
                return KERNELS[kernel_mode.lower()]
            else:
                raise BaseException, '''Invalid Kernel: %s
use `svm --help` to check usage.''' % kernel_mode
        except BaseException, e:
            click.echo(e)
            exit()

    @classmethod
    def threshold_function(cls, kernel_mode, data, clas, lagrangians, kernel, weight=[]):

        count = 0
        theta = 0

        if kernel_mode == 'linear':

            for i in range(len(lagrangians)):
                if lagrangians[i] > cls.DELTA:
                    count += 1
                    theta += kernel(weight, data[i]) - clas[i]

        else:

            for i in range(len(lagrangians)):
                if lagrangians[i] > cls.DELTA:
                    count += 1
                    inner_product = 0

                    for j in range(len(lagrangians)):
                        inner_product += lagrangians[j] * clas[j] * kernel(data[i], data[j])

                    theta += inner_product - clas[i]

        return theta / count if count != 0 else -sum(clas)/len(clas)

    @classmethod
    def get_classifier_function(cls, kernel_mode):

        def linear(x, weight, kernel, threshold):
            return kernel(x, weight) - threshold

        def nonlinear(x, data, clas, lagrangians, kernel, threshold):
            c = 0

            for i in range(len(lagrangians)):
                c += lagrangians[i] * clas[i] * kernel(x, data[i])

            return c - threshold

        if kernel_mode == 'linear':
            return linear
        else:
            return nonlinear

    @classmethod
    def plot(cls, kernel_mode, data, clas, kernel, threshold, lagrangians=[], weight=[]):
        class_func = cls.get_classifier_function(kernel_mode)

        # Plot data

        for i in range(len(data)):
            if clas[i] == 1:
                pyplot.plot(data[i][0], data[i][1], 'rx')
            else:
                pyplot.plot(data[i][0], data[i][1], 'bx')

        # Plot Classifier

        min_cell = min(map(lambda d : min(d), data))
        max_cell = max(map(lambda d : max(d), data))
        padding = (min_cell + max_cell) / 15.0
        m, M = min_cell - padding, max_cell + padding

        X1, X2 = numpy.meshgrid(numpy.linspace(m, M, 50),
                numpy.linspace(m, M, 50))
        width, height = X1.shape

        X1.resize(X1.size)
        X2.resize(X2.size)

        if kernel_mode == 'linear':
            Z = numpy.array([
                class_func(
                    numpy.array([x1, x2]),
                    weight, kernel, threshold
                    ) for (x1, x2) in zip(X1, X2)
                ])
        else:
            Z = numpy.array([
                class_func(
                    numpy.array([x1, x2]),
                    data, clas, lagrangians,
                    kernel, threshold
                    ) for (x1, x2) in zip(X1, X2)
                ])

        X1.resize((width, height))
        X2.resize((width, height))
        Z.resize((width, height))

        pyplot.contour(
                X1, X2, Z, [0.0],
                colors='k', linewidths=1, origin='lower'
                )

        pyplot.show()

    @classmethod
    def solve_qp(cls, data, clas, kernel, debug=False):
        # Minimize  : (1/2)(x^T)(Q)(x) + (q^t)(x)
        # Subject to: (A)(x) >= b, (A_0)(x) = b_0

        datasize = len(data)

        # Setup matrix and vector

        Q = numpy.zeros((datasize, datasize))
  
        for i in range(datasize):
            for j in range(datasize):
                Q[i][j] = clas[i] * clas[j] * kernel(data[i], data[j])
  
        q = - numpy.ones(datasize)
  
        A = numpy.diag([-1.0] * datasize)
        b = numpy.zeros(datasize)
  
        A_0 = clas
        b_0 = 0.0

        # Convert matrix and vector to cvxopt.matrix object
  
        Q = cvxopt.matrix(Q)
        q = cvxopt.matrix(q)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
        A_0 = cvxopt.matrix(A_0, (1, datasize))
        b_0 = cvxopt.matrix(b_0)

        # Print parameters if debug mode

        if debug:
            click.echo('Q:')
            click.echo(Q)
            click.echo('q:')
            click.echo(q)
            click.echo('A:')
            click.echo(A)
            click.echo('b:')
            click.echo(b)
            click.echo('A_0:')
            click.echo(A_0)
            click.echo('b_0:')
            click.echo(b_0)

        # Solve cone quadratic programming and return solution

        return cvxopt.solvers.qp(Q, q, A, b, A_0, b_0)

    @classmethod
    def cross_validate(cls, kernel_mode, data, clas, kernel, fold=10, debug=False):
        datasize = len(data)
        unitsize = datasize / fold

        result = 0

        for i in range(fold):
            # select data[i*unitsize:(i+1)*unitsize] for test data
            # and the others for training data
            i *= unitsize
            data_i = [x for (x, j) in zip(data, range(len(data))) if j not in range(i, i+unitsize)]
            clas_i = [x for (x, j) in zip(clas, range(len(clas))) if j not in range(i, i+unitsize)]

            test_data_i = data[i:i+unitsize]
            test_clas_i = clas[i:i+unitsize]

            class_func = cls.calc_classifier(kernel_mode, data_i, clas_i, kernel)
            if class_func == None:
                continue

            test_result = map(lambda j : class_func(test_data_i[j]) * test_clas_i[j], range(len(test_data_i)))
            test_passed = [x for x in test_result if x > 1]

            result += float(len(test_passed))/float(len(test_result))

        return result / fold


CONTEXT_SETTINGS = { 'help_option_names': ['-h', '--help'] }

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('datafile', type=click.Path(exists=True), required=True)
@click.option('-d', '--debug', default=False, is_flag=True, help='Execute program in debug mode')
@click.option('-f', '--fold', default=10, help='Number of subsamples into which original sample is partitioned')
@click.option('-k', '--kernel', default='linear', help='Type of Kernel [linear/poly/gauss]')
@click.option('-p', '--plot', default=False, is_flag=True, help='Plot Data and Classifier')
def cli(datafile, debug, fold, kernel, plot):

    options = {
            'datafile': datafile,
            'debug': debug,
            'fold': fold,
            'kernel': kernel,
            'plot': plot
            }

    svm = SVM(options)
    svm.run()

if __name__ == '__main__':
    argv = sys.argv

    if len(argv) < 2:
        click.echo('  Usage: python svm.py DATAFILE [KERNEL]')
        click.echo('    DATAFILE -- path to datafile')
        click.echo('    KERNEL   -- type of kernel [default:linear/poly/gauss]')
        exit()
    elif len(argv) == 2:

        options = {
                'datafile': argv[1],
                'debug': False,
                'fold': 10,
                'kernel': 'linear',
                'plot': False
                }

    elif len(argv) == 3:
 
        options = {
                'datafile': argv[1],
                'debug': False,
                'fold': 10,
                'kernel': argv[2],
                'plot': False
                }
    else:

        options = {
                'datafile': argv[1],
                'debug': False,
                'fold': int(argv[3]),
                'kernel': argv[2],
                'plot': False
                }

    svm = SVM(options)
    svm.run()
