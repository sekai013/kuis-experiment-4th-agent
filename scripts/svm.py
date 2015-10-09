# Modules
import sys
import click
import numpy
import cvxopt
from math import exp
import matplotlib.pyplot as pyplot

# Main class
class SVM(object):

    DELTA = 1.0e-6

    def __init__(self, options):
        self.kernel_mode = options['kernel']
        self.datafile = options['datafile']
        self.do_plot = options['plot']
        self.is_debug_mode = options['debug']

        cvxopt.solvers.options['show_progress'] = self.is_debug_mode

        self.kernel = SVM.get_kernel(self.kernel_mode)
        self.calc_threshold = SVM.get_threshold(self.kernel_mode)
        self.plot = SVM.get_plot(self.kernel_mode)

    def run(self):
        try:
            # Minimize  : (1/2)(x^T)(Q)(x) + (q^t)(x)
            # Subject to: (A)(x) >= b, (A_0)(x) = b_0

            data = numpy.loadtxt(self.datafile)
            clas = map(lambda d : d[-1], data)
            data = map(lambda d : d[:-1], data)

            datasize = len(data)

            # Setup matrix and vector

            Q = numpy.zeros((datasize, datasize))
  
            for i in range(datasize):
                for j in range(datasize):
                    Q[i][j] = clas[i] * clas[j] * self.kernel(data[i], data[j])
  
            q = - numpy.ones(datasize)
  
            A = numpy.diag([-1.0] * datasize)
            b = numpy.zeros(datasize)
  
            A_0 = clas
            b_0 = 0.0

            # convert matrix and vector to cvxopt.matrix object
  
            Q = cvxopt.matrix(Q)
            q = cvxopt.matrix(q)
            A = cvxopt.matrix(A)
            b = cvxopt.matrix(b)
            A_0 = cvxopt.matrix(A_0, (1, datasize))
            b_0 = cvxopt.matrix(b_0)

            # print parameters if debug mode

            if self.is_debug_mode:
                print 'Q:'
                print Q
                print 'q:'
                print q
                print 'A:'
                print A
                print 'b:'
                print b
                print 'A_0:'
                print A_0
                print 'b_0:'
                print b_0

            # solve cone quadratic programming and get lagrangians

            solution = cvxopt.solvers.qp(Q, q, A, b, A_0, b_0)
            # TODO: check solution['status']
            lagrangians = solution['x']

            # print Lagrangians if debug mode

            if self.is_debug_mode:
                print 'Lagrangians:'
                print lagrangians

            # calculate weight vector and threshold

            if self.kernel_mode == 'linear':
                weight = [0] * len(data[0])

                for i in range(len(lagrangians)):
                    if lagrangians[i] > self.DELTA:
                        weight = map(lambda j : weight[j] + lagrangians[i] * clas[i] * data[i][j], range(len(data[0])))

                threshold = self.calc_threshold(data, clas, lagrangians, self.kernel, weight)

                click.echo('Weight: %s' % weight)
                click.echo('Threshold: %f' % threshold)

                if self.do_plot and len(data[0]) == 2:
                    self.plot(data, clas, weight, threshold)

            else:
                threshold = self.calc_threshold(data, clas, lagrangians, self.kernel)

                click.echo('p:')
                click.echo(map(lambda i : lagrangians[i] * clas[i] if lagrangians[i] > self.DELTA else 0, range(len(lagrangians))))
                click.echo('Threshold t: %f' % threshold)
                click.echo('Classifier: f(x, p, t) = sign( sum(p[i] * Kernel(x, x_i)) - t)')

                if self.do_plot and len(data[0]) == 2:
                    self.plot(data, clas, lagrangians, threshold, self.kernel)

        except IOError:
            click.echo('Failed to open file %s' % self.datafile)
            exit()

    @classmethod
    def get_kernel(cls, kernel_mode='linear'):
        
        def linear(x, y):
            if len(x) != len(y):
                raise BaseException, 'size of x is %d but size of y is %d' % (len(x), len(y))

            inner_product = 0

            for i in range(len(x)):
                inner_product += x[i] * y[i]

            return inner_product

        def poly(x, y):
            return (1 + linear(x, y)) ** 2#len(x)

        def gauss(x, y):
            if len(x) != len(y):
                raise BaseException, 'size of x is %d but size of y is %d' % (len(x), len(y))

            sigma = 10
            norm = sum(map(lambda i : (x[i] - y[i]) ** 2, range(len(x))))
            return exp(- norm / (2 * (sigma ** 2)))

        KERNELS = {
                'linear': linear,
                'poly': poly,
                'gauss': gauss
                }

        kernel_mode = kernel_mode.lower()

        try:
            if kernel_mode in KERNELS:
                return KERNELS[kernel_mode]
            else:
                raise BaseException, 'Invalid Kernel:%s hit `svm --help` to check usage.' % kernel_mode
        except BaseException, e:
            click.echo(e)
            exit()

    @classmethod
    def get_threshold(cls, kernel_mode='linear'):

        def linear(data, clas, lagrangians, kernel, w=[]):
            count = 0
            theta = 0

            for i in range(len(lagrangians)):

                if lagrangians[i] > cls.DELTA:
                    count += 1
                    theta += kernel(w, data[i]) - clas[i]

            return theta / count

        def nonlinear(data, clas, lagrangians, kernel):
            count = 0
            theta = 0

            for i in range(len(lagrangians)):

                if lagrangians[i] > cls.DELTA:
                    count += 1
                    inner_product = 0

                    for j in range(len(lagrangians)):
                        inner_product += lagrangians[j] * clas[j] * kernel(data[i], data[j])

                    theta += inner_product - clas[i]

            return theta / count

        if kernel_mode == 'linear':
            return linear
        else:
            return nonlinear

    @classmethod
    def get_classifier(cls, kernel_mode='linear'):

        def linear(x, w, theta):
            return - (w[0] * x - theta) / w[1]

        def nonlinear(x, data, clas, lagrangians, theta, kernel):
            c = 0

            for i in range(len(lagrangians)):
                c += lagrangians[i] * clas[i] * kernel(x, data[i])

            return c - theta

        if kernel_mode == 'linear':
            return linear
        else:
            return nonlinear

    @classmethod
    def get_plot(cls, kernel_mode='linear'):
        
        def linear(data, clas, weight, threshold):
            classifier = cls.get_classifier(kernel_mode)

            # plot data
            for i in range(len(data)):
                if clas[i] == 1:
                    pyplot.plot(data[i][0], data[i][1], 'rx')
                else:
                    pyplot.plot(data[i][0], data[i][1], 'bx')

            # plot classifier
            min_cell = min(map(lambda d : min(d), data))
            max_cell = max(map(lambda d : max(d), data))
            padding = (min_cell + max_cell) / 15.0
            classifier_x = numpy.linspace(min_cell - padding, max_cell + padding, 1000)
            classifier_y = map(lambda x_i : classifier(x_i, weight, threshold), classifier_x)

            pyplot.plot(classifier_x, classifier_y, 'g-')
            pyplot.show()

        def nonlinear(data, clas, lagrangians, threshold, kernel):
            classifier = cls.get_classifier(kernel_mode)

            # plot data
            for i in range(len(data)):
                if clas[i] == 1:
                    pyplot.plot(data[i][0], data[i][1], 'rx')
                else:
                    pyplot.plot(data[i][0], data[i][1], 'bx')

            # plot classifier
            min_cell = min(map(lambda d : min(d), data))
            max_cell = max(map(lambda d : max(d), data))
            padding = (min_cell + max_cell) / 15.0
            X1, X2 = numpy.meshgrid(numpy.linspace(min_cell - padding, max_cell + padding, 50),
                                    numpy.linspace(min_cell - padding, max_cell + padding, 50))
            width, height = X1.shape
            X1.resize(X1.size)
            X2.resize(X2.size)
            Z = numpy.array([classifier(numpy.array([x1, x2]), data, clas, lagrangians, threshold, kernel) for (x1, x2) in zip(X1, X2)])
            X1.resize((width, height))
            X2.resize((width, height))
            Z.resize((width, height))

            pyplot.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
            pyplot.show()

        if kernel_mode == 'linear':
            return linear
        else:
            return nonlinear

CONTEXT_SETTINGS = { 'help_option_names': ['-h', '--help'] }

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-k', '--kernel', default='linear', help='Type of Kernel [linear/poly/gauss]')
@click.option('-p', '--plot', default=False, is_flag=True, help='Plot Data and Classifier')
@click.option('-d', '--debug', default=False, is_flag=True, help='Execute program in debug mode')
@click.argument('datafile', required=True)
def cli(kernel, datafile, plot, debug):

    options = {
            'kernel': kernel,
            'datafile': datafile,
            'plot': plot,
            'debug': debug
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
                'kernel': 'linear',
                'datafile': argv[1],
                'plot': False,
                'debug': False
                }

        svm = SVM(options)
        svm.run()
    else:

        options = {
                'kernel': argv[2],
                'datafile': argv[1],
                'plot': False,
                'debug': False
                }

        svm = SVM(options)
        svm.run()
