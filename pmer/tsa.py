import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import statsmodels.tsa.stattools


def acf(ts, nlags=40):
    return statsmodels.tsa.stattools.acf(ts, nlags=nlags)


def pacf(ts, nlags=40):
    return statsmodels.tsa.stattools.pacf(ts, nlags=nlags)


def diff(ts):
    """Compute first order difference of time series."""
    ts = np.asarray(ts)
    return ts[1:] - ts[:-1]


def plot_ts(ts):
    plt.figure()
    plt.plot(ts)
    plt.title('RAW')


def plot_kde(ts):
    """Display kernel density estimation."""
    ts = np.asarray(ts)
    xs = np.linspace(ts.min(), ts.max(), 1000)
    ys = scipy.stats.gaussian_kde(ts).pdf(xs)
    plt.figure()
    plt.plot(xs, ys)
    plt.title('Kernel Density Estimation')


def plot_acf(ts, nlags=10):
    _plot_cf(ts, acf, nlags)


def plot_pacf(ts, nlags=10):
    _plot_cf(ts, pacf, nlags)


def _plot_cf(ts, cf_func, nlags):
    """Helper function to display acf/pacf."""
    ts_cf = cf_func(ts, nlags=nlags)
    xs = np.arange(0, nlags+1, 1)
    plt.figure()
    _, stemlines, baseline = plt.stem(xs, ts_cf)
    plt.setp(stemlines, 'color', 'b', 'linestyle', '--', 'linewidth', 1)
    plt.setp(baseline, 'visible', False)
    plt.xticks(np.arange(0, xs[-1] + 1))
    plt.ylim(0, 1.1)
    plt.title(cf_func.__name__.upper())
