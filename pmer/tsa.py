import matplotlib
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


def plot_ts(xs, ts, title=None, ax=None):
    if ax is None:
        ax = plt.axes()
    ax.plot(xs, ts)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    if title:
        ax.set_title(title)


def plot_kde(ts, title=None, ax=None):
    """Display kernel density estimation."""
    if ax is None:
        ax = plt.axes()
    ts = np.asarray(ts)
    xs = np.linspace(ts.min(), ts.max(), 1000)
    ys = scipy.stats.gaussian_kde(ts).pdf(xs)
    ax.plot(xs, ys)
    if title:
        ax.set_title(title)


def plot_acf(ts, nlags=10, title=None, ax=None):
    if ax is None:
        ax = plt.axes()
    _plot_cf(ax, ts, acf, nlags)
    if title:
        ax.set_title(title)


def plot_pacf(ts, nlags=10, title=None, ax=None):
    if ax is None:
        ax = plt.axes()
    _plot_cf(ax, ts, pacf, nlags)
    if title:
        ax.set_title(title)


def _plot_cf(ax, ts, cf_func, nlags):
    """Helper function to display acf/pacf."""
    ts_cf = cf_func(ts, nlags=nlags)
    xs = np.arange(0, nlags+1, 1)
    _, stemlines, baseline = ax.stem(xs, ts_cf)
    plt.setp(stemlines, 'color', 'b', 'linestyle', '--', 'linewidth', 1)
    plt.setp(baseline, 'visible', False)
    ax.set_xticks(np.arange(0, xs[-1] + 1))
    ax.set_ylim(0, 1.1)
