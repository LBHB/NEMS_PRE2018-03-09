import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(x, bins=None):
    """Wrapper for matplotlib's plt.histogram for consistent formatting.

    Arguments:
    ----------
    x : array or sequence of arrays
        The data to be plotted.
    bins : integer, sequence, or 'auto'
        The number of bins for the data.
    See matplotlib.pyplot.hist for more details.

    Returns:
    --------
    None
    """
    plt.hist(x, bins=bins)

def pred_error_hist(resp, pred, ax=None, channel=0, bins=None, calc_bins=True,
                    bin_mult=5.0, trim_outliers=True, trim_within=2.0,
                    xlabel='|Resp - Pred|', ylabel='Count',
                    title='Prediction Error'):
    """Plots a histogram of the error between response and prediction.

    Arguments:
    ----------
    resp : signal
        response signal from a loaded recording object
    pred : signal
        prediction signal from a loaded recording object that
        has been used to fit a modelspec.
    ax : matplotlib ax object
        Will be used as the current plotting axes if provided.
    channel : int
        The channel for each of the signals that should be used.
        This should generally be 0 since responses normally only
        have one signal.
    bins : int, sequence, or 'auto'
        Number of bins for the data. See matplotlib.pyplot.his for details.
    calc_bins : boolean
        If true, bins will overwritten by a calculation based on
        the length of resp and pred.
    bin_mult : float
        The value returned by calc_bins is multipled by this factor,
        to allow finer adjustment of bin count.
    trim_outliers : boolean
        If true, err values that exceed mean err + some multiple
        (trim_within) of the standard deviation of the error
        are excluded from the data. This is useful for error data
        that has a long tail in either direction due to extreme outliers.
    trim_within : float
        Multiple of standard deviation that determines the range of
        'normal' versus 'outlier' values (see trim_outliers).
    xlabel, ylabel, title : str
        String identifiers that will be used to set title and axis labels
        for the plot.

    Returns:
    --------
    None

    """
    if ax:
        plt.sca(ax)
    resp_data = resp.as_continuous()[channel]
    pred_data = pred.as_continuous()[channel]
    err_data = np.absolute(resp_data - pred_data)
    # drop NaN values, plt.hist doesn't like them.
    err_data = err_data[~np.isnan(err_data)]

    if trim_outliers:
        std = np.std(err_data)
        mean = np.mean(err_data)
        max_err = mean + std*trim_within
        min_err = mean - std*trim_within
        if min_err < 0:
            min_err = 0
        err_data = err_data[err_data < max_err]
        err_data = err_data[err_data > min_err]
    if calc_bins:
        # TODO: Might need to adjust this calculation to find
        #       the optimum.
        length = err_data.shape[0]
        bins = int(bin_mult*(np.ceil(np.sqrt(length))))

    plot_histogram(err_data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)