import os
import logging
logging.basicConfig(level=logging.INFO)

import matplotlib.image as mpimg

import nems.modelspec as ms

def save_figure(fig, filepath=None, modelspecs=None, save_dir='/tmp',
                format='png'):
    """Saves the given matplotlib figure object, using either a specific
    filepath or a path determined from a list of modelspecs and a
    directory.

    Argumets
    --------
    fig : matplotlib figure
        Any matplotlib figure object

    filepath : str or fileobj
        Specifies the location where the figure should be saved.
        If not provided, modelspecs must be given instead. If a string,
        it should be a complete, absolute path to the save location.

    modelspecs : list
        A list of modelspecs that can be used to determine a filepath
        based on nems.modelspec.get_modelspec_longname() and save_dir.

    save_dir : str
        Specifies the base directory in which to save the figure,
        if modelspecs are used.
    format : str
        Specifies the file format that should be used to save the figure.
        Compatible choices depend on which matplotlib backend is running,
        but 'png', 'pdf', and 'svg' are almost always supported.

    Returns
    -------
    fname : str
        The filepath that was ultimately used for storage.

    """
    if filepath:
        fname = filepath
    else:
        if not modelspecs:
            raise ValueError("save_figure() must be provided either"
                             "a filepath or a list of modelspecs.")
        fname = _get_figure_name(save_dir, modelspecs, format)
    logging.info("Saving figure as: {}".format(fname))
    fig.savefig(fname)
    return fname

def load_figure_img(filepath=None, modelspecs=None, load_dir=None,
                    format='png'):
    """Loads a saved figure image as a numpy array that can be displayed
    inside python using matplotlib.pyplot.imshow().

    Argumets
    --------
    filepath : str or fileobj
        Specifies the location where the image was stored.
        If not provided, modelspecs must be given instead. If a string,
        it should be a complete, absolute path to the save location.

    modelspecs : list
        A list of modelspecs that can be used to determine a filepath
        based on nems.modelspec.get_modelspec_longname() and load_dir.

    load_dir : str
        Specifies the base directory in which to save the figure,
        if modelspecs are used.

    format : str
        Specifies the file format that was used to save the figure.

    Returns
    -------
    img : numpy ndarray
        Array containing the image data.

    """
    if filepath:
        fname = filepath
    else:
        if not modelspecs:
            raise ValueError("load_figure_img() must be provided either"
                             "a filepath or a list of modelspecs.")
        fname = _get_figure_name(load_dir, modelspecs, format)
    logging.info("Loading figure image from: {}".format(fname))
    img = mpimg.imread(fname)
    return img

def load_figure_bytes(filepath=None, modelspecs=None, load_dir=None,
                      format='png'):
    """Loads a saved figure image as a bytes object that can be used
    by the web UI or other functions.

    Argumets
    --------
    filepath : str or fileobj
        Specifies the location where the image was stored.
        If not provided, modelspecs must be given instead. If a string,
        it should be a complete, absolute path to the save location.

    modelspecs : list
        A list of modelspecs that can be used to determine a filepath
        based on nems.modelspec.get_modelspec_longname() and load_dir.

    load_dir : str
        Specifies the base directory in which to save the figure,
        if modelspecs are used.

    format : str
        Specifies the file format that was used to save the figure.

    Returns
    -------
    img : bytes object
        Contains the raw data for the loaded image.

    """
    if filepath:
        fname = filepath
    else:
        if not modelspecs:
            raise ValueError("load_figure_img() must be provided either"
                             "a filepath or a list of modelspecs.")
        fname = _get_figure_name(load_dir, modelspecs, format)
    logging.info("Loading figure image from: {}".format(fname))
    with open(fname, 'rb') as f:
        img = f.read()
    return img

def _get_figure_name(directory, modelspecs, format):
    """Determins a filepath based on a directory, a list of modelspecs,
    and a file format."""
    # TODO: Probably need a smarter way to do figure names since figures
    #       can come from an arbitrary number of modelspecs. Could just
    #       concatenate names for each modelspec? But that would mean some
    #       really long filenames.
    #       For now just uses the long name of the first modelspec until
    #       a better solution is decided on.
    mspec = modelspecs[0]
    mname = ms.get_modelspec_name(mspec)
    fname = os.path.join(directory, mname) + "." + format
    return fname