import numpy as np
import matplotlib.pyplot as plt
from .timeseries import timeseries_from_signals, timeseries_from_epoch
from .scatter import plot_scatter

def pred_vs_act_scatter(recording, modelspec, evaluator, transform_idx=-1,
                        pred_name='pred', act_name='resp'):
    transformed = evaluator(recording, modelspec, stop=transform_idx)
    # TODO: previous version only looks at first channel
    #       and one stim occurrence at a time?
    #       Copying for now but seems like we might want to change that.

    # get prediction and actual as chans x time, but only look at one channel
    predicted = transformed[pred_name]
    actual = recording[act_name]

    plot_scatter(actual, predicted)
    # TODO: Add text box with r_values (see master branch version)

def pred_vs_act_psth(recording, modelspec, evaluator, transform_idx=-1,
                     occurrence=0, channel=0, pred_name='pred',
                     act_name='resp'):
    transformed = evaluator(recording, modelspec, stop=transform_idx)
    predicted = transformed[pred_name]
    actual = recording[act_name]

    timeseries_from_signals([actual, predicted], channel=channel,
                            ylabel='Firing Rate')

# TODO: Looks like this is the one used m ost often by current NEMS.
#       Maybe just need a separate plot_timeseries_smoothed to run this through
def pred_vs_act_psth_smooth(recording, modelspec, evaluator, transform_idx=-1,
                            pred_name='pred', act_name='resp'):
    raise NotImplementedError

    transformed = evaluator(recording, modelspec, stop=transform_idx)
    predicted = transformed[pred_name]
    actual = recording[act_name]
    pred_vector = predicted.as_continuous()[0]
    act_vector = actual.as_continuous()[0]

    box_pts = 20
    box = np.ones(box_pts) / box_pts
    pred_convolved = np.convolve(pred_vector, box, mode='same')
    act_convolved = np.convolve(act_vector, box, mode='same')

    pred, = plt.plot(pred_convolved, label='Predicted')
    act, = plt.plot(act_convolved, 'r', label='Actual')
    plt.legend(handles=[pred, act])
    plt.xlabel('Time Step')
    plt.ylabel('Firing Rate')
