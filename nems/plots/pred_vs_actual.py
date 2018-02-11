import numpy as np
import matplotlib.pyplot as plt

def pred_vs_act_scatter(recording, modelspec, evaluator, transform_idx=-1,
                        pred_name='pred', act_name='resp', ax=None):

    if ax:
        plt.sca(ax)

    transformed = evaluator(recording, modelspec, stop=transform_idx)
    # TODO: previous version only looks at first channel
    #       and one stim occurrence at a time?
    #       Copying for now but seems like we might want to change that.

    # get prediction and actual as chans x time, but only look at one channel
    predicted = transformed[pred_name]
    actual = recording[act_name]
    pred_vector = predicted.as_continuous()[0]
    act_vector = actual.as_continuous()[0]
    # scatter pred vs act with black circle markers
    plt.plot(pred_vector, act_vector, 'ko')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # TODO: Add text box with r_values (see master branch version)

def pred_vs_act_psth(recording, modelspec, evaluator, transform_idx=-1,
                     pred_name='pred', act_name='resp', ax=None):

    # TODO: is this what our version does? doesn't look like it.
    #       -jacob
    # General approach per interwebs:
    # adjust bounds of stim epoch to a little bit before & after onset
    #   - how much before/after? base it on number of reps?
    # extract stim epoch to get 3darray of occurrence x chans x time
    #   - divide each occurrence into N bins of size delta
    # count number of spikes in each bin across all occurrences
    # x axis = bins, y = (num spikes / num occurrences * delta)
    # delta size should minimize (2*mean - variance)/delta squared

    if ax:
        plt.sca(ax)

    transformed = evaluator(recording, modelspec, stop=transform_idx)
    predicted = transformed[pred_name]
    actual = recording[act_name]
    pred_vector = predicted.as_continuous()[0]
    act_vector = actual.as_continuous()[0]
    fs = predicted.fs
    time_bins = np.arange(0, len(act_vector)) / fs

    pred, = plt.plot(time_bins, pred_vector, label='Predicted')
    act, = plt.plot(time_bins, act_vector, 'r', label='Actual')
    plt.legend(handles=[pred, act])
    plt.xlabel('Time')
    plt.ylabel('Firing Rate')

def pred_vs_act_psth_smooth(recording, modelspec, evaluator, transform_idx=-1,
                            pred_name='pred', act_name='resp', ax=None):

    # TODO: is this what our version does? doesn't look like it.
    #       -jacob
    # General approach per interwebs:
    # adjust bounds of stim epoch to a little bit before & after onset
    #   - how much before/after? base it on number of reps?
    # extract stim epoch to get 3darray of occurrence x chans x time
    #   - divide each occurrence into N bins of size delta
    # count number of spikes in each bin across all occurrences
    # x axis = bins, y = (num spikes / num occurrences * delta)
    # delta size should minimize (2*mean - variance)/delta squared
    if ax:
        plt.sca(ax)

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