# Epochs #

Each signal should have its own 'epochs' info because you may have to annotate that one recording channel failed, or a neuron died, or something glitched. I might want to split a recording based on the  reps info of "stim", or I might want to do it based on "blinks" of pupil.

Epochs files have only three columns: `start_index`, `end_index`, and `epoch_name`. Epoch names need not be unique, as they can also be indexed by the number of times repetition. 

```
0, 1000, sound1
239, 244, blinking
488, 490, blinking
1000, 2000, sound2
2000, 3000, sound1
```


selected1 = select_fn(rec['pupil'], 'blinking')
selected2 = select_fn(selected1['pupil'], 'blinking')

And select either all the times sound1 was played, or all the times after 'blinking' occurred.

-------------------------------------------------------------------------------

