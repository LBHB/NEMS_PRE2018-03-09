Signals are always stored 2D on disk.

Signal objects contain 2D data: channel x time_sample, like a CSV file with each row being a moment in time.

Trial-by-trial groupings shouldn't matter for the evaluation of a module (I hope?), and not supporting 4D arrays gets rid of nasty problems that occur at the boundaries of a convolved signal when it is folded by trial. 

Basically, then, if you want to have a model that uses the data from 100 neurons, you can either have a 100 channel Signal, or 100 one-channel signals. It's up to you.