from nems.Signal import load_signals_in_dir, split_signals_by_time


if __name__ == '__main__':
    signals = load_signals_in_dir('signals/gus027b13_p_PPS')

    # Use the first 80% as estimation data, and the last 20% as validation
    est, val = split_signals_by_time(signals, 0.8)

    model = Model()
    model.append(WeightChannelsGaussian(output_channels=2))
    model.append(FIR(n_taps=20))
    model.append(Nonlinearity(shape='double exponential'))

    # Fitter
    eval_fn = partial(model.evaluate, est)
    cost_fn = lambda i, o: MSE(i['resp'], o['pred'])
    fitter = LinearFitter(cost_fn, eval_fn)

    # The time consuming part
    #phi_distributions = fitter.fit(model)

    # Plot the prediction vs reality
    # phi_distributions.plot('/some/other/path.png')

    # TODO: Plot confidence intervals
    # phi_EV = phi_distributions.expected_value()
    # phi_10 = phi_distributions.percentile(10)
    # phi_90 = phi_distributions.percentile(90)
    # pred_EV = model.evaluate(phi_EV, val)
    # pred_10 = model.evaluate(phi_10, val)
    # pred_90 = model.evaluate(phi_90, val)
    # plot_signals('/some/path.png', pred_EV, pred_10, pred_90, ...)

    # TODO: At a later date, cross-validate
    # validator = CrossValidator(fitter, 20)
    # validator.fit(fitter, model, signals)

    # TODO: Measure various other performance metrics and save them
    # performance = {'mse': MSE(val, pred_EV),
    #                'logl': LogLikelihood(val, pred_EV)}
    # model.save('/yet/another/path.json', phi_distributions, performance)
