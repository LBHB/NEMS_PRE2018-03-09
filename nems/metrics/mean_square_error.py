def mse_cost_function(phi, model, signals):
    result = model.evaluate(phi, signals)
    delta = result['pred'] - result['resp']
    return np.mean(delta**2)
