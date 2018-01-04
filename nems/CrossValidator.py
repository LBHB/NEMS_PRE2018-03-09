###############################################################################
# Cross validator
###############################################################################
class CrossValidator:

    def __init__(self, folds):
        self.folds = folds

    def fit(self, fitter, model, signals):
        for train, test in self.split_train_test(signals):
            train_result = fitter.fit(model, train)
            test_result = model.evaluate(train_result, test)
