
# Plotting for double exponential
    def plot_coefficients(self, phi, data=None, axes=None):
        if data is not None:
            pred = data[self.input_name]
            x = np.linspace(pred.min(), pred.max(), 100)
        else:
            x = np.linspace(0, 1000, 100)

        if axes is None:
            ax = pl.gca()

        y = double_exponential(x, **phi)
        ax.plot(x, y, 'k-')
