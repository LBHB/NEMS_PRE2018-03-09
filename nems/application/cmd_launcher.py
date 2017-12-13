def main():
    from nems.main import fit_single_model
    import argparse

    parser = argparse.ArgumentParser(description='Fit single model')
    parser.add_argument('batch', type=int, help='Batch to use')
    parser.add_argument('cell', type=str, help='Cell from batch to fit')
    parser.add_argument('model', type=str, help='Model to fit')
    args = parser.parse_args()
    fit_single_model(args.cell, args.batch, args.model, autoplot=True)
