import LPR.lpr
from LPR.lpr import LPR as lpRules

if __name__ == '__main__':
    lpr = lpRules('example/kinship.data', train_size=0.6, test_size=0.2)
    model = lpr.fit(lpr.train, lpr.validate)
    results = LPR.lpr.predict(lpr.test, model)
