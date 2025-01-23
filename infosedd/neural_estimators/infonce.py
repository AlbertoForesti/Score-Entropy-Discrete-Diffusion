from bmi.estimators.neural import InfoNCEEstimator

class InfoNCE(InfoNCEEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, x, y):
        return self.estimate(x, y)