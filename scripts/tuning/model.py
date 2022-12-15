"""
    Object that stores identifying information for each of the generated models. Information is store in an
    object that is then stored in a dictionary to be put in a list for output using json dump. This is how
    we keep track of the efficiency of the models while containing them in one single place. Essential for effective
    computation of the most accurate results for determining which hyper-parameters perform the best given a
    specified scope.
"""


class Model:
    def __init__(self):
        self.zD = None
        self.fD1 = None
        self.accuracy = None
        self.max_depth = None
        self.eta = None
        self.mcc = None
        self.booster = None
        self.time = None
        self.f1 = None
        self.precision = None
        self.reg_lambda = None
        self.reg_alpha = None
        self.objective = None
        self.auc = None

    def get_model(self):
        object_dict = {
            "zd mass": self.zD,
            "fd1 mass": self.fD1,
            "accuracy": self.accuracy,
            "max depth": self.max_depth,
            "eta": self.eta,
            "mcc": self.mcc,
            "booster": self.booster,
            "time": self.time,
            "f1": self.f1,
            "precision": self.precision,
            "l1": self.reg_alpha,
            "l2": self.reg_lambda,
            "objective": self.objective,
            "auc": self.auc,
        }
        return object_dict
