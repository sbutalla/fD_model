import joblib
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import sys
import time
from tuning.model import Model
import warnings
from xgboost import XGBClassifier


"""
    Cleaned up version of xgb_plusplus.py for hyperparameter tuning. This version of the class is used
    strictly for tuning, all edits conflict wiht the original purpose of xg_plusplus.py. For organization 
    this new file was created so that it would be able to be added to the original repository.
    
    Given the processed data, the xgb function in the xgb class creates the model with the necessary
    parameters and dumps the history and classification report. As well as, create a new object to store
    a summary of the important information from the model. This includes the parameters as well as the
    necessary metrics in determining the efficiency of the model.
"""


result_dir = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/"


class colors:
    WHITE = "\033[97m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    ORANGE = "\033[38;5;208m"
    ENDC = "\033[39m"


plt.rcParams.update({"font.size": 26})  # Increase font size


class xgb:
    model_list = []

    def __init__(self, dataset, file_name=None):
        warnings.filterwarnings("ignore")
        if dataset not in ["mc", "bkg", "sig"]:
            print("Dataset type must be either 'mc' or 'bkg'.")
        else:
            self.dataset = dataset

        if self.dataset == "mc":
            self.file_name = "mc"
        elif self.dataset in ["bkg", "sig"]:
            if file_name is None:
                print(
                    colors.RED
                    + "The dataset name (e.g., file_name = qqToZZTo4L, file_name = DataAboveUpsilonCRSR_MZD_200_55_signal) must be provided!"
                )
                sys.exit()
            else:
                self.file_name = file_name

        try:
            os.makedirs(result_dir)  # create directory for results
        except FileExistsError:  # skip if directory already exists
            pass

    def split(self, dataframe_shaped, test=0.25, random=7, scalerType=None, ret=False):
        warnings.filterwarnings("ignore")
        # print("\n\n")
        # print(60 * "*")
        # print(colors.GREEN + "Splitting data into train/test datasets" + colors.ENDC)
        # print(60 * "*")

        X_data = dataframe_shaped[:, 0:23]
        Y_data = dataframe_shaped[:, 20:24]
        # X_data = dataframe_shaped[:]
        # Y_data = dataframe_shaped[:]

        if self.dataset == "mc":
            if scalerType is None:
                pass
            elif scalerType == "StandardScaler":
                scaler = StandardScaler().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "MaxAbsScaler":
                scaler = MaxAbsScaler().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "MaxAbsScaler":
                scaler = MinMaxScaler().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "Normalizer":
                scaler = Normalizer().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "PowerTransformer":
                scaler = PowerTransformer().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "QuantileTransformer":
                scaler = QuantileTransformer().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "RobustScaler":
                scaler = RobustScaler().fit(X_data)
                X_data = scaler.transform(X_data)

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, Y_data, test_size=test, random_state=random
        )  # split into train/test datasets

        self.trainX = pd.DataFrame(
            X_train,
            columns=[
                "selpT0",
                "selpT1",
                "selpT2",
                "selpT3",
                "selEta0",
                "selEta1",
                "selEta2",
                "selEta3",
                "selPhi0",
                "selPhi1",
                "selPhi2",
                "selPhi3",
                "selCharge0",
                "selCharge1",
                "selCharge2",
                "selCharge3",
                "dPhi0",
                "dPhi1",
                "dRA0",
                "dRA1",
                "event",
                "invMassA0",
                "invMassA1",
            ],
        )
        # self.trainX = self.trainX.drop(['event'], axis = 1)

        self.test_x = pd.DataFrame(
            X_test,
            columns=[
                "selpT0",
                "selpT1",
                "selpT2",
                "selpT3",
                "selEta0",
                "selEta1",
                "selEta2",
                "selEta3",
                "selPhi0",
                "selPhi1",
                "selPhi2",
                "selPhi3",
                "selCharge0",
                "selCharge1",
                "selCharge2",
                "selCharge3",
                "dPhi0",
                "dPhi1",
                "dRA0",
                "dRA1",
                "event",
                "invMassA0",
                "invMassA1",
            ],
        )

        self.train_y = pd.DataFrame(
            y_train, columns=["event", "invmA0", "invmA1", "pair"]
        )
        self.train_y = self.train_y.drop(["event", "invmA0", "invmA1"], axis=1)
        self.test_y = pd.DataFrame(
            y_test, columns=["event", "invmA0", "invmA1", "pair"]
        )
        self.test_y = self.test_y.drop(["event", "invmA0", "invmA1"], axis=1)

        if ret:
            return self.trainX, self.test_x, self.train_y, self.test_y
        else:
            return None

    def xgb(
        self,
        zd_mass,
        fd1_mass,
        result_dir,
        train_x,
        test_x,
        train_y,
        test_y,
        eta,
        max_depth,
        reg_lambda,
        reg_alpha,
        objective,
        met=True,
        save=True,
        single_pair=False,
        ret=True,
        verbose=True,
        saveFig=True,
        booster="gbtree",
        tree=False,
        ret_mod=True,
    ):
        if train_x.empty and train_y.empty and test_x.empty and test_y.empty:
            train_x = self.trainX
            train_y = self.train_y
            test_x = self.test_x
            test_y = self.test_y
            empty = True
        else:
            empty = False

        warnings.filterwarnings("ignore")
        global data_directory
        if self.dataset == "mc":
            data_directory = select_file(
                eta, max_depth, result_dir, reg_lambda, reg_alpha, objective
            )
            try:
                os.makedirs(data_directory)  # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass
        elif self.dataset == "bkg":
            data_directory = result_dir + "/" + self.file_name
            try:
                os.makedirs(data_directory)  # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass
        elif self.dataset == "sig":
            data_directory = result_dir + "/signal_MZD_"
            try:
                os.makedirs(data_directory)  # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass

        model = select_model(eta, max_depth, reg_lambda, reg_alpha, objective)

        start = time.time()
        eval_set = [(train_x, train_y), (test_x, test_y)]
        # model.fit(trainX, trainY, eval_metric = [])
        model.fit(
            train_x,
            train_y,
            early_stopping_rounds=10,
            eval_metric=["logloss", "error", "auc"],
            eval_set=eval_set,
            verbose=False,
        )
        end = time.time()

        if tree:
            filename = result_dir + "MZD_200_55_pd_model/effective_model_tree"

            # dot_data = sktree.export_graphviz(model)
            # graph = graphviz.Source(dot_data, format="png")
            # graph.render(filename)
            #
            # plot_tree(model)
            # fig = plt.gcf()
            # fig.set_size_inches(30, 15)
            # plt.show()
            # fig.show()

        total_time = end - start

        if save:
            # save the model to disk
            joblib.dump(model, result_dir + "/effective_model_tree")

        if met:
            # predictedY = model.predict(self.test_X)
            predictedY = model.predict(test_x)

            mod_probs = model.predict_proba(test_x)  # predict probabilities
            mod_probs = mod_probs[:, 1]  # probabilities for pos. outcome only
            mod_auc = roc_auc_score(test_y, mod_probs)  # model (logistic) AUC

            # Testing, original
            class_out = data_directory + "/classification_report.json"
            out_file = open(class_out, "w")
            # class_report = dict(classification_report(self.testY, predictedY, output_dict=True))
            class_report = dict(
                classification_report(test_y, predictedY, output_dict=True)
            )
            class_report["parameters"] = {
                "eta": eta,
                "max_depth": max_depth,
                "booster": booster,
                "l1": reg_alpha,
                "l2": reg_lambda,
                "objective": objective,
            }
            # class_report['mcc'] = matthews_corrcoef(self.testY, predictedY)
            class_report["mcc"] = matthews_corrcoef(test_y, predictedY)
            json.dump(class_report, out_file)

            """
                Filling the object that stores all of the data. Store the objects in a list to be sorted
                and outputted. Add to the model object for other parameters that will be tested.

                Booster?
                Check notebook for others.
            """
            mod = Model()
            mod.zD = zd_mass
            mod.fD1 = fd1_mass
            mod.eta = class_report["parameters"]["eta"]
            mod.max_depth = class_report["parameters"]["max_depth"]
            mod.booster = class_report["parameters"]["booster"]
            mod.accuracy = class_report["accuracy"]
            mod.mcc = class_report["mcc"]
            mod.time = total_time
            if empty:
                mod.f1 = class_report["1.0"]["f1-score"]
                mod.precision = class_report["1.0"]["precision"]
            else:
                mod.f1 = class_report["1"]["f1-score"]
                mod.precision = class_report["1"]["precision"]
            mod.reg_alpha = class_report["parameters"]["l1"]
            mod.reg_lambda = class_report["parameters"]["l2"]
            mod.objective = class_report["parameters"]["objective"]
            mod.auc = mod_auc
            mod.importance = model.feature_importances_
            mod_out = data_directory + "/model.json"
            out_file = open(mod_out, "w")
            json.dump(mod.get_model(), out_file)

            if ret_mod:
                return mod
            else:
                del mod
                return None

        return None


def select_model(eta, max_depth, reg_lambda, reg_alpha, objective) -> XGBClassifier:
    warnings.filterwarnings("ignore")
    # if eta == 0.6:
    #     model = XGBClassifier(
    #         eval_metric=["logloss", "error", "auc"],
    #         random_state=7,
    #         eta=eta,
    #         max_depth=max_depth,
    #         reg_lambda=reg_lambda,
    #         reg_alpha=reg_alpha,
    #         objective=objective,
    #     )
    # else:
    #     model = XGBClassifier(random_state=7, eval_metric=["logloss", "error", "auc"])

    model = XGBClassifier(
            eval_metric=["logloss", "error", "auc"],
            random_state=7,
            eta=eta,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective=objective,
        )

    return model


def select_file(eta, max_depth, result_dir, reg_lambda, reg_alpha, objective):
    data_dir = result_dir + (
        "/eta_%s/max_depth_%s/l1_%s/l2_%s/objective_%s"
        % (eta, max_depth, reg_alpha, reg_lambda, objective)
    )
    return data_dir
