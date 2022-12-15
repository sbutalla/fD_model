import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, matthews_corrcoef
import sys
import os
import json
from xgboost import plot_tree
from sklearn.model_selection import train_test_split
import graphviz
from tuning.parameter_processing import Process
from tuning.model import Model
import time
import xgboost
from sklearn import tree as sktree
import warnings

from tuning.model_per_sample import ModelPerSample

'''
    Cleaned up version of xgb_plusplus.py for hyperparameter tuning. This version of the class is used
    strictly for tuning, all edits conflict wiht the original purpose of xg_plusplus.py. For organization 
    this new file was created so that it would be able to be added to the original repository.
    
    Given the processed data, the xgb function in the xgb class creates the model with the necessary
    parameters and dumps the history and classification report. As well as, create a new object to store
    a summary of the important information from the model. This includes the parameters as well as the
    necessary metrics in determining the efficiency of the model.
'''


resultDir = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/"

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
            os.makedirs(resultDir)  # create directory for results
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

        self.testX = pd.DataFrame(
            X_test,
            columns=["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3", "selPhi0",
                     "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0",
                     "dPhi1", "dRA0", "dRA1", "event", "invMassA0", "invMassA1"],)

        self.trainY = pd.DataFrame(
            y_train, columns=["event", "invmA0", "invmA1", "pair"]
        )
        self.trainY = self.trainY.drop(["event", "invmA0", "invmA1"], axis=1)
        self.testY = pd.DataFrame(y_test, columns=["event", "invmA0", "invmA1", "pair"])
        self.testY = self.testY.drop(["event", "invmA0", "invmA1"], axis=1)

        if ret:
            return self.trainX, self.testX, self.trainY, self.testY
        else:
            return None

    def xgb(
            self,
            zd_mass,
            fd1_mass,
            resultDir,
            trainX=None,
            testX=None,
            trainY=None,
            testY=None,
            met=True,
            save=True,
            filename="MZD_200_55_pd_model.sav",
            single_pair=False,
            ret=True,
            verbose=True,
            saveFig=True,
            eta=None,
            max_depth=None,
            booster='gbtree',
            reg_alpha=None,
            reg_lambda=None,
            tree=False,
            objective=None,
            ret_mod=True
    ):

        # print("\n\n")
        # print(60 * "*")
        # print(colors.GREEN + "Building the XGBoost model and training" + colors.ENDC)
        # print(60 * "*")
        warnings.filterwarnings("ignore")
        proc = Process()
        global dataDir
        if self.dataset == "mc":
            mc_model = filename.split(".")[0]
            dataDir = proc.select_file(eta, max_depth, resultDir, mc_model, reg_lambda, reg_alpha, objective)
            try:
                os.makedirs(dataDir)  # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass
        elif self.dataset == "bkg":
            dataDir = resultDir + "/" + self.file_name
            try:
                os.makedirs(dataDir)  # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass
        elif self.dataset == "sig":
            dataDir = resultDir + "/signal_MZD_"
            try:
                os.makedirs(dataDir)  # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass

        model = proc.select_model(eta, max_depth, reg_lambda, reg_alpha, objective)

        start = time.time()
        eval_set = [(trainX, trainY), (testX, testY)]
        # model.fit(trainX, trainY, eval_metric = [])
        model.fit(trainX, trainY, early_stopping_rounds=10,
                  eval_metric=["logloss", "error", "auc"], eval_set=eval_set,
                  verbose=False)
        end = time.time()

        if tree:
            filename = resultDir + 'MZD_200_55_pd_model/effective_model_tree'
            #
            # dot_data = sktree.export_graphviz(model)
            # graph = graphviz.Source(dot_data, format="png")
            # graph.render(filename)

            # plot_tree(model)
            # fig = plt.gcf()
            # fig.set_size_inches(30, 15)
            # plt.show()
            # fig.show()

        total_time = end - start

        # model.fit(self.trainX, self.trainY, early_stopping_rounds=num_of_epochs)

        if save:
            # save the model to disk
            joblib.dump(model, filename)

        if met:
            # predictedY = model.predict(self.testX)
            predictedY = model.predict(testX)

            mod_probs = model.predict_proba(testX)  # predict probabilities
            mod_probs = mod_probs[:, 1]  # probabilities for pos. outcome only
            mod_auc = roc_auc_score(testY, mod_probs)  # model (logistic) AUC

            # Testing, original
            class_out = dataDir + "/classification_report.json"
            out_file = open(class_out, "w")
            #class_report = dict(classification_report(self.testY, predictedY, output_dict=True))
            class_report = dict(classification_report(testY, predictedY, output_dict=True))
            class_report['parameters'] = {'eta': eta, 'max_depth': max_depth, 'booster': booster,
                                          'l1': reg_alpha, 'l2': reg_lambda, 'objective': objective}
            # class_report['mcc'] = matthews_corrcoef(self.testY, predictedY)
            class_report['mcc'] = matthews_corrcoef(testY, predictedY)
            json.dump(class_report, out_file)



            '''
                Filling the object that stores all of the data. Store the objects in a list to be sorted
                and outputted. Add to the model object for other parameters that will be tested.

                Booster?
                Check notebook for others.
            '''
            mod = ModelPerSample()
            mod.zD = zd_mass
            mod.fD1 = fd1_mass
            mod.eta = class_report['parameters']['eta']
            mod.max_depth = class_report['parameters']['max_depth']
            mod.booster = class_report['parameters']['booster']
            mod.accuracy = class_report['accuracy']
            mod.mcc = class_report['mcc']
            mod.time = total_time
            mod.f1 = class_report['1']['f1-score']
            mod.precision = class_report['1']['precision']
            mod.reg_alpha = class_report['parameters']['l1']
            mod.reg_lambda = class_report['parameters']['l2']
            mod.objective = class_report['parameters']['objective']
            mod.auc = mod_auc

            mod_out = dataDir + "/model.json"
            out_file = open(mod_out, "w")
            json.dump(mod.get_model(), out_file)




            # mod = Model()
            # mod.set_eta(class_report['parameters']['eta'])
            # mod.set_max_depth(class_report['parameters']['max_depth'])
            # mod.set_booster(class_report['parameters']['booster'])
            # mod.set_accuracy(class_report['accuracy'])
            # mod.set_mcc(class_report['mcc'])
            # mod.set_time(total_time)
            # mod.set_f1(class_report['1']['f1-score'])
            # mod.set_precision(class_report['1']['precision'])
            # mod.set_reg_alpha(class_report['parameters']['l1'])
            # mod.set_reg_lambda(class_report['parameters']['l2'])
            # mod.set_objective(class_report['parameters']['objective'])
            # xgb.model_list.append(mod)

            if ret_mod:
                return mod
            else:
                del mod
                return None

        return None
