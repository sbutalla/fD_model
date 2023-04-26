import joblib
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import sys
import time
from tuning.model import Model
import warnings
from xgboost import XGBClassifier
import pickle
import numpy as np


"""
    Cleaned up version of xgb_plusplus.py for hyperparameter tuning. This version of the class is used
    strictly for tuning, all edits conflict wiht the original purpose of xg_plusplus.py. For organization 
    this new file was created so that it would be able to be added to the original repository.
    
    Given the processed data, the xgb function in the xgb class creates the model with the necessary
    parameters and dumps the history and classification report. As well as, create a new object to store
    a summary of the important information from the model. This includes the parameters as well as the
    necessary metrics in determining the efficiency of the model.
"""


# result_dir = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/"
result_dir = "/Users/spencerhirsch/Documents/research/important_models/"


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
        model_type,
        met=True,
        save=False,
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

        # feature things that need to get put elsewhere, not my problem atm
        # features = X_train_scaled.columns
        # features_importances = model.feature_importances_

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
            # mod.importance = model.feature_importances_
            mod_out = data_directory + "/model.json"
            out_file = open(mod_out, "w")
            json.dump(mod.get_model(), out_file)
            joblib.dump(model, data_directory + "/trained_model_%s.sav" % model_type)  # save the trained model
            pkl_name = (data_directory + '/trained_model_%s.pkl') % model_type
            pickle.dump(model, open(pkl_name, "wb"))


            # Confusion Matrix
            results = model.evals_result()
            types = [['TP', 'FP'], ['FN', 'TN']]
            tn, fp, fn, tp = confusion_matrix(test_y, predictedY).ravel()
            conf_mat = [[int(tp), int(fp)], [int(fn),
                                             int(tn)]]  # must convert to python built-in int and not np int64 (int64 not serializable to json)
            fig, ax = plt.subplots(figsize=(12, 12))
            im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.winter, vmin=20000, vmax=1000000)
            cbar = fig.colorbar(im, fraction=0.0458, pad=0.04, label='Number classifications')
            ax.set_title('Confusion matrix')
            ax.set_xticks([])
            ax.set_yticks([])

            ## place text labels of the TP, FP, FN, TN
            for ii in range(2):
                for jj in range(2):
                    plt.text(jj - 0.2, ii, '%s = %d' % (types[ii][jj], conf_mat[ii][jj]), fontsize=30)

            fig.tight_layout()

            fig.savefig(data_directory + "/confusion_matrix_%s.png" % model_type, transparent=True)
            # End Confusion Matrix


            '''
            Plotting the AUC
            '''

            ns_probs = np.zeros(len(test_y),
                                dtype=int)  # initialize array of probabilities for a classifier with no skill
            mod_probs = model.predict_proba(test_x)  # predict probabilities
            mod_probs = mod_probs[:, 1]  # keep probabilities for positive outcome only

            ## calculate scores for ROC and AUC
            ns_auc = roc_auc_score(test_y, ns_probs)  # no-skill AUC
            mod_auc = roc_auc_score(test_y, mod_probs)  # model (logistic) AUC

            ## summarize ROC with the area under the curve (AUC)

            print('No Skill: AUC = %.3f' % (ns_auc))
            print('Logistic: AUC = %.3f' % (mod_auc))

            ## generate roc curves
            ns_fpr, ns_tpr, _ = roc_curve(test_y, ns_probs)
            mod_fpr, mod_tpr, _ = roc_curve(test_y, mod_probs)

            ## make a pretty plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.grid(zorder=0)
            ax.plot(ns_fpr, ns_tpr, linestyle='--', color='tab:red', label='No Skill')
            ax.plot(mod_fpr, mod_tpr, marker='.', color='tab:blue', label='Logistic')
            ax.set_xlabel('False Positive Rate', loc='right')
            ax.set_ylabel('True Positive Rate', loc='top')
            ax.set_facecolor("white")
            ax.legend()
            textstr = '\n'.join((
                'No skill AUC = %.3f' % (ns_auc,),
                'Model AUC = %.3f' % (mod_auc,)))

            props = dict(boxstyle='round', facecolor='white', alpha=0.5)

            ax.text(0.5, 0.32, textstr, transform=ax.transAxes, fontsize=26, verticalalignment='top', bbox=props)
            fig.tight_layout()

            fig.savefig(data_directory + "/graph_auc_%s.png" % model_type, transparent=True)

            if ret_mod:
                return mod
            else:
                del mod
                return None

        return None


'''
    Function used to select the model that needs to be constructed. (Used to contain other code, has been removed).
    Function constructs the classifier for the model and returns the classifier to the caller.
'''

def select_model(eta, max_depth, reg_lambda, reg_alpha, objective) -> XGBClassifier:
    warnings.filterwarnings("ignore")
#    model = XGBClassifier(
#           eval_metric=["logloss", "error", "auc"],
#            random_state=7,
#            eta=eta,
#            max_depth=max_depth,
#            reg_lambda=reg_lambda,
#            reg_alpha=reg_alpha,
#            objective=objective,
#        )

#    return model

    model = XGBClassifier(
        n_jobs=-1,
        # use_label_encoder=False,
        eval_metric="logloss",
        random_state=7,
        eta=eta,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        objective=objective
        )

    return model


'''
    Used to select where to store the data. (Used to contain other code that has since been removed.)
'''


def select_file(eta, max_depth, result_dir, reg_lambda, reg_alpha, objective):
    data_dir = result_dir + (
        "/eta_%s/max_depth_%s/l1_%s/l2_%s/objective_%s"
        % (eta, max_depth, reg_alpha, reg_lambda, objective)
    )
    return data_dir
