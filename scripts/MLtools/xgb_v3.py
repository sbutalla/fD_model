'''
XGBoost machine learning tools for analyzing HEP
data.
Stephen D. Butalla & Mehdi Rahmani
2022/07/30, v. 0

'''

import joblib
import json 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sn
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sys
from xgboost import XGBClassifier
sys.path.insert(0, "/Users/spencer/Documents/GitHub/fD_model/scripts/utilities")
# sys.path.insert(0, "/Users/spencerhirsch/Documents/GitHub/fD_model/scripts/utilities")
from file_utils import *

global resultDir
global xgb_results
resultDir   = "dataframes/"
xgb_results = "xgb_results/"

key_dict = { # for plot titles/legends
    "qqToZZTo4L"  : r'$qq\rightarrow ZZ\rightarrow 4l$',
    "ggToZZTo4mu" : r'$gg\rightarrow ZZ \rightarrow 4\mu$',
    "ggHToZZTo4L" : r'$gg \rightarrow H \rightarrow ZZ \rightarrow 4l$',
    "DiLept"      : r'Dileptons',
    "DY0J"        : r'Drell-Yan with zero jets',
    "DY1J"        : r'Drell-Yan with one jet',
    "DY2J"        : r'Drell-Yan with two jets'
}

traintest_dict = {"validation_0": "Train", "validation_1": "Test"}
metric_dict    = {"logloss": "Log-loss", "error": "Error", "auc": "AUC"}

plt.rcParams.update({'font.size': 26}) # Increase font size for plotting

def scale_data(X_data, scaler_type, verbose = False):
    '''
    Function that scales feature data for input into a ML network.
    2022/11/21, v. 0
    
    Positional arguments:
    X_data:       (np.array; float/int) A numpy array of shape (N_events, 22)
                  contains the data features.
    scaler_type:  (str) The type of scaling to be applied to the data. Scaler
                  functions from the sklearn.preprocessing class.
                  Options:
                      - StandardScaler:      Scales features to Z-scores.
                      - MaxAbsScaler:        Scales features by their maximum
                                             absolute values.
                      - MinMaxScaler:        Scales features relative to the minimum
                                             and maximum values in the dataset.
                      - Normalizer:          Scales features to their unit norm.
                      - PowerTransformer:    Scales features by a power transformation
                                             resulting in a Gaussian-like distribution.
                      - QuantileTransformer: Scales features based on the quantiles of 
                                             their distribution.
                      - RobustScaler:        Scales features using the quantiles so that
                                             the scaled distribution is insensitive to
                                             outliers.

    '''
    if verbose:
        print_alert("\n\n")
        print_alert(60 * '*')
        print_alert("Splitting data into train/test datasets")
        print_alert(60 * '*')

    if scalerType is None:
        pass
    elif scaler_type == "StandardScaler":
        scaler = StandardScaler().fit(X_data)
        X_data = scaler.transform(X_data)
    elif scaler_type == "MaxAbsScaler":
        scaler = MaxAbsScaler().fit(X_data)
        X_data = scaler.transform(X_data)
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler().fit(X_data)
        X_data = scaler.transform(X_data)
    elif scaler_type == "Normalizer":
        scaler = Normalizer().fit(X_data)
        X_data = scaler.transform(X_data)
    elif scaler_type == "PowerTransformer":
        scaler = PowerTransformer().fit(X_data)
        X_data = scaler.transform(X_data)
    elif scaler_type == "QuantileTransformer":
        scaler = QuantileTransformer().fit(X_data)
        X_data = scaler.transform(X_data)
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler().fit(X_data)
        X_data = scaler.transform(X_data)

    return X_data

class xgb:
    '''
    Class for everything XGBoost-related. 
    2022/07/20, v. 0
    '''
    def __init__(self, dataset, file_name, custom_dir = None, ml_met = False):
        '''
        Constructor for the xgb class. Checks arguments and makes the
        result directory for saving data.

        2022/07/20, v. 0
        
        Positional arguments:
        dataset:    (srt) Type of dataset for model training or
                    predicting. Options:
                        'mc'    : Monte-Carlo (MC) simulated signal samples.
                        'bkg'   : MC simulated background samples.
                        'sig'   : General MC samples.
                        'ntuple': Raw n-tuples.
                        
        Optional arguments:
        file_name:   (str) The filename, which is later used for creating new directories and
                     saving results.
        custom_dir:  (str) The path to a directory where you want results to
                     to be saved 
        ml_met:      (bool) Calculate, plot, and (optionally) save additional 
                     machine learning metrics.

        Outputs:
        None
        '''
        if dataset not in ["mc", "bkg", "sig", "ntuple"]:
            print_error("Dataset type must be either 'mc' or 'bkg'.")
        else:
            self.dataset = dataset

        if self.dataset == "mc":
            self.file_name = file_name
        elif self.dataset in ["bkg", "sig", "ntuple"]:
            if file_name is None:
                print_error("The dataset name (e.g., file_name = qqToZZTo4L, file_name = DataAboveUpsilonCRSR_MZD_200_55_signal) must be provided!")
                sys.exit()
            else:
                self.file_name = file_name

        if custom_dir is not None:
            self.custom_dir = custom_dir
            if verbose:
                print_alert("Custom directory set to: %s" % self.custom_dir)
        else:
            self.custom_dir = None

        try:
            os.makedirs(resultDir) # create directory for results
        except FileExistsError: # skip if directory already exists
            pass
        
        self.ml_met = ml_met # calculate/plot and optionally save machine learning metrics 

    def split(self, dataframe, drop_evt = True, test = 0.30, random = 7, scaler_type = None, ret = True, verbose = False):
        '''
        Shuffles and splits the features/labels of the dataset given the raw numpy array with 
        shape (events, observables). Pre-scaling can optionally be applied to the data before splitting creating
        test/train split.
        2022/11/22, v. 2
        
        Positional arguments:
        dataframe:         (pd.DataFrame, float/int) The 2D array of event data in tabular format. Rows correspond
                            events and the columns represent the event data. Last column is the label of the
                            event (correct/incorrect permutation).

        Optional arguments:
        drop_evt:     (bool) Drop the 'event' index column. The 'event' index is a book-keeping device that tracks which
                      permutations correspond to which event. This column of the dataframe will look like
                      [0, 0, 0, 1, 1, 1, ..., N, N, N].
        test:         (float) The portion of the data to be reserved as the test dataset. The train dataset is 
                      calculated from 1 - test. Default test size = 0.25.
        scaler_type:  (str) The type of scaling to be applied to the data. Scaler
                      functions from the sklearn.preprocessing class.
        ret:          (bool) Return the single correct pair dataset. Default = True.
        verbose:      (bool) Increase verbosity. Default = False.
        
        Future updates:
        - Making a full metrics printout/saving to a text file, including the confusion matrix, FPR, TNR, FNR, TPR,
          ROC curves, area under the curve, plots, etc. Perhaps creating a new directory for all info (good for 
          running model on different datasets).

        Change log:
        S.D.B., 2022/09/06
            - Added fancy printing from file_utils.

        S.D.B., 2022/11/22
            - Simplified function.
            - Added 
        '''

        if verbose:
            print_alert("\n\n")
            print_alert(60 * '*')
            print_alert("Splitting data into train/test datasets")
            print_alert(60 * '*')
            
        if drop_evt:
            temp   = dataframe.drop(columns=['event']).to_numpy()
            X_data = temp[:, 0:22]             # features 
            Y_data = temp[:,   -1].astype(int) # event labels

            if scaler_type is not None:
                X_data = scale_data(X_data, scaler_type)

            indices   = np.arange(0, len(X_data), dtype = int) # array of indices for all events/permutations

            X_data = np.column_stack([X_data, indices]) # append as new columns the index and event indices
            Y_data = np.column_stack([Y_data, indices]) # append as new columns the index and event indices

            X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_data, Y_data, test_size = 0.7, random_state = 7) # split into 75/25 train/test datasets

            ## separate the total and event indices
            x_train_idx = X_train_full[:, -1].astype(int)
            x_test_idx  = X_test_full[:,  -1].astype(int)
            y_train_idx = y_train_full[:, -1].astype(int)
            y_test_idx  = y_test_full[:,  -1].astype(int)

            ## delete event and overall indices
            X_train = X_train_full[:, :-1:]
            X_test  = X_test_full[:,  :-1:]
            y_train = y_train_full[:, :-1:].reshape(-1) # reshape to (N_events, )
            y_test  = y_test_full[:,  :-1:].reshape(-1) # reshape to (N_events, )

            ## place the train and test datasets in dataframes
            self.trainX = pd.DataFrame(X_train, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
              "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "invMassA0",
              "invMassA1"])
            self.testX  = pd.DataFrame(X_test, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
              "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "invMassA0",
              "invMassA1"])

            self.trainY = pd.DataFrame(y_train, columns = ['pair'])
            self.testY  = pd.DataFrame(y_test,  columns = ['pair'])
        else:
            temp = dataframe.to_numpy()
            X_data = temp[:, 0:23]             # features 
            Y_data = temp[:,   23].astype(int) # event labels

            if scaler_type is not None:
                X_data = scale_data(X_data, scaler_type)

            indices   = np.arange(0, len(X_data), dtype = int) # array of indices for all events/permutations
            event_idx = X_data[:, 20]                          # event indices

            X_data = np.column_stack([X_data, indices, event_idx]) # append as new columns the index and event indices
            Y_data = np.column_stack([Y_data, indices, event_idx]) # append as new columns the index and event indices

            X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_data, Y_data, test_size = 0.7, random_state = 7) # split into 75/25 train/test datasets

            ## separate the total and event indices
            x_train_idx = X_train_full[:, -2].astype(int)
            x_test_idx  = X_test_full[:,  -2].astype(int)
            y_train_idx = y_train_full[:, -2].astype(int)
            y_test_idx  = y_test_full[:,  -2].astype(int)

            ## delete event and overall indices
            X_train = X_train_full[:, :-2:]
            X_test  = X_test_full[:,  :-2:]
            y_train = y_train_full[:, :-2:].reshape(-1) # reshape to (N_events, )
            y_test  = y_test_full[:,  :-2:].reshape(-1) # reshape to (N_events, )

            ## place the train and test datasets in dataframes
            self.trainX = pd.DataFrame(X_train, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
              "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "event", "invMassA0",
              "invMassA1"])
            self.testX  = pd.DataFrame(X_test, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
              "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "event", "invMassA0",
              "invMassA1"])

            self.trainY = pd.DataFrame(y_train, columns = ['pair'])
            self.testY  = pd.DataFrame(y_test,  columns = ['pair'])

        if ret:
            if verbose:
                print_alert("Arrays returned:\ntrainX\ntestX\ntrainY\ntestY")

            return self.trainX, self.testX, self.trainY, self.testY

    def xgb(self, met = True, save = True, filename = None, metrics = ["logloss", "error", "auc"], single_pair = False, ret = True, verbose = True, save_csv = True, save_fig = True):
        '''
        Builds and trains the XGBoost model.
        2022/07/30, v. 0

        Optional arguments:
        met:          (bool) Calculates the metrics. Prints the accuracy, plots the logloss and error.
                      Default = True.
        save:         (bool) Save the XGBoost model to a .sav file. Default = True.
        filename:     (str) The file name used for saving the model. Default = 'path/to/xgboost/results/self.file_name.sav'.
        metrics:      (list; str) A list of strings. Options: 
                          "logloss": Uses the log of the loss function value as the metric.
                          "error":   Uses the computed error of the loss function as the metric.
                          "auc":     Uses the area under the curve (AUC) as the metric.
                      Note: all three can be used; default if met = True is metrics = ["logloss", "error", "auc"],
                      else metrics = ["logloss"].
        single_pair   (bool) Removes duplicates from the dataset after training. Default = False.
        ret:          (bool) Return the single correct pair dataset. Default = True.
        verbose:      (bool) Increase verbosity. Default = False.
        save_csv:     (bool) Save the correct/wrong pair data after training to a file. Default = True.
        save_fig:     (bool) Saves the plots generated during the metric calculation. Default = True.
        
        Change log:
        S.D.B., 2022/09/06
            - Added fancy printing from file_utils.
            - Added additional metrics:
                - Logloss, error, and AUC
                - Plot of confusion matrix
                - Feature importance
        
        Future updates:
        - Making a full metrics printout/saving to a text file, including the confusion matrix, FPR, TNR, FNR, TPR,
          ROC curves, area under the curve, plots, etc. Perhaps creating a new directory for all info (good for 
          running model on different datasets).
        '''
        
        if verbose:
            print_alert("\n\n")
            print_alert(60 * '*')
            print_alert(colors.GREEN + "Building the XGBoost model and training" + colors.ENDC)
            print_alert(60 * '*')
            print_alert("\n")

        global dataDir # the directory for all saved data (plots and csv files).
        
        if self.dataset == "mc":
            dataDir = xgb_results + self.file_name
            try:
                os.makedirs(dataDir) # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass
            
        elif self.dataset == "bkg":
            dataDir = resultDir + self.file_name
            try:
                os.makedirs(dataDir) # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass
            
        elif self.dataset == "sig":
            if self.custom_dir is not None:
                dataDir = resultDir + self.custom_dir + "/" + self.file_name
            else:
                dataDir = resultDir + self.file_name
                
        elif self.dataset == "ntuple":
            if self.custom_dir is not None:
                dataDir = resultDir + self.custom_dir + "/" + self.file_name
            else:
                dataDir = resultDir + "/ntuple_" % self.file_name 

        try:
            os.makedirs(dataDir) # create directory for data/plots
        except FileExistsError: # skip if directory already exists
            pass
        
        if not met:
            metrics = ["logloss"]
        
        model = XGBClassifier(n_jobs = -1, use_label_encoder = False, eval_metric = metrics, random_state = 7) # build the classifier
        model.fit(self.trainX, self.trainY) # fit the classifier to the training data
        
        if save: # save the model to disk
            if filename is not None:
                joblib.dump(model, filename) # use a custom file name
            else:
                joblib.dump(model, dataDir + "/%s" % self.file_name)

        if met or self.ml_met:
            predictedY = model.predict(self.testX)
            print_alert('\nTraining Classification Report:\n\n', classification_report(self.testY, predictedY))
            
            if save:
                class_report_path = dataDir + "/class_report.json"
                with open(class_report_path, "w") as file:
                    json.dump(classification_report(self.testY, predictedY, output_dict = True), file)
                
                if verbose:
                    print_alert("Classification report saved to %s" + class_report_path)
            
            results  = model.evals_result() 
            conf_mat = confusion_matrix(predictedY, testY)
            classes  = ["Negative", "Positive"]
            types    = [['TN','FP'], ['FN', 'TP']]

            fig, ax = plt.subplots(figsize = (12, 12))  
            im      = ax.imshow(conf_mat, interpolation='nearest', cmap = plt.cm.winter)
            cbar    = fig.colorbar(im, fraction = 0.0458, pad = 0.04, label = "Number classifications")
            ax.set_title("Confusion matrix")
            ax.set_xticks([0, 1], classes)
            ax.set_yticks([0, 1], classes, rotation = 90)
            ax.set_xlabel("Predicted class", fontsize = 30)
            ax.set_ylabel("True class", fontsize = 30)

            for ii in range(2):
                for jj in range(2):
                    plt.text(jj - 0.2, ii, "%s = %d" % (types[ii][jj], conf_mat[ii][jj]))
            
            if save_fig:
                fig.savefig(dataDir + "/confusion_matrix.pdf")
            
            valid_sets  = list(results.keys())
            metric_keys = list(results[valid_sets[0]].keys())
            
            ## manually check number of wrongly classified events
            testY_shaped = testY.to_numpy(dtype = int).reshape(-1) # convert test labels to numpy int array and reshape to 1D
            wrong_class  = np.where(predictedY != testY_shaped)[0]  # get array of indices where predicted labels != actual labels
            ratio_wrong  = len(wrong_class) / len(testY_shaped)     # calculate ratio
            
            print_alert("Ratio of incorrectly classified events for test dataset: %.2f" % ratio_wrong)     
        
            accuracy = accuracy_score(self.testY, predictedY)
            print_alert("Accuracy calculated using metrics.accuracy_score(): %.2f%%" % (accuracy * 100.0))
            
            ##### plotting #####
            ## create individual plots for all metrics
            for key in metrics: # loop over and plot metrics
                fig, ax = plt.subplots(figsize = (12, 12))
                ax.grid(zorder = 0)
                for valid in valid_sets:
                    ax.plot(x_axis, results[valid][key], label = traintest_dict[valid])

                ax.legend()
                ax.set_xlabel('Epoch',  loc = "right")
                ax.set_ylabel('%s' % metric_dict[key], loc = "top")
                fig.tight_layout()
                
                if save_fig:
                    fig.savefig(dataDir + "/%s.pdf" % metric_dict[key])
                    
            ## create plots for all metrics
            if len(metrics) == 1: # skip if only one metric
                pass
            else:
                fig, ax = plt.subplots(1, len(metrics), figsize = (20, 10)) # create grid of plots
                ax.ravel() # unravel axes object to easily iterate over it
                for metric, plot in zip(metrics, range(len(metrics))):
                    for valid in valid_sets:
                        ax[plot].plot(x_axis, results[valid][metric], label = traintest_dict[valid])

                    ax[plot].grid(zorder = 0)
                    ax[plot].legend()
                    ax[plot].set_xlabel('Epoch',  loc = "right")
                    ax[plot].set_ylabel('%s' % metric_dict[metric], loc = "top")

                fig.tight_layout()
                fig.suptitle("XGBoost metrics")
                fig.subplots_adjust(top = 0.8)
            
                if save_fig:
                    fig.savefig(dataDir + "/all_metrics.pdf")
        
        y_train_all = pd.DataFrame(self.y_train_full, columns = ['pair', 'index', 'evt_idx'], dtype = int)
        y_test_all  = pd.DataFrame(self.y_test_full,  columns = ['pair', 'index', 'evt_idx'], dtype = int)

        ## total dfs with all info
        x_train_all = pd.DataFrame(self.X_train_full, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
          "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "event", "invMassA0",
          "invMassA1", "evt_idx"])
        x_test_all  = pd.DataFrame(self.X_test_full, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
          "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "event", "invMassA0",
          "invMassA1", "evt_idx"])
        
        x_total = pd.concat([x_train_all_info, x_test_all_info], axis = 0) # form total feature/indices dataframe
        y_total = pd.concat([y_train_all, y_test_all], axis = 0)           # form total label/indices dataframe

        assert x_total["evt_idx"].all() == y_total["evt_idx"].all() # make sure all indices (of unique permutations/events) are in order for 1-to-1 comparison

        self.total_x_df = x_total.drop("evt_idx")                          # remove unique index so the data can be fed to the model for prediction
        self.total_x_df['predict'] = model.predict(total_df_x).astype(int) # predict for all events/permutations

        self.correct_pair = self.total_x_df[self.total_x_df['predict'] == 1] # select correct matches (can include permutations incorrectly classified!)
        self.wrong_pair   = self.total_x_df[self.total_x_df['predict'] == 0] # select the wrong matches

        ## manually check number of wrongly classified events for entire dataset
        all_y_labels     = y_total['pair'].to_numpy(dtype = int).reshape(-1)             # convert test labels to numpy int array and reshape to 1D
        predicted_all    = self.total_x_df['predict'].to_numpy(dtype = int).reshape(-1)  # convert predicted labels to numpy int array and reshape to 1D
        all_wrong_class  = np.where(predicted_all != all_y_labels)[0]                    # get array of indices where predicted labels != actual labels
        all_ratio_wrong  = len(all_wrong_class) / len(all_y_labels)                      # calculate ratio of incorrect classifications

        print_alert("Incorrectly classified events for entire dataset: %.2f" % all_ratio_wrong) 
        
        if save_csv: # saved paired muons to a .csv for importing later
            self.correct_pair.to_csv(dataDir + ("/correct_pair_%s.csv" % self.file_name))
            self.wrong_pair.to_csv(dataDir + ("/wrong_pair_%s.csv"     % self.file_name))

        self.model_run = True # set bool to True to indicate this dataset was used for training
        
        if single_pair: # remove duplicate events (this includes incorrectly classified events!)
            self.single_correct_pair = self.correct_pair.drop_duplicates(subset = ['event', 'predict'], keep = 'last') 
            self.single_wrong_pair   = self.wrong_pair.drop_duplicates(subset   = ['event', 'predict'], keep = 'last')
            if save_csv:
                self.single_correct_pair.to_csv(dataDir + ("/single_correct_pair_%s.csv" % self.file_name))
                self.single_wrong_pair.to_csv(dataDir   + ("/single_wrong_pair_%s.csv"   % self.file_name))

        if ret:
            if single_pair:
                if verbose:
                    print_alert("Arrays returned:\ncorrect_pair\nwrong_pair\nsingle_correct_pair\nsingle_wrong_pair")
        
                return self.correct_pair, self.wrong_pair. self.single_correct_pair, self.single_wrong_pair
            else:
                if verbose:
                    print_alert("Arrays returned:\ncorrect_pair\nwrong_pair")
                return self.correct_pair, self.wrong_pair

    def predict(self, dataframe_shaped, labels = None, filename = "MZD_200_55_pd_model.sav", single_pair = False, ret = False, verbose = True, save_csv = True):
        '''
        Predicts the correct order/pairing of muons using a trained XGBoost model loaded
        from a saved file.
        2022/07/30, v. 0
        
        Positional arguments:
        dataframe_shaped: 

        Optional arguments:
        labels:       (np.array, list; int) Array/list of labels (for all permutations) for mc samples.
        met:          (bool) Calculates the metrics. Prints the accuracy, plots the logloss and error.
                      Default = True.
        filename:     (str) The filename of the saved the XGBoost model. Default = 'MZD_200_55_pd_model.sav'.
        metrics:      (list; str) A list of strings. Options: 
                          "logloss": Uses the log of the loss function value as the metric.
                          "error":   Uses the computed error of the loss function as the metric.
                          "auc":     Uses the area under the curve (AUC) as the metric.
                      Note: all three can be used; default if met = True is metrics = ["logloss", "error", "auc"],
                      else metrics = ["logloss"].
        single_pair   (bool) Removes duplicates from the dataset after training. Default = False.
        ret:          (bool) Return the single correct pair dataset. Default = True.
        verbose:      (bool) Increase verbosity. Default = False.
        save_csv:     (bool) Save the correct/wrong pair data after training to a file. Default = True.
        save_fig:     (bool) Saves the plots generated during the metric calculation. Default = True.
        
        Change log:
        S.D.B., 2022/09/06
            - Added fancy printing from file_utils.
            - Added additional metrics:
                - Logloss, error, and AUC
                - Plot of confusion matrix
                - Feature importance
        
        Future updates:
        - Adding comparison of true vs. predicted labels, confusion matrix.
        '''
        if self.dataset in ["mc", "bkg", "sig", "ntuple"]:
            global dataDir
            if self.dataset == "bkg":
                dataDir = resultDir + "/" + self.file_name
            elif self.dataset == "sig":
                if self.custom_dir is not None:
                    dataDir = resultDir + "/" + self.custom_dir + "/" + self.file_name
                else:
                    dataDir = resultDir + "/" + self.file_name
            elif self.dataset == "ntuple":
                if self.custom_dir is not None:
                    dataDir = resultDir + "/" + self.custom_dir + "/" + self.file_name
                else:
                    dataDir = resultDir + "/" + self.file_name 

            try:
                os.makedirs(dataDir) # create directory for data/plots
            except FileExistsError: # skip if directory already exists
                pass
            
            try:
                os.makedirs(dataDir) # create directory for data/plots
            except FileExistsError: # skip if directory already exists
                pass
        if self.dataset == "mc" and labels is none:
            print_alert("WARNING: To check gen/sel pairing accuracy for MC samples, please provide an array of labels when calling the function (e.g., labels = [1, 0, 0, 1,...]")
            print_alert("Classifying without checking the accuracy...")

        if verbose:
            print_alert("\n\n")
            print_alert(60 * '*')
            print_alert("Loading trained XGBoost model from %s" % filename)
            print_alert(60 * '*')

        loaded_model = joblib.load(filename)

        x_data = dataframe_shaped[:, 0:23]   
        y_data = dataframe_shaped[:, 20:24]

        x_df = pd.DataFrame(x_data, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
          "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "event", "invMassA0",
          "invMassA1"])
        
        y_df     = pd.DataFrame(y_data, columns = ['event', 'invmA0', 'invmA1', 'pair'])
        x_sorted = x_df.sort_values('event')

        predY = loaded_model.predict(x_sorted)
        arr   = predY

        x_sorted['Predict'] = arr.tolist()

        if verbose:
            print_alert("\n\n")
            print_alert(60 * '*')
            print_alert("Determining correct and wrong pairs for the %s dataset" % self.file_name)
            print_alert(60 * '*')

        self.correct_pair = x_sorted[x_sorted['Predict'] == 1] 
        self.wrong_pair   = x_sorted[x_sorted['Predict'] == 0] 

        if save_csv:
            self.correct_pair.to_csv(dataDir + ("/correct_pair_%s.csv" % self.file_name))
            self.wrong_pair.to_csv(dataDir + ("/wrong_pair_%s.csv" % self.file_name))

        if single_pair:
            self.single_correct_pair = self.correct_pair.drop_duplicates(subset = ['event', 'Predict'], keep = 'last')
            self.single_wrong_pair   = self.wrong_pair.drop_duplicates(subset   = ['event', 'Predict'], keep = 'last')

            if save_csv:
                self.single_correct_pair.to_csv(dataDir + ("/single_correct_pair_%s.csv" % self.file_name))
                self.single_wrong_pair.to_csv(dataDir + ("/single_wrong_pair_%s.csv" % self.file_name))

        if ret:
            return self.single_correct_pair
            
    def plotMatch(self, save = True, verbose = False, key_bkg = None):
        if key_bkg is not None and self.dataset == "sig":
            print(colors.YELLOW + "Background keys should not be provided when analyzing signal data! Please provide key as the argument key_sig =" + colors.ENDC)

        if verbose:
            print("\n\n")
            print(60 * '*')
            print(colors.GREEN + "Plotting correctly and incorrectly matched muons" + colors.ENDC)
            print(60 * '*')

        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))
        ax = ax.ravel()

        for pair in range(2):
            if pair == 0:
                self.correct_pair.plot(x = 'invMassA0', y = 'invMassA1', kind = "scatter", ax = ax[pair], color = 'darkred', grid = True, zorder = 3)
                ax[pair].set_title('Correct Pair')
                ax[pair].set_ylim(0, 80)
                ax[pair].set_xlim(0, 80)
            elif pair == 1:
                self.wrong_pair.plot(x = 'invMassA0', y = 'invMassA1', kind = "scatter", ax = ax[pair], color = 'darkred', grid = True, zorder = 3)
                ax[pair].set_title('Wrong Pair')
                ax[pair].set_ylim(0, 250)
                ax[pair].set_xlim(0, 250)
                
            #ax[pair].grid(zorder = 0)
            ax[pair].set_xlabel(r'$m_{\mu\mu_{1}}$[GeV]', loc = 'right')
            ax[pair].set_ylabel(r'$m_{\mu\mu_{2}}$[GeV]', loc = 'top')

            if key_bkg is not None:
                fig.suptitle(key_dict[key_bkg])

            if self.dataset == "sig":
                fig.suptitle(r'$m_{Z_{D}} = %s$ GeV, $m_{f_{D1}} = %s$ GeV' % (self.file_name.split("_")[1],  self.file_name.split("_")[2]))

        fig.tight_layout()

        if save:
            if verbose:
                print(colors.YELLOW + "Figure save at %s" % dataDir + ("/2DInvMass_dPhiCor_%s.pdf" % self.file_name) + colors.ENDC)

            fig.savefig(dataDir + ("/2DInvMass_dPhiCor_%s.pdf" % self.file_name))

        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (20, 20)) #, constrained_layout = True)
        ax      = ax.ravel()      
        partial = "invMassA"

        for pair in range(4):
            if pair <= 1:
                if pair == 0:
                    key = partial +  "0"
                    ax[pair].set_xlabel(r'$m_{\mu\mu_{1}}$[GeV]', loc = 'right')
                elif pair == 1:
                    key = partial +  "1"
                    ax[pair].set_xlabel(r'$m_{\mu\mu_{2}}$[GeV]', loc = 'right')
                self.correct_pair[key].plot.hist(bins = 100, alpha = 0.9, range = (0, 100), color = 'darkred', ax = ax[pair], grid = True, zorder = 3)
                ax[pair].set_title('Correct Pair')
            elif pair > 1:
                if pair == 2:
                    key = partial +  "0"
                    ax[pair].set_xlabel(r'$m_{\mu\mu_{1}}$[GeV]', loc = 'right')
                elif pair == 3:
                    key = partial +  "1"
                    ax[pair].set_xlabel(r'$m_{\mu\mu_{2}}$[GeV]', loc = 'right')
                    
                self.wrong_pair[key].plot.hist(bins = 100, alpha = 0.9, range = (0, 200), color = 'darkred', grid = True, ax = ax[pair], zorder = 3)
                ax[pair].set_title('Wrong Pair')
            
            #ax[pair].grid(zorder = 0)
            ax[pair].set_ylabel("Frequency", loc = "top")

        if key_bkg is not None:
            fig.suptitle(key_dict[key_bkg])

        if self.dataset == "sig":
            fig.suptitle(r'$m_{Z_{D}} = %s$ GeV, $m_{f_{D1}} = %s$ GeV' % (self.file_name.split("_")[1], self.file_name.split("_")[2]))

        fig.tight_layout()

        if save:
            if verbose:
                print(colors.YELLOW + "Figure save at %s" % dataDir + ("/1DInvMass_dPhiCor_%s.pdf" % self.file_name) + colors.ENDC)
            
            fig.savefig(dataDir + ("/1DInvMass_dPhiCor_%s.pdf" % self.file_name), bbox_inches='tight')


