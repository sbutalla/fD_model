from tuning.hyper_parameter_tuning_xgb import xgb
from tuning.plotting import plot_data, heat_map
from utilities.utilities import process_data
from MLtools.xgb_v3 import xgb as new_xgb
import json
import time
import pprint
import pandas as pd
import os
from tqdm import tqdm
import warnings

root_file = (
    "/Users/spencerhirsch/Documents/research/root_files/MZD_200_ALL/MZD_200_55.root"
)
root_dir = "cutFlowAnalyzerPXBL4PXFL3;1/Events;1"
resultDir = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/MZD_200_55_pd_model"


'''
    Function that deals with processing data. Utilizes the process_data class from the utilities 
    directory. Returns a final array with all of the values necessary for generating the models.
'''


def process():
    data = process_data("mc")  # declare data processing object
    data.extract_data(root_file, root_dir)
    data.prelim_cuts()
    data.dRgenCalc()
    data.SSM()

    d_r_cut = 0.05
    cut = "all"  # cut type ("all" applies cuts to every observable)
    data.dRcut(d_r_cut, cut)

    data.permutations()
    data.invMassCalc()
    data.dR_diMu()
    data.permutations()
    data.invMassCalc()
    data.dR_diMu()
    final_array = data.fillAndSort(save=True)

    return final_array


'''
    Function handles data processing for building the various xgboost models for analysis. Stores the data in
    their respective files and directories. Calls all of the necessary functions and creates all objects
    used for analysis. Driver function to collect all data for further analysis. 
'''


def xgmain():
    warnings.filterwarnings("ignore")
    start = time.time()
    final_array = process()
    boost = xgb("mc")
    boost.split(final_array)

    '''
        Arrays to store the values of the hyperparameters that are run with the model. Need to look into
        using a more efficient testing algorithm. Bayesian hyperparameter tuning?

        Removed gblinear for booster parameter, inefficient and complications with xgboost, data for
        default stored in archive file.
    '''

    booster_list = ['gbtree', 'dart'] # Determined gbtree is best
    alpha_array = [0, 1, 2, 3, 4, 5]   # L1 Regularization
    lambda_array = [0, 1, 2, 3, 4, 5]   # L2 Regularization
    eta_array = [0.6, 0.5, 0.4, 0.3, 0.1]    # Learning rate
    max_depth_array = [3, 6, 10, 12, 15]   # Maximum depth of the tree
    objective_array = ['binary:logistic', 'binary:hinge', 'reg:squarederror']

    '''
        Iterate through all of the hyper parameters. Currently looking into the learning rate (eta) and the max
        depth of the tree.
    '''

    # for val_eta in tqdm(eta_array):
    #     for val_alpha in alpha_array:
    #         for val_lambda in lambda_array:
    #             for val_obj in objective_array:
    #                 for val_max_depth in max_depth_array:
    #                     _ = boost.xgb(single_pair=True, ret=True, eta=val_eta, max_depth=val_max_depth,
    #                                   reg_lambda=val_lambda, reg_alpha=val_alpha, objective=val_obj)


    '''
        Testing purposes, got conflicting results for MCC of default values. Quite a large difference, 
        testing is necessary.
    '''
    for val_eta in tqdm(eta_array):
        for val_alpha in alpha_array:
            for val_lambda in lambda_array:
                for val_max_depth in max_depth_array:
                    _ = boost.xgb(single_pair=True, ret=True, eta=val_eta, max_depth=val_max_depth,
                                  reg_lambda=val_lambda, reg_alpha=val_alpha, objective='reg:squarederror')


    '''
        Iterating over only the values of learning rate and the maximum depth of the tree.
        This data has been collected and logged in the archive file in the external SSD.
    '''
    # for val1 in eta_array:
    #     for val2 in max_depth_array:
    #         _ = boost.xgb(single_pair=True, ret=True, eta=val1, max_depth=val2)

    boost.model_list.sort(key=lambda x: (x.mcc, x.accuracy), reverse=True)

    obj_list = []
    for val in boost.model_list:
        obj_list.append(val.get_model())

    print("Completed.")

    class_out = resultDir + "/model_list.json"
    out_file = open(class_out, "w")
    json.dump(obj_list, out_file, indent=4)

    end = time.time()
    total = end - start
    t_hours = (total / 60) / 60

    class_out = resultDir + "/time.json"
    out_file = open(class_out, "w")
    json.dump(t_hours, out_file)
    print(t_hours)


def draw_tree():
    final_array = process()
    boost = xgb("mc")
    boost.split(final_array)

    dir = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/archive/data_102522_346PM/' \
          'MZD_200_55_pd_model/model_list.json'

    f = open(dir)
    data = json.load(f)
    data = sorted(data, key=lambda x: x['mcc'], reverse=True)
    pprint.pprint(data)

    value = data[0]

    val_eta = value['eta']
    val_max_depth = value['max depth']
    val_alpha = value['l1']
    val_lambda = value['l2']

    _ = boost.xgb(single_pair=True, ret=True, eta=val_eta, max_depth=val_max_depth,
                  reg_lambda=val_lambda, reg_alpha=val_alpha, tree=True)


def pre_processed(all_models=False):
    preprocessed_dir = "/Volumes/SA Hirsch/Florida Tech/research/dataframe_csv_fD_model"
    zd_mass = None
    fd1_mass = None
    path_dict = {}
    model_list = []
    default_list = []
    optimal_list = []

    with open('/Volumes/SA Hirsch/Florida Tech/research/sorted_csv_list_mine.json') as json_file:
        path_dict = json.load(json_file)
    temp_path = {}
    if not all_models:
        for outer in path_dict:
            if outer in str([85, 95, 150, 200, 300, 400]):
                inner = {}
                path_list = list(path_dict[outer].items())
                first = path_list[0]
                first_dict = {first[0]: first[1]}
                last = path_list[-1]
                last_dict = {last[0]: last[1]}
                inner.update(first_dict)
                inner.update(last_dict)
                temp_path[outer] = inner

        path_dict = temp_path

    parent = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/optimized_all/'
    dict_of_model_direct = {}
    for outer in path_dict:
        # dict_of_model_direct = outer
        parent_dir = parent + "/" + outer
        try:
            os.makedirs(parent_dir)
        except FileExistsError:
            pass
        inner_dict = {}
        for inner in path_dict[outer]:
            child_path = parent_dir + "/" + inner # Path to the child
            inner_dict.update({inner: child_path})
            try:
                os.makedirs(child_path)
            except FileExistsError:
                pass

            path = path_dict[outer][inner]
            first_split = path.split('/')
            useful = first_split[-1]
            useful_list = useful.split('_')
            for i in range(len(useful_list)):
                if useful_list[i] == "MZD":
                    zd_mass = useful_list[i + 1]
                    fd1_mass = useful_list[i + 2].partition(".")[0]
                    default, optimal = run_xgb(pd.read_csv(path_dict[outer][inner]), zd_mass, fd1_mass, child_path)
                    model_list.append(default)
                    model_list.append(optimal)
                    default_list.append(default)
                    optimal_list.append(optimal)
        dict_of_model_direct[outer] = inner_dict

    # after running all of the datasets, retrieve all of the values stored in the list
    model_list.sort(key=lambda x: (x.mcc, x.accuracy), reverse=True)
    default_list.sort(key=lambda x: (x.mcc, x.accuracy), reverse=True)
    optimal_list.sort(key=lambda x: (x.mcc, x.accuracy), reverse=True)


    obj_list = []
    for val in model_list:
        obj_list.append(val.get_model())

    default_obj = []
    for val in default_list:
        default_obj.append(val.get_model())

    optimal_obj = []
    for val in optimal_list:
        optimal_obj.append(val.get_model())


    print("Completed.")

    class_out = parent + "model_list.json"
    out_file = open(class_out, "w")
    json.dump(obj_list, out_file, indent=4)

    default_out = parent + "default_model_list.json"
    default_out_file = open(default_out, "w")
    json.dump(default_obj, default_out_file, indent=4)

    optimal_out = parent + "optimal_model_list.json"
    optimal_out_file = open(optimal_out, "w")
    json.dump(optimal_obj, optimal_out_file, indent=4)

    # end = time.time()
    # total = end - start
    # t_hours = (total / 60) / 60

    # class_out = resultDir + "/time.json"
    # out_file = open(class_out, "w")
    # json.dump(t_hours, out_file)
    # print(t_hours)


def run_xgb(final_array, zd_mass, fd1_mass, result_dir, optimal=False):
    warnings.filterwarnings("ignore")
    start = time.time()
    filename = "MZD_%s_%s" % (zd_mass, fd1_mass)
    boost = new_xgb("mc", filename)
    old_boost = xgb('mc')
    trainX, testX, trainY, testY = boost.split(final_array)

    '''
        Arrays to store the values of the hyperparameters that are run with the model. Need to look into
        using a more efficient testing algorithm. Bayesian hyperparameter tuning?

        Removed gblinear for booster parameter, inefficient and complications with xgboost, data for
        default stored in archive file.
    '''

    hyper_parameters = {'alpha': [0, 1, 2, 3, 4, 5], 'lambda': [0, 1, 2, 3, 4, 5], 'eta': [0.6, 0.5, 0.4, 0.3, 0.1],
                        'max': [3, 6, 10, 12, 15], 'objective': ['binary:logistic', 'binary:hinge', 'reg:squarederror']}

    if optimal:
        # Run with the default parameters
        default = old_boost.xgb(zd_mass, fd1_mass, result_dir, trainX, testX, trainY, testY, single_pair=True, ret=True, eta=0.3, max_depth=6,
                      reg_lambda=1, reg_alpha=0, objective='reg:squarederror')
        # Run with the optimal parameters
        optimal = old_boost.xgb(zd_mass, fd1_mass, result_dir, trainX, testX, trainY, testY, single_pair=True, ret=True, eta=0.6, max_depth=6,
                      reg_lambda=2, reg_alpha=4, objective='binary:logistic')
        return default, optimal
    else:
        for val_eta in tqdm(hyper_parameters['eta']):
            for val_alpha in hyper_parameters['alpha']:
                for val_lambda in hyper_parameters['lambda']:
                    for val_obj in hyper_parameters['objective']:
                        for val_max_depth in hyper_parameters['max']:
                            _ = boost.xgb(trainX, testX, trainY, testY, single_pair=True, ret=True, eta=val_eta, max_depth=val_max_depth,
                                          reg_lambda=val_lambda, reg_alpha=val_alpha, objective=val_obj)



def run_on_all(all_models=False):
    preprocessed_dir = "/Volumes/SA Hirsch/Florida Tech/research/dataframe_csv_fD_model"
    zd_mass = None
    fd1_mass = None
    path_dict = {}
    model_list = []
    default_list = []
    optimal_list = []

    with open('/Volumes/SA Hirsch/Florida Tech/research/sorted_csv_list_mine.json') as json_file:
        path_dict = json.load(json_file)
    temp_path = {}
    if not all_models:
        for outer in path_dict:
            if outer in str([85, 95, 150, 200, 300, 400]):
                inner = {}
                path_list = list(path_dict[outer].items())
                first = path_list[0]
                first_dict = {first[0]: first[1]}
                last = path_list[-1]
                if int(last[0]) > 55:       #
                    for val in path_list:  #
                        if '55' in val[1]: #
                            last = val  #
                last_dict = {last[0]: last[1]}
                inner.update(first_dict)
                inner.update(last_dict)
                temp_path[outer] = inner
            else:   #
                inner = {}
                path_list = list(path_dict[outer].items())
                first = path_list[0]
                first_dict = {first[0]: first[1]}
                inner.update(first_dict)
                temp_path[outer] = inner

        path_dict = temp_path

    parent = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/optimized_all/'

    class_out = parent + "paths.json"
    out_file = open(class_out, "w")
    json.dump(path_dict, out_file)

    dict_of_model_direct = {}
    all_models_list = []
    overall_time = time.time()
    for outer in path_dict:
        # dict_of_model_direct = outer
        parent_dir = parent + outer
        try:
            os.makedirs(parent_dir)
        except FileExistsError:
            pass
        inner_dict = {}
        for inner in path_dict[outer]:
            child_path = parent_dir + "/" + inner  # Path to the child
            inner_dict.update({inner: child_path})
            try:
                os.makedirs(child_path)
            except FileExistsError:
                pass

            path = path_dict[outer][inner]
            first_split = path.split('/')
            useful = first_split[-1]
            useful_list = useful.split('_')
            for i in range(len(useful_list)):
                if useful_list[i] == "MZD":
                    zd_mass = useful_list[i + 1]
                    fd1_mass = useful_list[i + 2].partition(".")[0]

            data = pd.read_csv(path_dict[outer][inner])
            warnings.filterwarnings("ignore")
            start = time.time()
            filename = "MZD_%s_%s" % (zd_mass, fd1_mass)
            boost = new_xgb("mc", filename)
            old_boost = xgb('mc')
            trainX, testX, trainY, testY = boost.split(data)

            hyper_parameters = {'alpha': [0, 1, 2, 3, 4, 5], 'lambda': [0, 1, 2, 3, 4, 5],
                                'eta': [0.6, 0.5, 0.4, 0.3, 0.1],
                                'max': [3, 6, 10, 12, 15],
                                'objective': ['binary:logistic', 'binary:hinge', 'reg:squarederror']}
            model_list = []
            sample_start = time.time()
            for val_eta in tqdm(hyper_parameters['eta']):
                for val_alpha in hyper_parameters['alpha']:
                    for val_lambda in hyper_parameters['lambda']:
                        for val_obj in hyper_parameters['objective']:
                            for val_max_depth in hyper_parameters['max']:
                                model_object = old_boost.xgb(zd_mass, fd1_mass, child_path, trainX, testX, trainY, testY, single_pair=True, ret=True, eta=val_eta,
                                              max_depth=val_max_depth,
                                              reg_lambda=val_lambda, reg_alpha=val_alpha, objective=val_obj)
                                model_list.append(model_object)
                                all_models_list.append(model_object)

            sample_end = time.time()
            sample_time = sample_end - sample_start
            t_hours = (sample_time / 60) / 60

            class_out = child_path + "/time.json"
            out_file = open(class_out, "w")
            json.dump(t_hours, out_file)
            print(t_hours)

            model_list.sort(key=lambda x: (x.mcc, x.accuracy), reverse=True)

            obj_list = []
            for val in model_list:
                obj_list.append(val.get_model())

            class_out = child_path + "/model_list.json"
            out_file = open(class_out, "w")
            json.dump(obj_list, out_file, indent=4)


        dict_of_model_direct[outer] = inner_dict


    all_models_list.sort(key=lambda x: (x.mcc, x.accuracy), reverse=True)

    overall_end = time.time()

    overall = overall_end - overall_time
    t_hours = (overall / 60) / 60

    class_out = parent + "time.json"
    out_file = open(class_out, "w")
    json.dump(t_hours, out_file)
    print(t_hours)

    obj_list = []
    for val in all_models_list:
        obj_list.append(val.get_model())

    class_out = parent + "model_list.json"
    out_file = open(class_out, "w")
    json.dump(obj_list, out_file, indent=4)


'''
    Driver function that handles each respective function. Takes input from the standard input stream
    and calls each function for its desired ability specified by the user. Makes the code much cleaner 
    and more concise.
'''

def main():
    print('Which program would you like to use:\n(1): Generate permutations of models \n(2): Generate Heat Map '
          '\n(3): Plot data \n(4): Output tree\n(5): Run preprocessed data\n')

    choice = input('Choice: ')

    if choice == '1':
        xgmain()
    elif choice == '2':
        print("Which metric would you like to plot?\n (1): time\n (2): mcc\n (3): f1")
        input_val = input('Choice: ')
        metric = ''

        if input_val == '1':
            metric = 'time'
        elif input_val == '2':
            metric = 'mcc'
        elif input_val == '3':
            metric = 'f1'
        else:
            print('Invalid input.')

        heat_map(metric)
    elif choice == '3':
        plot_data()
    elif choice == '4':
        draw_tree()
    elif choice == '5':
        # pre_processed()
        run_on_all()
    else:
        print("Input invalid.")


main()

