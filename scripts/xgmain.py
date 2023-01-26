import json
from MLtools.xgb_v3 import xgb as new_xgb
import os
import pandas as pd
import pprint
import sys
import time
from tqdm import tqdm
from tuning.hyper_parameter_tuning_xgb import xgb as xgb
from tuning.plotting import plot_data, heat_map

#sys.path.insert(0, "/Users/spencerhirsch/Documents/GitHub/fD_model/scripts/utilities/")
sys.path.insert(0, "/Users/spencer/Documents/GitHub/fD_model/scripts/utilities")
from utilities import process_data
import warnings

parent_directory = "/Volumes/SA Hirsch/Florida Tech/research/dataframes"

"""
    Function that deals with processing data. Utilizes the process_data class from the utilities 
    directory. Returns a final array with all of the values necessary for generating the models.
    Some samples already have preprocessed data with them, no reason to process data if .csv file
    is already generated for the sample.
"""


def process(root_file, root_dir):
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


def find_masses(root_file, root_file_bool):
    zd_mass, fd_mass, result_directory = "", "", ""
    if root_file_bool:
        zd_mass = root_file.split("/")[-1].split("_")[1]
        fd_mass = root_file.split("/")[-1].split("_")[2].split(".")[0]
        result_directory = parent_directory + "/MZD_%s_%s_fd_model" % (zd_mass, fd_mass)

    return zd_mass, fd_mass, result_directory


"""
    Given the root file for a single event, perform hyper-parameter tuning on single model. Processes the model
    to return the final array for the data. Runs all data through the parameters to generate all models for each
    set up parameters.
"""

def single_preprocessed():
    warnings.filterwarnings("ignore")
    parent = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/single_dataset/"
    filename = '/Volumes/SA Hirsch/Florida Tech/research/archive_csv_fD_model/MZD_200/MZD_200_55/total_df_MZD_200_55.csv'
    data = pd.read_csv(filename)
    zd_mass = 200
    fd1_mass = 55
    resultant_filename = ('MZD_%s_%s', zd_mass, fd1_mass)
    start = time.time()
    boost = new_xgb("mc", resultant_filename)
    old_boost = xgb("mc")
    train_x, test_x, train_y, test_y = boost.split(data)

    hyper_parameters = {
        "alpha": [0, 1, 2, 3, 4, 5],
        "lambda": [0, 1, 2, 3, 4, 5],
        "eta": [0.6, 0.5, 0.4, 0.3, 0.1],
        "max": [3, 6, 10, 12, 15],
        "objective": ["binary:logistic", "binary:hinge", "reg:squarederror"],
    }

    model_list = []
    sample_start = time.time()
    for val_eta in tqdm(hyper_parameters["eta"]):
        for val_alpha in hyper_parameters["alpha"]:
            for val_lambda in hyper_parameters["lambda"]:
                for val_obj in hyper_parameters["objective"]:
                    for val_max_depth in hyper_parameters["max"]:
                        model_object = old_boost.xgb(
                            zd_mass,
                            fd1_mass,
                            parent,
                            train_x,
                            test_x,
                            train_y,
                            test_y,
                            single_pair=True,
                            ret=True,
                            eta=val_eta,
                            max_depth=val_max_depth,
                            reg_lambda=val_lambda,
                            reg_alpha=val_alpha,
                            objective=val_obj,
                        )
                        model_list.append(model_object)

    end = time.time()
    sample_time = end - start
    t_hours = (sample_time / 60) / 60

    class_out = parent + "time.json"
    out_file = open(class_out, "w")
    json.dump(t_hours, out_file)
    print(t_hours)




def process_single():
    # Setting up file system
    root_file = (
        "/Users/spencerhirsch/Documents/research/root_files/MZD_200_ALL/MZD_200_55.root"
    )
    root_dir = "cutFlowAnalyzerPXBL4PXFL3;1/Events;1"
    zd_mass, fd_mass, result_directory = find_masses(root_file, True)

    """
        In the case of the root file, extract all of the necessary data and store it into a dataframe for
        training the xgboost model. In the case of preprocessed data this has already been taken care of.
    """

    warnings.filterwarnings("ignore")
    start = time.time()
    final_array = process(root_file, root_dir)
    boost = xgb("mc")
    boost.split(final_array)

    """
        Hyper-parameter tuning, grid search method. Hard coded values for the hyper-parameters we are interested in
        testing.
    """

    alpha_array = [0, 1, 2, 3, 4, 5]  # L1 Regularization
    lambda_array = [0, 1, 2, 3, 4, 5]  # L2 Regularization
    eta_array = [0.6, 0.5, 0.4, 0.3, 0.1]  # Learning rate
    max_depth_array = [3, 6, 10, 12, 15]  # Maximum depth of the tree
    objective_array = ["binary:logistic", "binary:hinge", "reg:squarederror"]

    """
        Construct each model to be stored for 
    """

    for val_eta in tqdm(eta_array):
        for val_alpha in alpha_array:
            for val_lambda in lambda_array:
                for val_obj in objective_array:
                    for val_max_depth in max_depth_array:
                        _ = boost.xgb(
                            zd_mass=zd_mass,
                            fd1_mass=fd_mass,
                            result_dir=result_directory,
                            train_x=pd.DataFrame(),
                            test_x=pd.DataFrame(),
                            train_y=pd.DataFrame(),
                            test_y=pd.DataFrame(),
                            eta=val_eta,
                            max_depth=val_max_depth,
                            reg_lambda=val_lambda,
                            reg_alpha=val_alpha,
                            objective=val_obj,
                            single_pair=True,
                            ret=True,
                        )

    boost.model_list.sort(
        key=lambda x: (x.mcc, x.accuracy), reverse=True
    )  # Sort based on important values

    obj_list = []
    for val in boost.model_list:
        obj_list.append(val.get_model())

    print("Completed.")

    class_out = result_directory + "/model_list.json"
    out_file = open(class_out, "w")
    json.dump(obj_list, out_file, indent=4)

    end = time.time()
    total = end - start
    t_hours = (total / 60) / 60

    class_out = result_directory + "/time.json"
    out_file = open(class_out, "w")
    json.dump(t_hours, out_file)
    print(t_hours)


def pre_processed():
    parent = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/optimized_all/"
    all_models = False
    model_list, default_list, optimal_list = [], [], []

    with open(
        "/Volumes/SA Hirsch/Florida Tech/research/sorted_csv_list_mine.json"
    ) as json_file:
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

    dict_of_model_direct = {}
    for outer in path_dict:
        parent_dir = parent + "/" + outer
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
            first_split = path.split("/")
            useful = first_split[-1]
            useful_list = useful.split("_")
            for i in range(len(useful_list)):
                if useful_list[i] == "MZD":
                    zd_mass = useful_list[i + 1]
                    fd1_mass = useful_list[i + 2].partition(".")[0]
                    default, optimal_parameters = optimal_default_models(
                        pd.read_csv(path_dict[outer][inner]),
                        zd_mass,
                        fd1_mass,
                        child_path,
                    )
                    model_list.append(default)
                    model_list.append(optimal_parameters)
                    default_list.append(default)
                    optimal_list.append(optimal_parameters)
        dict_of_model_direct[outer] = inner_dict

    # after running the datasets, retrieve the values stored in the list
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


def optimal_default_models(final_array, zd_mass, fd1_mass, result_dir, optimal=True):
    warnings.filterwarnings("ignore")
    start = time.time()
    filename = "MZD_%s_%s" % (zd_mass, fd1_mass)
    boost = new_xgb("mc", filename)
    old_boost = xgb("mc")
    train_x, test_x, train_y, test_y = boost.split(final_array)

    """
        Arrays to store the values of the hyperparameters that are run with the model. Need to look into
        using a more efficient testing algorithm. Bayesian hyperparameter tuning?

        Removed gblinear for booster parameter, inefficient and complications with xgboost, data for
        default stored in archive file.
    """

    hyper_parameters = {
        "alpha": [0, 1, 2, 3, 4, 5],
        "lambda": [0, 1, 2, 3, 4, 5],
        "eta": [0.6, 0.5, 0.4, 0.3, 0.1],
        "max": [3, 6, 10, 12, 15],
        "objective": ["binary:logistic", "binary:hinge", "reg:squarederror"],
    }

    if optimal:
        # Run with the default parameters
        default = old_boost.xgb(
            zd_mass=zd_mass,
            fd1_mass=fd1_mass,
            result_dir=result_dir,
            train_x=train_x,
            test_x=test_x,
            train_y=train_y,
            test_y=test_y,
            eta=0.3,
            max_depth=6,
            reg_lambda=1,
            reg_alpha=0,
            objective="reg:squarederror",
            single_pair=True,
            ret=True,
        )

        # Run with the optimal parameters
        optimal_parameters = old_boost.xgb(
            zd_mass=zd_mass,
            fd1_mass=fd1_mass,
            result_dir=result_dir,
            train_x=train_x,
            test_x=test_x,
            train_y=train_y,
            test_y=test_y,
            eta=0.6,
            max_depth=6,
            reg_lambda=2,
            reg_alpha=4,
            objective="binary:logistic",
            single_pair=True,
            ret=True,
        )
        return default, optimal_parameters


def run_on_all():
    all_models = False
    zd_mass, fd1_mass = None, None
    pre_processed_csv = (
        "/Volumes/SA Hirsch/Florida Tech/research/sorted_csv_list_mine.json"
    )

    """
        Input the path to the csv file that contains all of the preprocessed data csv lists.
    """
    with open(pre_processed_csv) as json_file:
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
                if int(last[0]) > 55:  #
                    for val in path_list:  #
                        if "55" in val[1]:  #
                            last = val  #
                last_dict = {last[0]: last[1]}
                inner.update(first_dict)
                inner.update(last_dict)
                temp_path[outer] = inner
            else:  #
                inner = {}
                path_list = list(path_dict[outer].items())
                first = path_list[0]
                first_dict = {first[0]: first[1]}
                inner.update(first_dict)
                temp_path[outer] = inner

        path_dict = temp_path

    parent = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/optimized_all/"

    class_out = parent + "paths.json"
    out_file = open(class_out, "w")
    json.dump(path_dict, out_file)

    dict_of_model_direct = {}
    all_models_list = []
    overall_time = time.time()
    for outer in path_dict:
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
            first_split = path.split("/")
            useful = first_split[-1]
            useful_list = useful.split("_")
            for i in range(len(useful_list)):
                if useful_list[i] == "MZD":
                    zd_mass = useful_list[i + 1]
                    fd1_mass = useful_list[i + 2].partition(".")[0]

            data = pd.read_csv(path_dict[outer][inner])
            warnings.filterwarnings("ignore")
            filename = "MZD_%s_%s" % (zd_mass, fd1_mass)
            boost = new_xgb("mc", filename)
            old_boost = xgb("mc")
            train_x, test_x, train_y, test_y = boost.split(data)

            hyper_parameters = {
                "alpha": [0, 1, 2, 3, 4, 5],
                "lambda": [0, 1, 2, 3, 4, 5],
                "eta": [0.6, 0.5, 0.4, 0.3, 0.1],
                "max": [3, 6, 10, 12, 15],
                "objective": ["binary:logistic", "binary:hinge", "reg:squarederror"],
            }
            model_list = []
            sample_start = time.time()
            for val_eta in tqdm(hyper_parameters["eta"]):
                for val_alpha in hyper_parameters["alpha"]:
                    for val_lambda in hyper_parameters["lambda"]:
                        for val_obj in hyper_parameters["objective"]:
                            for val_max_depth in hyper_parameters["max"]:
                                model_object = old_boost.xgb(
                                    zd_mass,
                                    fd1_mass,
                                    child_path,
                                    train_x,
                                    test_x,
                                    train_y,
                                    test_y,
                                    single_pair=True,
                                    ret=True,
                                    eta=val_eta,
                                    max_depth=val_max_depth,
                                    reg_lambda=val_lambda,
                                    reg_alpha=val_alpha,
                                    objective=val_obj,
                                )
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


"""
    Draw the outputted tree that is constructed from the model. Useful to better help visualize how the tree
    is making its decisions. 
"""


def draw_tree():
    """
    Construct the tree based off the supplied root file.
    """
    root_file = (
        "/Users/spencerhirsch/Documents/research/root_files/MZD_200_ALL/MZD_200_55.root"
    )

    """
        Give directory of list of models that the tree is to be constructed from.
    """

    directory = (
        "/Volumes/SA Hirsch/Florida Tech/research/dataframes/archive/data_102522_346PM/"
        "MZD_200_55_pd_model/model_list.json"
    )
    root_dir = "cutFlowAnalyzerPXBL4PXFL3;1/Events;1"
    final_array = process(root_file, root_dir)
    boost = xgb("mc")
    boost.split(final_array)

    f = open(directory)
    data = json.load(f)
    data = sorted(data, key=lambda x: x["mcc"], reverse=True)
    pprint.pprint(data)

    value = data[0]

    val_eta = value["eta"]
    val_max_depth = value["max depth"]
    val_alpha = value["l1"]
    val_lambda = value["l2"]
    val_obj = value["objective"]
    zd_mass, fd_mass, result_directory = find_masses(root_file, True)

    """
        Construct most effective model, and build the tree based on those values.
    """
    _ = boost.xgb(
        zd_mass=zd_mass,
        fd1_mass=fd_mass,
        result_dir=result_directory,
        train_x=pd.DataFrame(),
        test_x=pd.DataFrame(),
        train_y=pd.DataFrame(),
        test_y=pd.DataFrame(),
        eta=val_eta,
        max_depth=val_max_depth,
        reg_lambda=val_lambda,
        reg_alpha=val_alpha,
        objective=val_obj,
        single_pair=True,
        ret=True,
    )


"""
Given the preprocessed data csv file, perform a grid search on the dataset while pulling the various features from the
model. Once there is significant fall of in the effectiveness of the model. Terminate the process and return all results.
"""


def feature_extraction():
    file = "/Volumes/SA Hirsch/Florida Tech/research/dataframe_csv_fD_model/total_df_MZD_200_55.csv"
    result = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/optimized_all/"
    data = pd.read_csv(file)
    columns = [
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
        "pair",
    ]

    zd_mass = 200
    fd1_mass = 55

    warnings.filterwarnings("ignore")
    start = time.time()
    filename = "MZD_%s_%s" % (zd_mass, fd1_mass)
    boost = new_xgb("mc", filename)
    old_boost = xgb("mc")
    train_x, test_x, train_y, test_y = boost.split(data)

    hyper_parameters = {
        "alpha": [0, 1, 2, 3, 4, 5],
        "lambda": [0, 1, 2, 3, 4, 5],
        "eta": [0.6, 0.5, 0.4, 0.3, 0.1],
        "max": [3, 6, 10, 12, 15],
        "objective": ["binary:logistic", "binary:hinge", "reg:squarederror"],
    }

    model_list = []
    sample_start = time.time()
    for val_eta in tqdm(hyper_parameters["eta"]):
        for val_alpha in hyper_parameters["alpha"]:
            for val_lambda in hyper_parameters["lambda"]:
                for val_obj in hyper_parameters["objective"]:
                    for val_max_depth in hyper_parameters["max"]:
                        model_object = old_boost.xgb(
                            zd_mass,
                            fd1_mass,
                            result,
                            train_x,
                            test_x,
                            train_y,
                            test_y,
                            single_pair=True,
                            ret=True,
                            eta=val_eta,
                            max_depth=val_max_depth,
                            reg_lambda=val_lambda,
                            reg_alpha=val_alpha,
                            objective=val_obj,
                        )
                        model_list.append(model_object)

"""
    Driver function that handles each respective function. Takes input from the standard input stream
    and calls each function for its desired ability specified by the user. Makes the code much cleaner 
    and more concise.
"""


def main():
    print(
        "Which program would you like to use:\
        \n(1): Generate permutations of models \
        \n(2): Generate Heat Map \
        \n(3): Plot data \
        \n(4): Output tree \
        \n(5): Run all models\n"
    )

    choice = input("Choice: ")

    if choice == "1":
        process_single()
    elif choice == "2":
        print("Which metric would you like to plot?\n (1): time\n (2): mcc\n (3): f1")
        input_val = input("Choice: ")
        metric = ""

        if input_val == "1":
            metric = "time"
        elif input_val == "2":
            metric = "mcc"
        elif input_val == "3":
            metric = "f1"
        else:
            print("Invalid input.")

        heat_map(metric)
    elif choice == "3":
        plot_data()
    elif choice == "4":
        draw_tree()
    elif choice == "5":
        print(
            "What do you want to test the models with:\
        \n(1): Run optimal and default hyper-parameters \
        \n(2): Run select samples with all hyper-parameters \
        \n(3): Run on single preprocessed"
        )
        choice = input("Choice: ")
        if choice == "1":
            pre_processed()
        elif choice == "2":
            run_on_all()
        elif choice == "3":
            single_preprocessed()
        else:
            "Invalid input."
    else:
        print("Input invalid.")


main()
