import matplotlib.pyplot as plt
import os
import json
import numpy as np



'''
    Plotting for various hyper-parameters in the model to take care of comparisons for eta and max depth.
    Shows all key metrics on one singular plot by fixing the value of the max depth of the tree.
'''


def plot_data():
    dir = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/archive/data_1021_647PM/model_list.json'
    f = open(dir)
    data = json.load(f)

    eta_array = [0.4, 0.3, 0.1, 0.01, 0.001, 0.0001]
    max_depth_array = [3, 6, 10, 20, 30, 50, 75, 100]

    data = sorted(data, key=lambda x: x['eta'])

    index = 0
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    for i in range(len(max_depth_array)):
        storage = []
        for val in data:
            if val['max depth'] == max_depth_array[index]:
                storage.append(val)

        eta = []
        max_depth = []
        f1 = []
        precision = []
        mcc = []
        accuracy = []
        for val in storage:
            eta.append(val['eta'])
            max_depth.append(val['max depth'])
            f1.append(val['f1'])
            precision.append(val['precision'])
            mcc.append(val['mcc'])
            accuracy.append(val['accuracy'])

        fig, ax = plt.subplots()
        plt.title('Fixed Max Depth of %s' % max_depth_array[index], fontsize=15)
        default_x_ticks = range(len(eta))
        plt.xlabel('Learning Rate (eta)', fontsize=10, loc='right')
        plt.ylabel('Value', fontsize=10, loc='top')
        ax.grid()
        ax.plot(eta, f1, label='f1 Score', marker='D', linewidth=1)
        ax.plot(eta, mcc, label='mcc', marker='D', linewidth=1)
        ax.plot(eta, precision, label='precision', marker='D', linewidth=1)
        ax.plot(eta, accuracy, label='accuracy', marker='D', linewidth=1)
        ax.legend(loc='lower right', prop={'size': 12})


        fig.canvas.draw()
        plt.show()
        path = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/plots'

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        fig.savefig(path + '/%s_max_depth_comparison.pdf' % max_depth_array[index])

        index += 1


'''
    Function used to generate heat maps. Will generates the maps for f1-score, matthew correlation
    coefficient (mcc) and execution time. Genearated heat maps show the values against the 
    learning rate (eta) and the max depth of the model. Better helps clearer show the data that
    was collected.
'''


def heat_map(metric):
    # Declaration and initialization of vmin and vmax.
    vmin = 0
    vmax = 1
    if metric is 'f1':
        vmin = 0.8
    elif metric is 'mcc':
        vmin = 0.7
    elif metric is 'time':
        vmin = 1
        vmax = 11

    # Directory the stores the object list json file.
    dir = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/archive/data_102822_843AM/model_list.json'

    f = open(dir)
    data = json.load(f)

    # max_depth_array = [3, 6, 10, 12, 15]
    # eta_array = [0.1, 0.3, 0.4, 0.5, 0.6]
    value_array = []

    # Sort dict based on learning rate in increasing order
    data = sorted(data, key=lambda x: x['eta'], reverse=False)
    index = 0

    eta_array = []
    max_depth_array =[]
    for val in data:
        if val['eta'] not in eta_array:
            eta_array.append(val['eta'])

        if val['max depth'] not in max_depth_array:
            max_depth_array.append(val['max depth'])

    eta_array.sort()
    max_depth_array.sort()

    '''
        Iterate through the dictionary finding values of the same maximum depth to group the data.
        Store the dictionary values in a temporary list. Append the temporary list to the list to 
        contain all lists. This will create a 2d Array to be plotted.
    '''
    for i in range(len(max_depth_array)):
        storage = []
        temp_value_array = []
        data = sorted(data, key=lambda x: x['eta'])
        for val in data:
            if val['max depth'] == max_depth_array[index]:
                storage.append(val)

        for val in storage:
            temp_value_array.append(val['%s' % metric])
        value_array.append(temp_value_array)
        index += 1

    value_array.reverse()   # Reverse the array storing all of the values
    # value_array = np.array(value_array, dtype=object)     # Convert to numpy array
    value_array = np.array(value_array)     # Convert to numpy array
    # print(value_array)


    plt.rcParams.update({'font.size': 14})  # Increase font size for plotting
    fig, ax = plt.subplots(figsize=(40, 4)) # Initialize plot
    # im = ax.imshow(value_array)
    im = ax.imshow(value_array, vmin=vmin, vmax=vmax)
    ax.set_xlabel(r'Learning rate ($\eta$)', loc="right")
    ax.set_ylabel('Max depth', loc="top")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('%s' % metric.capitalize(), rotation=-90, va="bottom")

    max_depth_array.reverse()   # Reverse the y-axis to increasing order.

    ax.set_xticks(np.arange(len(eta_array)), labels=eta_array)
    ax.set_yticks(np.arange(len(max_depth_array)), labels=max_depth_array)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    '''
        Iterate through the values in the numpy array and assign the value to a section.
        In the case of the time plot check to see if the time is greater than a specific
        value and convert the color of the text on the heatmap to black.
    '''

    for i in range(len(max_depth_array)):
        for j in range(len(eta_array)):
            if metric is not 'time':
                text = ax.text(j, i, str(value_array[i, j])[:5],
                               ha="center", va="center", color="w", fontsize=8)
            else:
                if float(value_array[i, j]) > 6.4:
                    text = ax.text(j, i, str(value_array[i, j])[:5],
                                   ha="center", va="center", color="k", fontsize=10)
                else:
                    text = ax.text(j, i, str(value_array[i, j])[:5],
                                   ha="center", va="center", color="w", fontsize=10)

    fig.tight_layout()
    plt.show()

    # Output path for the generated plot
    path = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/plots'

    try:
        os.mkdir(path)
    except OSError as error:
        pass

    fig.savefig(path + '/heat_map_%s' % metric)

    index += 1


def numerous_heat_map():
    # Directory the stores the object list json file.
    dir = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/archive/data_102822_843AM/model_list.json'

    f = open(dir)
    data = json.load(f)

    # Sort dict based on learning rate in increasing order
    data = sorted(data, key=lambda x: x['eta'], reverse=False)
    index = 0

    '''
        Collect all potential values in the the dictionaries. This is used as opposed to hard coding
        lists because the values tested have been changed in the past and it will eliminate the problem
        of changing the values.
    '''

    eta_list = list(set(x['eta'] for x in data))
    max_list = list(set(x['max depth'] for x in data))
    l1_list = list(set(x['l1'] for x in data))
    l2_list = list(set(x['l1'] for x in data))
    objective_list = list(set(x['objective'] for x in data))

    eta_list.sort()
    max_list.sort()
    l1_list.sort()
    l1_list.sort()

    objectives = []
    for val_obj in objective_list:
        temp_obj = [x for x in data if x['objective'] == val_obj]
        objectives.append(temp_obj)


    '''
        Create a list that stores all of the data, list of lists of list. Outer most list stores data for the objective
        parameters. The next list will store all values with a fixed l1 value and the last list will store all values
        with the same l2 value. This is the easiest way to store the values for plotting.
    '''

    # fixed = []
    fixed = {}
    for objective in objectives:
        fixed_object = {}
        for val in l1_list:
            tfixed = {}
            for l2_val in l2_list:
                # tfixed = [x for x in objective if x['l1'] == val and x['l2'] == l2_val]
                tfixed['l2_%s' % l2_val] = [x for x in objective if x['l1'] == val and x['l2'] == l2_val]
                # fixed_object.append(tfixed)
                fixed_object[('l1_%s' % val)] = tfixed

        fixed[('%s' % objective[0]['objective'])] = fixed_object

    print()



    '''
        Iterate through the dictionary finding values of the same maximum depth to group the data.
        Store the dictionary values in a temporary list. Append the temporary list to the list to
        contain all lists. This will create a 2d Array to be plotted.
    '''

    '''
        Sort all mcc values based on criteria.
        - 0 1 2 3 4 5   L1 VALUES
        0 - - - - - -
        1 - - - - - -
        2 - - - - - -
        3 - - - - - -
        4 - - - - - -
        5 - - - - - - 
                        * Each empty space represents a heat map that displays the mcc
        L2
        V
        A
        L
        U
        E
        S
        
            data = sorted(data, key=lambda x: x['eta'], reverse=False)

    '''

    print(fixed['binary:hinge']['l1_0']['l2_0'])

    obj = {}
    for objective in objective_list:
        store = {}
        for l1 in l1_list:
            for l2 in l2_list:
                # store = {}
                storage = []
                for i in range(len(max_list)):
                    temp_storage = []
                    fixed['%s' % objective]['l1_%s' % l1]['l2_%s' % l2] = sorted(fixed['%s' % objective]['l1_%s' % l1]
                                                        ['l2_%s' % l2], key=lambda x: x['eta'], reverse=False)
                    for ele in fixed['%s' % objective]['l1_%s' % l1]['l2_%s' % l2]:
                        if ele['max depth'] == max_list[i]:
                            temp_storage.append(ele)

                    storage.append(temp_storage)
                store['l1_%s_&_l2_%s' % (l1, l2)] = storage
        obj.update({'%s' % objective: store})


    '''
        Where all plotting will take place, extract the necessary data. MCC, f1-score, and time. Plot them accordingly.
    '''

    for ele in obj:
        fig, ((ax, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, az9, az10), (az11, az12, az13, az14, ax15),
              (ax16, ax17, ax18, ax19, ax20), (ax21, ax22, ax23, ax24, ax25)) = plt.subplots(2, 2, figsize=(8, 6))

        for ele_1 in ele:
            for l1 in l1_list:
                for l2 in l2_list:
                    store = []
                    temp_list = []
                    for val in obj['%s' % ele]['l1_%s_&_l2_%s' % (l1, l2)]:
                        for item in val:
                            temp_list.append(item['mcc'])
                    store.append(temp_list)

                    store.reverse()
                    store = np.array(store)



                    print(store)






    # for i in range(len(max_depth_array)):
    #     storage = []
    #     temp_value_array = []
    #     data = sorted(data, key=lambda x: x['eta'])
    #     for val in data:
    #         if val['max depth'] == max_depth_array[index]:
    #             storage.append(val)
    #
    #     for val in storage:
    #         temp_value_array.append(val['%s' % metric])
    #     value_array.append(temp_value_array)
    #     index += 1
    #
    # value_array.reverse()  # Reverse the array storing all of the values
    # # value_array = np.array(value_array, dtype=object)     # Convert to numpy array
    # value_array = np.array(value_array)  # Convert to numpy array
    # # print(value_array)
    #
    # plt.rcParams.update({'font.size': 14})  # Increase font size for plotting
    # fig, ax = plt.subplots(figsize=(40, 4))  # Initialize plot
    # # im = ax.imshow(value_array)
    # im = ax.imshow(value_array, vmin=vmin, vmax=vmax)
    # ax.set_xlabel(r'Learning rate ($\eta$)', loc="right")
    # ax.set_ylabel('Max depth', loc="top")
    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel('%s' % metric.capitalize(), rotation=-90, va="bottom")
    #
    # max_depth_array.reverse()  # Reverse the y-axis to increasing order.
    #
    # ax.set_xticks(np.arange(len(eta_array)), labels=eta_array)
    # ax.set_yticks(np.arange(len(max_depth_array)), labels=max_depth_array)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")


# numerous_heat_map()
