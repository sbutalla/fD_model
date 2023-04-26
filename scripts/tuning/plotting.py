import json
import matplotlib.pyplot as plt
import numpy as np
import os


"""
    Plotting for various hyper-parameters in the model to take care of comparisons for eta and max depth.
    Shows all key metrics on one singular plot by fixing the value of the max depth of the tree.
"""


def plot_data():
    directory = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/archive/data_1021_647PM/model_list.json"
    f = open(directory)
    data = json.load(f)

    eta_array = [0.4, 0.3, 0.1, 0.01, 0.001, 0.0001]
    max_depth_array = [3, 6, 10, 20, 30, 50, 75, 100]

    data = sorted(data, key=lambda x: x["eta"])

    index = 0
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    for i in range(len(max_depth_array)):
        storage = []
        for val in data:
            if val["max depth"] == max_depth_array[index]:
                storage.append(val)

        eta = []
        max_depth = []
        f1 = []
        precision = []
        mcc = []
        accuracy = []
        for val in storage:
            eta.append(val["eta"])
            max_depth.append(val["max depth"])
            f1.append(val["f1"])
            precision.append(val["precision"])
            mcc.append(val["mcc"])
            accuracy.append(val["accuracy"])

        fig, ax = plt.subplots()
        plt.title("Fixed Max Depth of %s" % max_depth_array[index], fontsize=15)
        default_x_ticks = range(len(eta))
        plt.xlabel("Learning Rate (eta)", fontsize=10, loc="right")
        plt.ylabel("Value", fontsize=10, loc="top")
        ax.grid()
        ax.plot(eta, f1, label="f1 Score", marker="D", linewidth=1)
        ax.plot(eta, mcc, label="mcc", marker="D", linewidth=1)
        ax.plot(eta, precision, label="precision", marker="D", linewidth=1)
        ax.plot(eta, accuracy, label="accuracy", marker="D", linewidth=1)
        ax.legend(loc="lower right", prop={"size": 12})

        fig.canvas.draw()
        plt.show()
        path = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/plots"

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        fig.savefig(path + "/%s_max_depth_comparison.pdf" % max_depth_array[index])

        index += 1


"""
    Function used to generate heat maps. Will generates the maps for f1-score, matthew correlation
    coefficient (mcc) and execution time. Genearated heat maps show the values against the 
    learning rate (eta) and the max depth of the model. Better helps clearer show the data that
    was collected.
"""


def heat_map(metric, l1, l2):
    # Declaration and initialization of vmin and vmax.
    vmin = 0
    vmax = 1
    if metric == "f1":
        vmin = 0.8
    elif metric == "mcc":
        vmin = 0.91
        vmax = 0.948
    elif metric == "time":
        vmin = 1
        vmax = 11

    # Directory the stores the object list json file.
    #dir = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/archive/data_102822_843AM/model_list.json"
    #dir = "/Users/spencerhirsch/Documents/research/output.json"
    dir = '/Users/spencerhirsch/Documents/research/isolated_jsons/%s_%s_models.json' % (l1, l2)
    f = open(dir)
    data = json.load(f)

    # max_depth_array = [3, 6, 10, 12, 15]
    # eta_array = [0.1, 0.3, 0.4, 0.5, 0.6]
    value_array = []

    # Sort dict based on learning rate in increasing order
    data = sorted(data, key=lambda x: x["eta"], reverse=False)
    index = 0

    eta_array = []
    max_depth_array = []
    for val in data:
        if val["eta"] not in eta_array:
            eta_array.append(val["eta"])

        if val["max depth"] not in max_depth_array:
            max_depth_array.append(val["max depth"])

    eta_array.sort()
    max_depth_array.sort()

    """
        Iterate through the dictionary finding values of the same maximum depth to group the data.
        Store the dictionary values in a temporary list. Append the temporary list to the list to 
        contain all lists. This will create a 2d Array to be plotted.
    """
    for i in range(len(max_depth_array)):
        storage = []
        temp_value_array = []
        data = sorted(data, key=lambda x: x["eta"])
        for val in data:
            if val["max depth"] == max_depth_array[index]:
                storage.append(val)

        for val in storage:
            temp_value_array.append(val["%s" % metric])
        value_array.append(temp_value_array)
        index += 1

    value_array.reverse()  # Reverse the array storing all of the values
    value_array = np.array(value_array)  # Convert to numpy array

    plt.rcParams.update({"font.size": 55})  # was 14I ncrease font size for plotting
    fig, ax = plt.subplots(figsize=(40, 20))  # was 40, 4 Initialize plot
    im = ax.imshow(value_array, vmin=vmin, vmax=vmax)
    ax.set_xlabel(r"Learning rate ($\eta$)", loc="right")
    ax.set_ylabel("Max depth", loc="top")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("%s" % metric.capitalize(), rotation=-90, va="bottom")

    max_depth_array.reverse()  # Reverse the y-axis to increasing order.

    ax.set_xticks(np.arange(len(eta_array)), labels=eta_array)
    ax.set_yticks(np.arange(len(max_depth_array)), labels=max_depth_array)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.title("L1 = %s and L2 = %s" % (l1, l2))

    """
        Iterate through the values in the numpy array and assign the value to a section.
        In the case of the time plot check to see if the time is greater than a specific
        value and convert the color of the text on the heatmap to black.
    """

    for i in range(len(max_depth_array)):
        for j in range(len(eta_array)):
            if metric != "time":
                text = ax.text(
                    j,
                    i,
                    str(value_array[i, j])[:5],
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=40,
                )
            else:
                if float(value_array[i, j]) > 6.4:
                    text = ax.text(
                        j,
                        i,
                        str(value_array[i, j])[:5],
                        ha="center",
                        va="center",
                        color="k",
                        fontsize=10,
                    )
                else:
                    text = ax.text(
                        j,
                        i,
                        str(value_array[i, j])[:5],
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=10,
                    )

    fig.tight_layout()
    # plt.show()

    # Output path for the generated plot
    # path = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/plots"
    path = '/Users/spencerhirsch/Documents/research/isolated_jsons'
    try:
        os.mkdir(path)
    except OSError as error:
        pass

    fig.savefig(path + "/heat_map_%s_%s_%s" % (metric, l1, l2))

    index += 1


def numerous_heatmaps(metric):
    vmin = 0
    vmax = 1
    if metric == "f1":
        vmin = 0.93
        vmax = 0.965
    elif metric == "mcc":
        vmin = 0.91
        vmax = 0.948
    elif metric == "time":
        vmin = 500
        vmax = 1650
    elif metric == "accuracy":
        vmin = 0.96
        vmax = 0.98
    elif metric == "precision":
        vmin = 0.94
        vmax = 0.97
    dir_list = []
    for i in range(5, -1, -1):
        for j in range(0, 6):
            dir = '/Users/spencerhirsch/Documents/research/isolated_jsons/%s_%s_models.json' % (i, j)
            dir_list.append(dir)

    # dir_list.reverse()
    print(dir_list)
    fontsize = 35
    # fig, ax = plt.subplots(6, 6, gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
    fig, ax = plt.subplots(6, 6, figsize=(50, 50))
    # fig.tight_layout(pad=5.0)
    fig.tight_layout(pad=3.0)
    for direct in dir_list:
        f = open(direct)
        data = json.load(f)
        value_array = []

        # Sort dict based on learning rate in increasing order
        data = sorted(data, key=lambda x: x["eta"], reverse=False)
        index = 0

        eta_array = []
        max_depth_array = []
        for val in data:
            if val["eta"] not in eta_array:
                eta_array.append(val["eta"])

            if val["max depth"] not in max_depth_array:
                max_depth_array.append(val["max depth"])

        eta_array.sort()
        max_depth_array.sort()

        for i in range(len(max_depth_array)):
            storage = []
            temp_value_array = []
            data = sorted(data, key=lambda x: x["eta"])
            for val in data:
                if val["max depth"] == max_depth_array[index]:
                    storage.append(val)

            for val in storage:
                temp_value_array.append(val["%s" % metric])
            value_array.append(temp_value_array)
            index += 1

        value_array.reverse()  # Reverse the array storing all of the values
        value_array = np.array(value_array)  # Convert to numpy array

        plt.rcParams.update({"font.size": fontsize})  # was 14I ncrease font size for plotting
        # fig, ax = plt.subplots(figsize=(40, 20))  # was 40, 4 Initialize plot
        split = direct.split('/')
        split = split[len(split) - 1].split('_')
        l1 = int(split[0])
        l2 = int(split[1])
        print(l1)
        print(l2)
        xpos = abs(l1-5)

        im = ax[xpos, l2].imshow(value_array, vmin=vmin, vmax=vmax)
        ax[xpos, l2].set_xlabel(r"Learning rate ($\eta$)", loc="right",fontsize=fontsize)
        ax[xpos, l2].set_ylabel("Max depth", loc="top", fontsize=fontsize)
        ax[xpos, l2].set_title("L1=%s and L2=%s" % (l1, l2))
        max_depth_array.reverse()  # Reverse the y-axis to increasing order.
        ax[xpos, l2].set_xticks(np.arange(len(eta_array)), labels=eta_array, fontsize=fontsize, minor=False)
        ax[xpos, l2].set_yticks(np.arange(len(max_depth_array)), labels=max_depth_array, fontsize=fontsize, minor=False)
        plt.setp(ax[xpos, l2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # for i in range(len(max_depth_array)):
        #     for j in range(len(eta_array)):
        #         if metric != "time":
        #             text = ax[xpos, l2].text(
        #                 j,
        #                 i,
        #                 str(value_array[i, j])[:5],
        #                 ha="center",
        #                 va="center",
        #                 color="w",
        #                 fontsize=6,
        #             )
        #         else:
        #             if float(value_array[i, j]) > 6.4:
        #                 text = ax[l1, l2].text(
        #                     j,
        #                     i,
        #                     str(value_array[i, j])[:5],
        #                     ha="center",
        #                     va="center",
        #                     color="k",
        #                     fontsize=10,
        #                 )
        #             else:
        #                 text = ax[l1, l2].text(
        #                     j,
        #                     i,
        #                     str(value_array[i, j])[:5],
        #                     ha="center",
        #                     va="center",
        #                     color="w",
        #                     fontsize=10,
        #                 )

    # l1_list = [0, 1, 2, 3, 4, 5]
    # l2_list = [0, 1, 2, 3, 4, 5]
    # for l1 in l1_list:
    #     fig.text(-0.03, l1, r'$L_{1} = %s$' % l1, fontsize=44)
    # for l2 in l1_list:
    #     fig.text(l2, -0.01, r'$L_{2} = %s$' % l2, fontsize=44)
    plt.rcParams.update({"font.size": 100})  # was 14I ncrease font size for plotting
    fig.colorbar(im, ax=ax.ravel().tolist(), label=("%s (Seconds)" % metric.capitalize()))
    # cbar.ax.tick_params(labelsize=50)
    # plt.subplot_tool()
    # plt.show()
    fig.savefig("/Users/spencerhirsch/Desktop/heat_map_%s.png" % metric)
    fig.savefig("/Users/spencerhirsch/Documents/research/heatmap/heat_map_%s.png" % metric)
