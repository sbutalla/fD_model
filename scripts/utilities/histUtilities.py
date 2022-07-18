import numpy as np
import seaborn as sn
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import NullFormatter
from matplotlib.colors import LogNorm
import sys
import os

global numBins
global hist_range
global scale_factors_bkg
global scale_factors_sig
global plot_order
global color_list
global color_sig
global dataDir
global signalDir

numBins           = 17
hist_range        = (0, 60) 
scale_factors_bkg = {
"DY0J"        : 3.4145686E+00,
"DY1J"        : 6.2974242E-01,
"DY2J"        : 3.5720940E-01, 
"qqToZZTo4L"  : 4.1624890E-03,
"DiLept"      : 1.0998854E-01,
"ggHToZZTo4L" : 7.6245783E-04,
"ggToZZTo4mu" : 1.1110732E-04
}
'''
scale_factors_sig = {
"MZD_200_55" : (2.04E+02 * 1.454E-02 * 59.9)/30000,
"MZD_350_30" : (3.01E+01 * 1.200E-02 * 59.9)/30000,
"MZD_200_30" : (2.04E+02 * 1.454E-02 * 59.9)/30000,
"MZD_350_55" : (3.01E+01 * 1.200E-02 * 59.9)/30000,
"MZD_400_15" : (1.89E+01 * 1.200E-02 * 59.9)/30000,
"MZD_190_55" : 1,
"MZD_250_55" : (9.43E+01 * 1.200E-02 * 59.9)/30000,
"MZD_180_30" : (2.95E+02 * 1.441E-02 * 59.9)/30000,
"MZD_300_30" : (5.12E+01 * 1.200E-02 * 59.9)/30000,
"MZD_250_30" : (9.43E+01 * 1.200E-02 * 59.9)/30000,
"MZD_190_30" : 1,
"MZD_300_55" : (3.01E+01 * 1.200E-02 * 59.9)/30000,
"MZD_180_55" : (2.95E+02 * 1.133E-02 * 59.9)/30000,
"MZD_190_15" : 1,
"MZD_95_25"  : 1,
"MZD_250_15" : (1.02E+03 * 1.200E-02 * 59.9)/30000,
"MZD_85_25"  : 1,
"MZD_300_15" : (5.12E+01 * 1.200E-02 * 59.9)/30000,
"MZD_180_15" : (2.95E+02 * 1.651E-02 * 59.9)/30000,
"MZD_400_55" : (1.89E+01 * 1.200E-02 * 59.9)/30000,
"MZD_95_15"  : 1,
"MZD_85_15"  : 1,
"MZD_400_30" : (1.89E+01 * 1.200E-02 * 59.9)/30000,
"MZD_200_15" : 1,
"MZD_350_15" : (3.01E+01 * 1.200E-02 * 59.9)/30000,
"MZD_160_15" : 1,
"MZD_170_15" : (3.62E+02 * 1.649E-02 * 59.9)/30000,
"MZD_150_15" : (5.71E+02 * 1.644E-02 * 57.9)/30000,
"MZD_85_35"  : 1,
"MZD_125_15" : (1.23E+03 * 3.259E-02 * 59.9)/30000,
"MZD_150_55" : (5.71E+02 * 9.33E-03  * 59.9)/30000,
"MZD_125_30" : (1.23E+03 * 1.354E-02 * 59.9)/30000,
"MZD_150_30" : (5.71E+02 * 1.407E-02 * 57.9)/30000,
"MZD_125_55" : (1.23E+03 * 4.960E-03 * 59.9)/30000,
"MZD_160_55" : 1,
"MZD_95_45"  : (2.39E+04 * 2.012E-03 * 59.9)/10000,
"MZD_170_30" : 1,
"MZD_160_30" : 1,
"MZD_170_55" : (3.62E+02 * 1.085E-02 * 59.9)/30000
}
'''
scale_factors_sig = {
 'MZD_200_55': 0.0059224328,
 'MZD_350_30': 0.000721196,
 'MZD_200_30': 0.0059224328,
 'MZD_350_55': 0.000721196,
 'MZD_400_15': 0.00045284399999999994,
 'MZD_190_55': 0.005704955866666667,
 'MZD_250_55': 0.002259428,
 'MZD_180_30': 0.008487730166666665,
 'MZD_300_30': 0.001226752,
 'MZD_250_30': 0.002259428,
 'MZD_190_30': 0.0070544629333333326,
 'MZD_300_55': 0.000721196,
 'MZD_180_55': 0.006673558833333334,
 'MZD_190_15': 0.0080531956,
 'MZD_95_25': 0.6384980599999999,
 'MZD_250_15': 0.0244392,
 'MZD_85_25': 0.03688442333333333,
 'MZD_300_15': 0.001226752,
 'MZD_180_15': 0.009724665166666667,
 'MZD_400_55': 0.00045284399999999994,
 'MZD_95_15': 0.7649569433333333,
 'MZD_85_15': 0.04597524666666666,
 'MZD_400_30': 0.00045284399999999994,
 'MZD_200_15': 0.0067370728,
 'MZD_350_15': 0.000721196,
 'MZD_160_15': 0.014732524799999999,
 'MZD_170_15': 0.011918862066666666,
 'MZD_150_15': 0.018117373200000002,
 'MZD_85_35': 0.021519773833333332,
 'MZD_125_15': 0.08003778100000002,
 'MZD_150_55': 0.0106371019,
 'MZD_125_30': 0.033252885999999995,
 'MZD_150_30': 0.0155055621,
 'MZD_125_55': 0.012181263999999999,
 'MZD_160_55': 0.009141858133333333,
 'MZD_95_45': 0.28803993199999994,
 'MZD_170_30': 0.010350400533333333,
 'MZD_160_30': 0.012710939733333333,
 'MZD_170_55': 0.007842307666666666
}
plot_order       = ["DY0J", "DY1J", "DY2J", "qqToZZTo4L", "DiLept", "ggHToZZTo4L", "ggToZZTo4mu"]
color_list       = {
"DY0J"        : "darkkhaki", 
"DY1J"        : "mediumaquamarine",
"DY2J"        : "lightslategrey",
"qqToZZTo4L"  : "slateblue",
"DiLept"      : "mediumseagreen",
"ggHToZZTo4L" : "cyan",
"ggToZZTo4mu" : "magenta"}

dataDir   = "dataframes/"

plt.rcParams.update({'font.size': 26}) # Increase font size for plots

class colors:
    WHITE   = '\033[97m'
    CYAN    = '\033[96m'
    MAGENTA = '\033[95m'
    BLUE    = '\033[94m'
    YELLOW  = '\033[93m'
    GREEN   = '\033[92m'
    RED     = '\033[91m'
    ORANGE  = '\033[38;5;208m'
    ENDC    = '\033[39m'

class histogram:
    def __init__(self, dataset, sigDir = None):
        if dataset not in ["sig", "bkg"] or dataset == None:
            raise ValueError("Dataset must be specified ('sig' or 'bkg').\nExiting...")
            sys.exit()
            
        self.dataset = dataset

        if self.dataset == "sig":
            if sigDir is None:
                print(colors.YELLOW + "A list of signal directories is required (without the preceding parent directories, e.g., sigDir = [MZD_150_55, MZD_95_15]). Exiting...\n" + colors.ENDC)
            elif sigDir is not None:
                self.sig_dir = sigDir
        
    def import_csv(self, csv_type):
        if self.dataset == "bkg":
            if csv_type == "single":
                single_correct_pair_all = {} # initialize dictionary
                for ii in range(len(plot_order)):
                    temp_file = dataDir + plot_order[ii] + "/single_correct_pair_%s.csv" % plot_order[ii]
                    single_correct_pair_all[plot_order[ii]] = pd.read_csv(temp_file)

                self.from_csv_single = True
                return single_correct_pair_all
        elif self.dataset == "sig":
            if csv_type == "single":
                single_correct_pair_all = {} # initialize dictionary
                for dataset in range(len(self.sig_dir)):
                    temp_file = dataDir + self.sig_dir[dataset] + "/single_correct_pair_%s.csv" % self.sig_dir[dataset]
                    single_correct_pair_all[self.sig_dir[dataset]] = pd.read_csv(temp_file)

                self.from_csv_single = True
                return single_correct_pair_all
                 
    def import_correctPairs(self, dictionary):
        if self.dataset == "bkg":
            keyList        = list(dictionary.keys()) 
            self.diMuonKey = []
            for pair in range(2):
                self.diMuonKey.append("invMassA%i" % pair)
                
            self.invMass_bkg = {}
            for key in keyList:
                self.invMass_bkg[key] = {self.diMuonKey[0]: None, self.diMuonKey[1]: None}
                for pair in range(2):
                    self.invMass_bkg[key][self.diMuonKey[pair]] = dictionary[key][self.diMuonKey[pair]].to_numpy()

        elif self.dataset == "sig":
            keyList        = list(dictionary.keys())
            self.diMuonKey = []
            for pair in range(2):
                self.diMuonKey.append("invMassA%i" % pair)

            self.invMass_sig = {}
            for key in self.sig_dir:
                self.invMass_sig[key] = {self.diMuonKey[0]: None, self.diMuonKey[1]: None}
                for pair in range(2):
                    self.invMass_sig[key][self.diMuonKey[pair]] = dictionary[key][self.diMuonKey[pair]].to_numpy()

    def create_hist(self, sf = True, verbose = False):
        if self.dataset == "bkg":
            self.histData_bkg = {}

            for key in plot_order:
                self.histData_bkg[key] = {self.diMuonKey[0]: None, self.diMuonKey[1]: None}

                for pair in self.diMuonKey:
                    temp_hist, temp_bins = np.histogram(self.invMass_bkg[key][pair], bins = numBins, range = hist_range, weights = None, density = None)

                    self.histData_bkg[key][pair] = {"hist": temp_hist, "bins": temp_bins}

            if sf:
                for key in plot_order:
                    for pair in self.diMuonKey:
                        if verbose:
                            print("Applying scale factor of %f to %s\n\n" % (scale_factors_bkg[key], key))

                        self.histData_bkg[key][pair]["hist"] = self.histData_bkg[key][pair]["hist"] * scale_factors_bkg[key]
        elif self.dataset == "sig":
            self.histData_sig = {}
            
            for key in self.sig_dir:
                self.histData_sig[key] = {self.diMuonKey[0]: None, self.diMuonKey[1]: None}
                for pair in self.diMuonKey:
                    temp_hist, temp_bins = np.histogram(self.invMass_sig[key][pair], bins = numBins, range = hist_range, weights = None, density = None)

                    self.histData_sig[key][pair] = {"hist": temp_hist, "bins": temp_bins}

                    if sf:
                        if verbose:
                            print("Applying scale factor of %f to %s\n\n" % (scale_factors_sig[key], "signal data"))

                        self.histData_sig[key][pair]["hist"] = self.histData_sig[key][pair]["hist"] * scale_factors_sig[key]

    def create_hist_2d(self, sf = True, verbose = False):
        if self.dataset == "bkg":
            self.histData_2d_bkg = {}

            if sf:
                self.total_2d_hist_bkg = 0
        
            for key in plot_order:
                temp_hist, temp_x_edge, temp_y_edge = np.histogram2d(self.invMass_bkg[key][self.diMuonKey[0]], self.invMass_bkg[key][self.diMuonKey[1]], bins = numBins)

                self.histData_2d_bkg[key] = {"hist": temp_hist, "x_edge": temp_x_edge, "y_edge": temp_y_edge}

                if sf:
                    if verbose:
                        print("Applying scale factor of %f to %s\n\n" % (scale_factors_bkg[key], key))

                    self.histData_2d_bkg[key]["hist"] = self.histData_2d_bkg[key]["hist"].T * scale_factors_bkg[key]

                    self.total_2d_hist_bkg += self.histData_2d_bkg[key]["hist"]

        elif self.dataset == "sig":
            self.histData_2d_sig = {}
            
            if sf:
                self.total_2d_hist_sig = 0

            for key in self.sig_dir:
                temp_hist, temp_x_edge, temp_y_edge = np.histogram2d(self.invMass_sig[key][self.diMuonKey[0]], self.invMass_sig[key][self.diMuonKey[1]], bins = numBins)

                self.histData_2d_sig[key] = {"hist": temp_hist, "x_edge": temp_x_edge, "y_edge": temp_y_edge}

                if sf:
                    if verbose:
                        print("Applying scale factor of %f to %s\n\n" % (scale_factors_sig[key], key))

                    self.histData_2d_sig[key]["hist"] = self.histData_2d_sig[key]["hist"].T * scale_factors_sig[key]

                    self.total_2d_hist_sig += self.histData_2d_sig[key]["hist"]
                                
    def plotHist(self, save = True, sig_list = None):

        if self.dataset == "bkg":
            mu_cnt = 1 # counter for dimuon pair
            for pair in self.diMuonKey:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.grid(zorder = 0)
                cnt = 0

                for key in plot_order:
                    temp_bottom = 0
                    if cnt == 0:
                        pass
                    else:
                        for ii in range(cnt):
                            temp_bottom += self.histData_bkg[plot_order[ii]][pair]["hist"]

                    if pair == "invMassA0":
                        temp_string = "diMuonC_m_%s" % key
                    elif pair == "invMassA1":
                        temp_string = "diMuonF_m_%s" % key

                    if cnt == 0:
                        plt.bar(self.histData_bkg[key][pair]["bins"][:-1], self.histData_bkg[key][pair]["hist"], width = 3.5, label = temp_string, edgecolor='black', color = color_list[key], zorder = 3)
                    elif cnt >= 1:
                        plt.bar(self.histData_bkg[key][pair]["bins"][:-1], self.histData_bkg[key][pair]["hist"], width = 3.5, bottom = temp_bottom, label = temp_string, edgecolor='black', color = color_list[key], zorder = 3)

                    cnt += 1

                plt.ylabel("Number of Events/3.5 GeV", fontsize = 30, loc = "top")
                plt.xlabel(r'$m_{\mu\mu_{%i}}$ [GeV]' % mu_cnt, fontsize = 30, loc = "right")
                plt.xlim(hist_range)
                plt.ylim(0, 15)

                leg = plt.legend(loc = 'upper left', fontsize = 20)
                leg.get_frame().set_edgecolor('b')
                plt.show()
                
                if save:
                    plt.savefig(dataDir + "all_bkg_hist_diMu_%s.pdf" % pair)

                mu_cnt += 1
        elif self.dataset == "sig":
    
            for key in self.sig_dir:
                mu_cnt = 1 # counter for dimuon pair
                for pair in self.diMuonKey:
                    fig, ax = plt.subplots(figsize = (10, 10))
                    ax.grid(zorder = 0)

                    plt.bar(self.histData_sig[key][pair]["bins"][:-1], self.histData_sig[key][pair]["hist"], width = 3.5, label = "MC Signal Data\n"  + r'$m_{Z_{D}} = $ %s\n, $m_{f_{D1}} = $ %s' % (key.split("_")[1], key.split("_")[2]), edgecolor = 'black', color = "darkred", zorder = 3)

                    plt.ylabel("Number of Events/3.5 GeV", fontsize = 30, loc = "top")
                    plt.xlabel(r'$m_{\mu\mu_{%i}}$ [GeV]' % mu_cnt, fontsize = 30, loc = "right")
                    plt.xlim(hist_range)
                    #plt.ylim(0, 15)

                    leg = plt.legend(loc = 'upper left', fontsize = 20)
                    leg.get_frame().set_edgecolor('b')
                    plt.show()
                    
                    if save:
                        plt.savefig(dataDir + key + "/signal_%s_hist_diMu_%s.pdf" % (key, pair))

                    mu_cnt += 1
        
    def plot2Dhist(self, save = True):
        
        if self.dataset == "bkg":
            fig = plt.figure(figsize = (10, 10))

            temp_key = list(self.histData_2d_bkg.keys())[0] # to index array when setting extent for 2D histogram

            left, bottom, width, height = 0.1, 0.1, 0.65, 0.65
            bottom_h = left_h = left + width + 0.02

            rect_scatter = [left, bottom, width, height]
            rect_histx   = [left, bottom_h, width, 0.3]
            rect_histy   = [left_h, bottom, 0.3, height]

            # add the axes to the figure
            ax2d    = plt.axes(rect_scatter)
            axHistx = plt.axes(rect_histx)
            axHisty = plt.axes(rect_histy)

            # no labels for the sidecar histograms, because the 2D plot has them
            nullfmt = NullFormatter()         
            axHistx.xaxis.set_major_formatter(nullfmt)
            axHisty.yaxis.set_major_formatter(nullfmt)

            # the 2D plot:
            plot2d = ax2d.imshow(self.total_2d_hist_bkg, interpolation = 'none', origin = 'lower',\
            extent = [self.histData_2d_bkg[temp_key]["x_edge"][0], self.histData_2d_bkg[temp_key]["x_edge"][-1], self.histData_2d_bkg[temp_key]["y_edge"][0], self.histData_2d_bkg[temp_key]["y_edge"][-1]], aspect = 'auto', cmap = cm.plasma)
            ax2d.set_xlabel(r"$m_{\mu\mu_{1}}$ [GeV]", loc='right')
            ax2d.set_ylabel(r"$m_{\mu\mu_{2}}$ [GeV]", loc='top')
            plot2d.set_clim(0, 8)
            plt.colorbar(plot2d)

            # the 1-D histograms: first the X-histogram
            xhist = np.sum(self.total_2d_hist_bkg, axis = 1)
            axHistx.grid(zorder = 0)
            axHistx.bar(self.histData_2d_bkg[temp_key]["x_edge"][:-1], xhist, width = 3.5, zorder = 3)

            axHistx.set_xlim(0, 60) # x-limits match the 2D plot
            axHistx.set_ylim(0, 8)
            
            # then the Y-histogram
            yhist = np.sum(self.total_2d_hist_bkg, axis = 0)
            # use barh instead of bar here because we want a horizontal histogram
            axHisty.barh(self.histData_2d_bkg[temp_key]["y_edge"][:-1], yhist, 3.5, zorder = 3)
            axHisty.grid(zorder = 0)
            axHisty.set_ylim(0, 60) # y-limits match the 2D plot
            axHisty.set_xlim(0, 10)
            
            if save:
                plt.savefig(dataDir + "/hist_2D_all_bkg.pdf")
            
        elif self.dataset == "sig":
            for key in self.sig_dir:
                fig = plt.figure(figsize=(10, 10))

                left, bottom, width, height = 0.1, 0.1, 0.65, 0.65
                bottom_h = left_h = left + width + 0.02

                rect_scatter = [left, bottom, width, height]
                rect_histx   = [left, bottom_h, width, 0.3]
                rect_histy   = [left_h, bottom, 0.3, height]

                # add the axes to the figure
                ax2d    = plt.axes(rect_scatter)
                axHistx = plt.axes(rect_histx)
                axHisty = plt.axes(rect_histy)

                # no labels for the sidecar histograms, because the 2D plot has them
                nullfmt = NullFormatter()         
                axHistx.xaxis.set_major_formatter(nullfmt)
                axHisty.yaxis.set_major_formatter(nullfmt)

                # the 2D plot:
                plot2d = ax2d.imshow(self.total_2d_hist_sig[key], interpolation = 'none', origin = 'lower',\
                extent = [self.histData_2d_sig[key]["x_edge"][0], self.histData_2d_sig[key]["x_edge"][-1], self.histData_2d_sig[key]["y_edge"][0], self.histData_2d_sig[key]["y_edge"][-1]], aspect = 'auto', cmap = cm.plasma)
                ax2d.set_xlabel(r"$m_{\mu\mu_{1}}$ [GeV]", loc = 'right')
                ax2d.set_ylabel(r"$m_{\mu\mu_{2}}$ [GeV]", loc = 'top')
                plot2d.set_clim(0, 10)
                plt.colorbar(plot2d)

                # the 1-D histograms: first the X-histogram
                xhist = np.sum(self.total_2d_hist_sig, axis = 1)
                axHistx.grid(zorder = 0)
                axHistx.bar(self.histData_2d_sig["x_edge"][:-1], xhist, width = 3.5, zorder = 3)

                axHistx.set_xlim(0, 60) # x-limits match the 2D plot
                axHistx.set_ylim(0, 65)
                
                # then the Y-histogram
                yhist = np.sum(self.total_2d_hist_sig, axis = 0)
                # use barh instead of bar here because we want a horizontal histogram
                axHisty.barh(self.histData_2d_sig[key]["y_edge"][:-1], yhist, 3.5, zorder = 3)
                axHisty.grid(zorder = 0)
                axHisty.set_ylim(0, 70) # y-limits match the 2D plot
                axHisty.set_xlim(hist_range)
                
                if save:
                    plt.savefig(dataDir + key + "/hist_2D_sig_%s.pdf" % key)

def ratio_hist(hist_2D_total_bkg, hist_2D_total_sig, histData_2d_sig, save = True):
    hist_2d_ratio = hist_2D_total_bkg / np.sqrt(hist_2D_total_sig)
    hist_2D_total_bkg_sq = np.sqrt(hist_2D_total_bkg)
    hist_2D_divide = np.divide(hist_2D_total_sig, hist_2D_total_bkg_sq)
    
    fig = plt.figure(figsize=(10, 10))

    # define where the axes go
    left, width,  bottom, height = 0.1, 0.65, 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx   = [left, bottom_h, width, 0.3]
    rect_histy   = [left_h, bottom, 0.3, height]

    # add the axes to the figure
    ax2d    = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels for the sidecar histograms, because the 2D plot has them
    nullfmt   = NullFormatter()         
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the 2D plot:
    # note the all-important transpose!
    plot2d = ax2d.imshow(hist_2D_divide, interpolation='none', origin='lower',\
    extent=[histData_2d_sig["x_edge"][0], histData_2d_sig["x_edge"][-1], histData_2d_sig["y_edge"][0], histData_2d_sig["y_edge"][-1]],aspect='auto',cmap=cm.plasma)
    ax2d.set_xlabel(r"$m_{\mu\mu_{1}}$ [GeV]",loc='right')
    ax2d.set_ylabel(r"$m_{\mu\mu_{2}}$ [GeV]",loc='top')
    plot2d.set_clim(0,20)
    plt.colorbar(plot2d)

    # the 1-D histograms: first the X-histogram
    xhist = hist_2D_divide.sum(axis=1) # note x-hist is axis 1, not 0

    axHistx.bar(histData_2d_sig["x_edge"][:-1], xhist, width = 3.5, zorder = 3)
    axHistx.grid(zorder = 0)
    axHistx.set_xlim( 0,60) # x-limits match the 2D plot
    axHistx.set_ylim(0,200)


    # then the Y-histogram
    yhist = hist_2D_divide.sum(axis=0) # note y-hist is axis 0, not 1
    # use barh instead of bar here because we want a horizontal histogram
    axHisty.barh(histData_2d_sig["y_edge"][:-1], yhist, 3.5, zorder = 3)
    axHisty.grid(zorder = 0)
    axHisty.set_ylim(0, 60) # y-limits match the 2D plot
    axHisty.set_xlim(0,200)
    
    if save:
        plt.savefig(dataDir + "ratio_hist.pdf")