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

plt.rcParams.update({'font.size': 22}) # Increase font size

global numBins
global hist_range
global scale_factors
global plot_order
global color_list
global dataDir

numBins          = 17
hist_range    = (0, 60) 
scale_factors = {
"DY0J"        : 3.4145686E+00,
"DY1J"        : 6.2974242E-01,
"DY2J"        : 3.5720940E-01, 
"qqToZZTo4L"  : 4.1624890E-03,
"DiLept"      : 1.0998854E-01,
"ggHToZZTo4L" : 7.6245783E-04,
"ggToZZTo4mu" : 1.1110732E-04
}
plot_order = ["DY0J", "DY1J", "DY2J", "qqToZZTo4L", "DiLept", "ggHToZZTo4L", "ggToZZTo4mu"]
color_list = {
"DY0J"        : "darkkhaki", 
"DY1J"        : "mediumaquamarine",
"DY2J"        : "lightslategrey",
"qqToZZTo4L"  : "slateblue",
"DiLept"      : "mediumseagreen",
"ggHToZZTo4L" : "cyan",
"ggToZZTo4mu" : "magenta"}
dataDir = "dataframes/"

class histogram:
    def __init__(self, file_name = None):
        print("Initialized")
        
    def import_correctPairs(self, dictionary, reorder = True):
        keyList = list(dictionary.keys())

        # get number of events for each dataset
        
        self.diMuonKey = []
        for pair in range(2):
            self.diMuonKey.append("invMassA%i" % pair)
                
        self.invMass = {}
        for key in keyList:
            self.invMass[key] = {"invMassA0": None, "invMassA1": None}

            for pair in range(2):
                self.invMass[key][self.diMuonKey[pair]] = dictionary[key]["single"][self.diMuonKey[pair]].to_numpy()

    def create_hist(self, sf = True, verbose = False):
        self.histData = {}
    
        for key in plot_order:
            self.histData[key] = {self.diMuonKey[0]: None, self.diMuonKey[1]: None}

            for pair in self.diMuonKey:
                temp_hist, temp_bins = np.histogram(self.invMass[key][pair], bins = numBins, range = hist_range, weights=None, density=None)

                self.histData[key][pair] = {"hist": temp_hist, "bins": temp_bins}
                
        if sf:
            for key in plot_order:
                for pair in self.diMuonKey:
                    if verbose:
                        print("Applying scale factor of %f to %s\n\n" % (scale_factors[key], key))
                    
                    self.histData[key][pair]["hist"] = self.histData[key][pair]["hist"] * scale_factors[key]

    def plotHist(self, pair = "both"):
        mu_cnt = 1
        for pair in self.diMuonKey:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.grid(zorder = 0)
            cnt        = 0
            
            for key in plot_order:
                #print("key: ", key)
                t_str = ""
                temp_bottom = 0
                if cnt == 0:
                    pass
                else:
                    
                    for ii in range(cnt):
                        temp_bottom += self.histData[plot_order[ii]][pair]["hist"]
                        #t_str += plot_order[ii] + " + "
                
                print(t_str)
                if pair == "invMassA0":
                    temp_string = "diMuonC_m_%s" % key
                elif pair == "invMassA1":
                    temp_string = "diMuonF_m_%s" % key
                    
                if cnt == 0:
                    plt.bar(self.histData[key][pair]["bins"][:-1], self.histData[key][pair]["hist"], width = 3.5, label = temp_string, edgecolor='black', color = color_list[key], zorder = 3)
                elif cnt >= 1:
                    plt.bar(self.histData[key][pair]["bins"][:-1], self.histData[key][pair]["hist"], width = 3.5, bottom = temp_bottom, label = temp_string, edgecolor='black', color = color_list[key], zorder = 3)
                
                cnt += 1
            
            plt.ylabel("Number of Events/3.5 GeV", fontsize = 30, loc = "top")
            plt.xlabel(r'$m_{\mu\mu_{%i}}$ [GeV]' % mu_cnt, fontsize = 30, loc = "right")
            plt.xlim(0, 60)
            plt.ylim(0, 15)
            
            leg = plt.legend(loc = 'upper left')
            leg.get_frame().set_edgecolor('b')
            plt.show()
            plt.savefig(dataDir + "all_bkg_hist_diMu_%s.pdf" % pair)
            
            mu_cnt += 1

   # def plot2Dhist(self):