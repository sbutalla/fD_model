from scipy import stats
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

global points
global numBins
global hist_range
global scale_factors_bkg
global scale_factor_sig
global plot_order
global color_list
global color_sig
global dataDir
global signalDir

points            = [0, 60, 100]
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
scale_factor_sig = (2.04E+02 * 57.9)/30000
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
signalDir = "DataAboveUpsilonCRSR_MZD_200_55_signal"

plt.rcParams.update({'font.size': 24}) # Increase font size for plots

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

class kde:
    def __init__(self, dataset, sig_dirs = None):
        if dataset not in ["sig", "bkg"] or dataset == None:
            raise ValueError("Dataset must be specified (dataset = 'sig' or 'bkg').\nExiting...")
            sys.exit()
        else:
            self.dataset = dataset
            
        if dataset == "sig" and sig_dirs is None:
            print("The keys for each signal dataset must be specified in a list, e.g., [MZD_150_15, MZD_200_20]. Exiting...\n")
            sys.exit()
        elif dataset == "sig" and sig_dirs is not None: 
            self.sig_dir = sig_dirs
        
    def calc_kde_1D(self, invMass, kdeMethod = "scipy", kernel_type = "gaussian", bw = None): 
        self.invMass = invMass
        del invMass
                
        if kdeMethod == "sklearn" and bw is None:
            print("The bandwidth ('bw') should be provided when executing the function call. Setting the BW to the default of [1.99]. One specify more than one bandwidth in list format.")
            self.bw = [1.99]
        else:
            self.bw = bw

        if self.dataset == "bkg":
            self.pdf_bkg_1D = {} # initialize empty dict for pdfs
            self.numEvents  = {} # initialize empty dict for number of events
            keys_bkg        = list(self.invMass.keys())
            self.diMuonKey  = list(self.invMass[keys_bkg[0]].keys()) # extract keys for invMass pairs
            
            if kdeMethod == "scipy":
                self.kde_method = "scipy"
                for key in plot_order_kde:
                    self.pdf_bkg_1D[key] = {self.diMuonKey[0]: None, self.diMuonKey[1]: None} # create nested dict for both dimuon pairs
                    self.numEvents[key]  = len(self.invMass[key][self.diMuonKey[0]])
                    for pair in self.diMuonKey:
                        gkde = stats.gaussian_kde(self.invMass[key][pair])
                        self.pdf_bkg_1D[key][pair] = gkde.evaluate(ind)
                        
            elif kdeMethod == "pd":
                self.kde_method = "pd"
                print("KDE performed when calling plot_kde_1D()")
                        
            elif kdeMethod == "sklearn":
                self.kde_method     = "sklearn"
                self.invMass_shaped = {}
                self.vals           = np.linspace(0, 60, 100)
                for key in plot_order_kde:
                    self.invMass_shaped[key] = {self.diMuonKey[0]: None, self.diMuonKey[1]: None} # create nested dict for both dimuon pairs
                    self.pdf_bkg_1D[key]     = {self.diMuonKey[0]: None, self.diMuonKey[1]: None}
                    self.numEvents[key]      = len(self.invMass[key][self.diMuonKey[0]])
                    for pair in self.diMuonKey:
                        for bandwidth in self.bw:
                            self.invMass_shaped[key][pair]   = self.invMass[key][pair].reshape(-1, 1)
                            temp_kde  = KernelDensity(kernel = kernel_type, bandwidth = bandwidth).fit(self.invMass_shaped[key][pair]) # kde object
                            temp_prob = temp_kde.score_samples(self.vals.reshape(-1,1))
                            self.pdf_bkg_1D[key][pair] = np.exp(temp_prob)

        elif self.dataset == "sig":
            self.pdf_sig_1D = {} # initialize empty dict for pdfs
            self.numEvents  = {} # initialize empty dict for number of events
            self.keys_sig   = list(self.invMass.keys())
            self.diMuonKey  = list(self.invMass[self.keys_sig[0]].keys()) # extract keys for invMass pairs
            
            if kdeMethod == "scipy":
                self.kde_method = "scipy"
                for key in self.sig_dir:
                    self.pdf_sig_1D[key] = {self.diMuonKey[0]: None, self.diMuonKey[1]: None}
                    self.numEvents[key]  = len(self.invMass[key][self.diMuonKey[0]])
                    for pair in self.diMuonKey:
                        gkde = stats.gaussian_kde(self.invMass[key][pair])
                        self.pdf_sig_1D[key][pair] = gkde.evaluate(ind)
            
            elif kdeMethod == "pd":
                self.kde_method = "pd"
                print("KDE performed when calling plot_kde_1D()")
                        
            elif kdeMethod == "sklearn":
                self.kde_method     = "sklearn"
                self.invMass_shaped = {}
                self.vals           = np.linspace(0, 60, 100)
                for key in plot_order_kde:
                    self.invMass_shaped[key] = {self.diMuonKey[0]: None, self.diMuonKey[1]: None} # create nested dict for both dimuon pairs
                    self.pdf_sig_1D[key]     = {self.diMuonKey[0]: None, self.diMuonKey[1]: None}
                    self.numEvents[key]      = len(self.invMass[key][self.diMuonKey[0]])
                    for pair in self.diMuonKey:
                        for bandwidth in self.bw:
                            self.invMass_shaped[key][pair]   = self.invMass[key][pair].reshape(-1, 1)
                            temp_kde  = KernelDensity(kernel = kernel_type, bandwidth = bandwidth).fit(self.invMass_shaped[key][pair]) # kde object
                            temp_prob = temp_kde.score_samples(self.vals.reshape(-1,1))
                            self.pdf_sig_1D[key][pair] = np.exp(temp_prob)
    
    def plot_kde_1D(self, save = True, scott = True, bw = None):
        if self.dataset == "bkg":
            if self.kde_method == "scipy":
                for key in plot_order_kde:
                    fig, ax = plt.subplots(figsize = (20, 10), nrows = 1, ncols = 2)
                    ax = ax.ravel() # unravel ax object to easily index 
                    cnt = 0
                    for pair in self.diMuonKey:
                        label_name = key + "_" + pair[-2:]
                        ax[cnt].hist(self.invMass[key][pair], bins = 100, range = (0, 60), label = label_name, facecolor = hist_facecolor, edgecolor = hist_edgecolor, alpha = 0.7, density = 1)
                        ax[cnt].plot(ind, self.pdf_bkg_1D[key][pair], color = kde_color, linewidth = 3, label = 'BW: Scott')
                        ax[cnt].set_xlim(hist_range)
                        ax[cnt].set_xlabel(r"$m_{\mu\mu_{%d}}$ [GeV]" % (cnt + 1), loc = 'right')
                        ax[cnt].set_ylabel('Probability', loc = "top")
                        ax[cnt].grid()
                        ax[cnt].grid(which = 'minor', alpha = 0.2)
                        ax[cnt].grid(which = 'major', alpha = 0.5)
                        ax[cnt].legend(loc = 'upper left', fontsize = 18, shadow = True, borderpad = 1)
                        
                        cnt += 1
                
                    plt.show()    
                    if save:
                        fig.savefig(dataDir + key + "/" + key + "_kde_1D_%s.pdf" % self.kde_method, bbox_inches='tight')
                   
            elif self.kde_method == "pd":
                for key in plot_order_kde:
                    fig, ax = plt.subplots(figsize = (20, 10), nrows=1, ncols=2)
                    ax      = ax.ravel() # unravel ax object to easily index 
                    cnt     = 0
                    for pair in self.diMuonKey:
                        label_name = key + "_" + pair[-2:]
                        colName    = key + "_" + pair
                        tempDF     = pd.DataFrame(self.invMass[key][pair], columns = [colName])

                        tempDF.hist(bins = 100, range = (-20, 80), label = label_name, facecolor = hist_facecolor, edgecolor = hist_edgecolor, alpha = 0.8, density = 1, ax = ax[cnt])
                        if scott:
                            tempDF.plot.kde(label = 'KDE: Scott', color = "red", linewidth = 3, ax = ax[cnt]);

                        if bw is not None:
                            if type(bw) is list:
                                for bandwidth in bw:
                                    tempKDEplot = tempDFplot.kde(bw_method = bandwidth, label ='KDE: BW = %.2f' % bandwidth, linewidth = 3, ax = ax[cnt])
                            elif type(bw) is float or int:
                                tempKDEplot = tempDFplot.kde(bw_method = bandwidth, label ='KDE: BW = %.2f' % bandwidth, linewidth = 3, ax = ax[cnt])

                        ax[cnt].set_xlabel(r"$m_{\mu\mu_{%d}}$ [GeV]" % (cnt + 1), loc = 'right')
                        ax[cnt].set_ylabel('Probability')
                        ax[cnt].grid()
                        ax[cnt].legend(['KDE: Scott', 'KDE: BW = 0.64', 'Hist'], loc = 'upper left', fontsize = 16, shadow = True, borderpad = 1)
                        ax[cnt].grid(which = 'minor', alpha = 0.2)
                        ax[cnt].grid(which = 'major', alpha = 0.5)

                        if save:
                            fig.savefig(dataDir + key + "/" + key + "_kde_1D_%s.pdf" % self.kde_method, bbox_inches='tight')
                                
            elif self.kde_method == "sklearn":
                for key in plot_order_kde:
                    fig, ax = plt.subplots(figsize = (20, 10), nrows = 1, ncols = 2)
                    ax      = ax.ravel() # unravel ax object to easily index 
                    cnt     = 0
                    for pair in self.diMuonKey:
                        temp_df    = pd.DataFrame(self.invMass[key][pair])
                        label_name = key + "_" + pair[-2:]
                        ax[cnt].hist(self.invMass_shaped[key][pair], bins = 100, range = hist_range, label = label_name, facecolor = hist_facecolor, edgecolor = hist_edgecolor, alpha = 0.7, density = 1)
                        ax[cnt].plot(self.vals, self.pdf_bkg_1D[key][pair], color = "g", linewidth = 3)
                        temp_df.plot.kde(label = 'KDE: Scott', color = "red", linewidth = 3, ax = ax[cnt])
                        #ax[cnt].set_ylim(0, 0.4)
                        ax[cnt].set_xlim(hist_range)
                        ax[cnt].grid()
                        ax[cnt].set_xlabel(r"$m_{\mu\mu_{%d}}$ [GeV]" % (cnt + 1), loc = 'right')
                        ax[cnt].set_ylabel('Probability')
                        ax[cnt].legend(['KDE: BW = %.2f' % self.bw, 'Scott', label_name], loc = 'upper left', fontsize = 16, shadow = True, borderpad = 1)
                        # ax.legend([hist, kde_sc, kde_0p03], ['Hist', 'KDE: Scott', 'KDE: BW = 0.03'])
                        ax[cnt].grid(which = 'minor', alpha = 0.2)
                        ax[cnt].grid(which = 'major', alpha = 0.5)
                        
                        cnt += 1

                    if save:
                        fig.savefig(dataDir + key + "/" + key + "_kde_1D_%s.pdf" % self.kde_method, bbox_inches = 'tight')
                              
        if self.dataset == "sig":
            if self.kde_method == "scipy":
                for key in self.sig_dir:
                    fig, ax = plt.subplots(figsize = (20, 10), nrows = 1, ncols = 2)
                    ax      = ax.ravel() # unravel ax object to easily index 
                    cnt     = 0
                    for pair in self.diMuonKey:
                        label_name = key + "_" + pair[-2:]
                        ax[cnt].hist(self.invMass[key][pair], bins = 100, range = (0, 60), label = label_name, facecolor = hist_facecolor, edgecolor = hist_edgecolor, alpha = 0.7, density = 1)
                        ax[cnt].plot(ind, self.pdf_sig_1D[key][pair], color = kde_color, linewidth = 3, label = 'BW: Scott')
                        ax[cnt].set_xlim(hist_range)
                        ax[cnt].set_xlabel(r"$m_{\mu\mu_{%d}}$ [GeV]" % (cnt + 1), loc = 'right')
                        ax[cnt].set_ylabel('Probability', loc = "top")
                        ax[cnt].grid()
                        ax[cnt].grid(which = 'minor', alpha = 0.2)
                        ax[cnt].grid(which = 'major', alpha = 0.5)
                        ax[cnt].legend(loc = 'upper left', fontsize = 18, shadow = True, borderpad = 1)
                        
                        cnt += 1
                        plt.close()
                        
                    if save:
                        fig.savefig(dataDir + key + "/" + key + "_kde_1D_%s.pdf" % self.kde_method, bbox_inches='tight')
                        
                    
            elif self.kde_method == "pd":
                for key in self.sig_dir:
                    fig, ax = plt.subplots(figsize = (20, 10), nrows = 1, ncols = 2)
                    ax = ax.ravel() # unravel ax object to easily index 
                    cnt = 0
                    for pair in self.diMuonKey:
                        label_name = key + "_" + pair[-2:]
                        colName    = key + "_" + pair
                        tempDF     = pd.DataFrame(self.invMass[key][pair], columns = [colName])

                        tempDF.hist(bins = 100, range = (-20, 80), label = label_name, facecolor = hist_facecolor, edgecolor = hist_edgecolor, alpha = 0.8, density = 1, ax = ax[cnt])
                        if scott:
                            tempDF.plot.kde(label = 'KDE: Scott', color = "red", linewidth = 3, ax = ax[cnt]);

                        if bw is not None:
                            if type(bw) is list:
                                for bandwidth in bw:
                                    tempKDEplot = tempDFplot.kde(bw_method = bandwidth, label ='KDE: BW = %.2f' % bandwidth, linewidth = 3, ax = ax[cnt])
                            elif type(bw) is float or int:
                                tempKDEplot = tempDFplot.kde(bw_method = bandwidth, label ='KDE: BW = %.2f' % bandwidth, linewidth = 3, ax = ax[cnt])

                        ax[cnt].set_xlabel(r"$m_{\mu\mu_{%d}}$ [GeV]" % (cnt + 1), loc = 'right')
                        ax[cnt].set_ylabel('Probability')
                        ax[cnt].grid()
                        ax[cnt].legend(['KDE: Scott', 'KDE: BW = 0.64', 'Hist'], loc = 'upper left', fontsize = 16, shadow = True, borderpad = 1)
                        ax[cnt].grid(which='minor', alpha=0.2)
                        ax[cnt].grid(which='major', alpha=0.5)

                        if save:
                            fig.savefig(dataDir + key + "/" + key + "_kde_1D_%s.pdf" % self.kde_method, bbox_inches='tight')
                                
            elif self.kde_method == "sklearn":
                for key in self.sig_dir:
                    fig, ax = plt.subplots(figsize = (20, 10), nrows = 1, ncols = 2)
                    ax      = ax.ravel() # unravel ax object to easily index 
                    cnt     = 0
                    for pair in self.diMuonKey:
                        temp_df    = pd.DataFrame(self.invMass[key][pair])
                        label_name = key + "_" + pair[-2:]
                        ax[cnt].hist(self.invMass_shaped[key][pair], bins = 100, range = hist_range, label = label_name, facecolor = hist_facecolor, edgecolor = hist_edgecolor, alpha = 0.7, density = 1)
                        ax[cnt].plot(self.vals, self.pdf_sig_1D[key][pair], color = "g", linewidth = 3)
                        temp_df.plot.kde(label = 'KDE: Scott', color = "red", linewidth = 3, ax = ax[cnt])
                        #ax[cnt].set_ylim(0, 0.4)
                        ax[cnt].set_xlim(hist_range)
                        ax[cnt].grid()
                        ax[cnt].set_xlabel(r"$m_{\mu\mu_{%d}}$ [GeV]" % (cnt + 1), loc = 'right')
                        ax[cnt].set_ylabel('Probability')
                        ax[cnt].legend(['KDE: BW = %.2f' % self.bw, 'Scott', label_name], loc = 'upper left', fontsize = 16, shadow = True, borderpad = 1)
                        # ax.legend([hist, kde_sc, kde_0p03], ['Hist', 'KDE: Scott', 'KDE: BW = 0.03'])
                        ax[cnt].grid(which = 'minor', alpha = 0.2)
                        ax[cnt].grid(which = 'major', alpha = 0.5)
                        
                        cnt += 1

                    if save:
                        fig.savefig(dataDir + key + "/" + key + "_kde_1D_%s.pdf" % self.kde_method, bbox_inches='tight')
                    
    def total_bkg_1D(self, sf = True):
        if sf:
            self.bkg_1D_scaled = True
            
        if self.kde_method == "pd":
            print("Cannot calculate the total KDE estimate for the background using the pandas ('pd') method. Exiting...\n")
            sys.exit()
        
        if self.dataset == "bkg":
            if self.kde_method == "scipy":
                self.total_pdf_1D = {self.diMuonKey[0]: None, self.diMuonKey[1]: None}
                tempA0 = 0
                tempA1 = 0
                for key in plot_order_kde:
                    for pair in self.diMuonKey:
                        if pair == "invMassA0":
                            if sf:
                                tempA0 +=  self.pdf_bkg_1D[key][pair] * scale_factors_bkg[key] * self.numEvents[key]
                            else:
                                tempA0 +=  self.pdf_bkg_1D[key][pair]
                        elif pair == "invMassA1":
                            if sf:
                                tempA1 +=  self.pdf_bkg_1D[key][pair] * scale_factors_bkg[key] * self.numEvents[key]
                            else:
                                tempA1 +=  self.pdf_bkg_1D[key][pair]

                self.total_pdf_1D["invMassA0"] = tempA0
                self.total_pdf_1D["invMassA1"] = tempA1
        elif self.dataset == "sig":
            print(colors.RED+ "Not implemented for signal dataset! Exiting...\n" + colors.ENDC)
            sys.exit()
    
    def calc_kde_2D(self, method = 0):
        epsilon_inverse   = 1 / sys.float_info.epsilon
        xx, yy            = np.mgrid[hist_range[0]:hist_range[1]:100j, hist_range[0]:hist_range[1]:100j]
        self.kde_2D       = {}
        self.kde_2Dmethod = {}
        if self.dataset == "bkg":
            if self.kde_method == "scipy" or self.kde_method == "sklearn":
                if method == 0: # perform kde estimate on 2D field                    
                    for key in plot_order_kde:
                        self.kde_2Dmethod[key] = 0
                        values    = np.vstack([self.invMass[key][self.diMuonKey[0]], self.invMass[key][self.diMuonKey[1]]])
                        if np.linalg.cond(values) < epsilon_inverse:
                            print(colors.RED + "WARNING: Singular matrix encountered using method %d for the %s dataset. Trying method 1..." % (self.kde_2Dmethod[key], key) + colors.ENDC)

                            self.kde_2Dmethod[key] = 1
                            self.kde_2D[key]       = np.outer(self.pdf_bkg_1D[key][self.diMuonKey[0]], self.pdf_bkg_1D[key][self.diMuonKey[1]])
                        else:
                            positions = np.vstack([xx.ravel(), yy.ravel()])
                            kernel    = st.gaussian_kde(values)
                            
                            self.kde_2D[key] = np.reshape(kernel(positions).T, xx.shape)

                elif method == 1: # use outer product of kde pdfs previously calculated
                    for key in plot_order_kde:
                        self.kde_2Dmethod[key] = 1
                        self.kde_2D[key] = np.outer(self.pdf_bkg_1D[key][self.diMuonKey[0]], self.pdf_bkg_1D[key][self.diMuonKey[1]])
            elif self.kde_method == "pd":
                print(colors.RED + "2D KDE not implemented for pandas method." + colors.ENDC)
                sys.exit()
                
        elif self.dataset == "sig":
            if self.kde_method == "scipy" or self.kde_method == "sklearn":
                if method == 0: # perform kde estimate on 2D field
                    #self.kde_2Dmethod = 0
                    for key in self.sig_dir:
                        self.kde_2Dmethod[key] = 0
                        values = np.vstack([self.invMass[key][self.diMuonKey[0]], self.invMass[key][self.diMuonKey[1]]])
                        if np.linalg.cond(values) < epsilon_inverse: # Check condition number; if singular, skip to method 1
                            print(colors.RED + "WARNING: Singular matrix encountered using method %d for the %s dataset. Trying method 1..." % (self.kde_2Dmethod[key], key) + colors.ENDC)
                            self.kde_2Dmethod[key] = 1
                            self.kde_2D[key] = np.outer(self.pdf_sig_1D[key][self.diMuonKey[0]], self.pdf_sig_1D[key][self.diMuonKey[1]])
                        else:
                            xx, yy    = np.mgrid[hist_range[0]:hist_range[1]:100j, hist_range[0]:hist_range[1]:100j]
                            positions = np.vstack([xx.ravel(), yy.ravel()])
                            kernel    = st.gaussian_kde(values)
                            
                            self.kde_2D[key] = np.reshape(kernel(positions).T, xx.shape)                        
                elif method == 1: # use outer product of kde pdfs previously calculated
                    for key in self.sig_dir:
                        self.kde_2Dmethod[key] = 1
                        self.kde_2D[key] = np.outer(self.pdf_sig_1D[key][self.diMuonKey[0]], self.pdf_sig_1D[key][self.diMuonKey[1]])
            elif self.kde_method == "pd":
                print("2D KDE not implemented for pandas method.")
                sys.exit()
                
        if np.array(list(self.kde_2Dmethod.values())).all() != 1 or np.array(list(self.kde_2Dmethod.values())).all() == 0:
            print(colors.YELLOW + "WARNING: For the %s dataset not all 2D KDEs have been calculated using the same method! See below:\n" % key, self.kde_2Dmethod)
                                                            
    def calc_total_bkg_2D(self, sf = True, ret = True):
        if self.dataset == "sig":
            print(colors.RED + "calc_total_bkg_2D() can only be used for background data! Exiting...\n\n" + colors.ENDC)
            sys.exit()
        elif self.dataset == "bkg":
            self.total_pdf_2D = 0
            print("total_pdf_2D initial ", self.total_pdf_2D)
            for key in plot_order_kde:
                print("method: ", self.kde_2Dmethod[key])
                if self.kde_2Dmethod[key] == 0:
                    if sf:
                        self.total_pdf_2D +=  self.kde_2D[key] * scale_factors_bkg[key] * self.numEvents[key]  # ?????
                    else:
                        self.total_pdf_2D +=  self.kde_2D[key]
                elif self.kde_2Dmethod[key] == 1:
                    if self.bkg_1D_scaled: # Don't apply scale factors a second time when calculating the total bkg KDE
                        self.total_pdf_2D += np.outer(self.total_pdf_1D[self.diMuonKey[0]], self.total_pdf_1D[self.diMuonKey[1]]) 
                    else:
                        self.total_pdf_2D += np.outer(self.total_pdf_1D[self.diMuonKey[0]], self.total_pdf_1D[self.diMuonKey[1]]) * scale_factors_bkg[key] * self.numEvents[key] 
        if ret:
            return self.total_pdf_2D
            
    def plot_kde_total_1D(self, save = True):
        fig, ax = plt.subplots(figsize = (20, 10), nrows = 1, ncols = 2)
        ax = ax.ravel()
        cnt = 0
        for pair in self.diMuonKey:
            ax[cnt].plot(ind, self.total_pdf_1D[pair], color = "darkred", linewidth = 3)
            ax[cnt].set_xlabel(r"$m_{\mu\mu_{%s}}$ [GeV]" % str(cnt + 1), loc = "right")
            ax[cnt].set_ylabel('Number of events', loc = "top")
            ax[cnt].grid()
            ax[cnt].grid(which = "minor", alpha = 0.2)
            ax[cnt].grid(which = "major", alpha = 0.5)
            ax[cnt].set_ylim(0, 2)
            
            if save:
                fig.savefig(dataDir + "total_background_A0_A1_kde_1D.pdf", bbox_inches = 'tight')
            cnt += 1
        
    def plot_kde_2D(self, style = "filled", verbose = False, save = True):
        xx, yy = np.mgrid[hist_range[0]:hist_range[1]:100j, hist_range[0]:hist_range[1]:100j]
        if self.dataset == "bkg":
            if verbose:
                print(colors.GREEN + "Plotting the 2D KDEs for the background dataset\n\n" + colors.ENDC)
                
            if self.kde_method == "scipy" or self.kde_method == "sklearn":
                for key in plot_order_kde:
                    fig, ax = plt.subplots(figsize = (14, 10))
                    ax.set_xlim(hist_range)
                    ax.set_ylim(hist_range)
                    if style == "contour":
                        cset = ax.contour(xx, yy, self.kde_2D[key], colors = 'k')
                        ax.clabel(cset, inline = 1, fontsize = 10)
                        
                    # Filled contour plot
                    plt.contourf(xx, yy, self.kde_2D[key], cmap = 'coolwarm')
                    plt.colorbar()
                    ax.set_xlabel(r"$m_{\mu\mu_{1}}$ [GeV]", loc = "right")
                    ax.set_ylabel(r"$m_{\mu\mu_{2}}$ [GeV]", loc = "top")
                    ax.set_title(titles[key])
                    
                    if save:
                        if style == "filled": 
                            fig.savefig(dataDir + key + "/" + key + "_kde_2D_filled-contour_method_%d.pdf" % self.kde_2Dmethod[key], bbox_inches = 'tight')
                        elif style == "contour":
                            fig.savefig(dataDir + key + "/" + key + "_kde_2D_contour_method_%d.pdf" % self.kde_2Dmethod[key], bbox_inches = 'tight')
                            
        if self.dataset == "sig":
            if self.kde_method == "scipy" or self.kde_method == "sklearn":
                if verbose:
                    print(colors.GREEN + "Plotting the 2D KDEs for the signal dataset\n\n" + colors.ENDC)
                for key in self.sig_dir:
                    fig, ax = plt.subplots(figsize = (14, 10))
                    ax.set_xlim(hist_range)
                    ax.set_ylim(hist_range)
                    if style == "contour":
                        cset = ax.contour(xx, yy, self.kde_2D[key], colors = 'k')
                        ax.clabel(cset, inline = 1, fontsize = 10)
                        
                    # Filled contour plot
                    plt.contourf(xx, yy, self.kde_2D[key], cmap = 'coolwarm')
                    plt.colorbar()
                    ax.set_xlabel(r"$m_{\mu\mu_{1}}$ [GeV]", loc = "right")
                    ax.set_ylabel(r"$m_{\mu\mu_{2}}$ [GeV]", loc = "top")                    
                    ax.set_title(r'$m_{Z_{D}} = $ %s, $m_{f_{D1}} = $ %s' % (key.split("_")[1], key.split("_")[2]))
                    plt.show()
                    if save:
                        if style == "filled": 
                            fig.savefig(dataDir + key + "/" + key + "_kde_2D_filled-contour_method_%d.pdf" % self.kde_2Dmethod[key], bbox_inches = 'tight')
                        elif style == "contour":
                            fig.savefig(dataDir + key + "/" + key + "_kde_2D_contour_method_%d.pdf" % self.kde_2Dmethod[key], bbox_inches = 'tight')
    
    def plot_total_kde_2D(self, verbose = False, save = True):
        
        if self.kde_2Dmethod == None:
            print(colors.RED + "Must perform 2D KDE calculation using calc_kde_2D(). Exiting..." + colors.ENDC)
            sys.exit()
        if self.dataset == "sig":
            print(colors.RED + "Total KDE can only be plotted for background samples. Exiting..." + colors.ENDC)
            sys.exit()
        elif self.dataset == "bkg":
            if verbose:
                print(colors.GREEN + "Plotting the total 2D background KDE\n\n" + colors.ENDC)
                
            if self.kde_method == "scipy" or "sklearn":
                # Peform the kernel density estimate
                xx, yy = np.mgrid[hist_range[0]:hist_range[1]:100j, hist_range[0]:hist_range[1]:100j]
                fig = plt.figure(figsize=(14, 10))
                ax  = fig.gca()
                ax.set_xlim(hist_range)
                ax.set_ylim(hist_range)
                plt.contourf(xx, yy, self.total_pdf_2D, cmap = 'coolwarm')
                plt.colorbar()
                ax.set_xlabel(r"$m_{\mu\mu_{1}}$ [GeV]", loc = "right")
                ax.set_ylabel(r"$m_{\mu\mu_{2}}$ [GeV]", loc = "top")
                ax.set_title('Total Background')
                plt.show()
                
                if np.array(list(self.kde_2Dmethod.values())).all() == 0:
                    if save:
                        fig.savefig(dataDir + "_kde_total_2D_method_0.pdf", bbox_inches = 'tight')
                elif np.array(list(self.kde_2Dmethod.values())).all() == 1:
                    if save:
                        fig.savefig(dataDir + "_kde_total_2D_method_1.pdf", bbox_inches = 'tight')
                else:
                    print(colors.YELLOW + "Not all 2D KDEs performed using the same method, saving as " + dataDir + "_kde_total_background_2D_method_0and1.pdf" + colors.ENDC)
                    if save:
                        fig.savefig(dataDir + "_kde_total_background_2D_method_0and1.pdf", bbox_inches = 'tight')
    
    def french_flag(self, total_bkg_kde_2D, contour = False, verbose = False, save = True):
        if self.kde_method == "bkg":
            print(colors.YELLOW + "French flag plots created from signal / background; please initialize the kde() object as 'sig' and load input the total 2D background KDE at the time of execution." + colors.ENDC)
        elif self.kde_method == "sig":
            if total_bkg_kde_2D is None:
                print(colors.YELLOW + "Please load  the total 2D background KDE using load_total_bkg_kde(total_bkg_kde_2D)" + colors.ENDC)
        
        if verbose:
            print(colors.GREEN + "Producing the French flag plots\n\n" + colors.ENDC)
            
        cMap   = ListedColormap(['blue', 'white', 'red']) # define custom color map (french flag)
        xx, yy = np.mgrid[hist_range[0]:hist_range[1]:100j, hist_range[0]:hist_range[1]:100j]
        
        for key in self.sig_dir:
            divided_kde = np.divide((self.kde_2D[key] * scale_factors_sig[key] * self.numEvents[key]), np.sqrt(total_bkg_kde_2D))
            
            fig  = plt.figure(figsize = (13, 10))
            ax   = fig.gca()
            
            ax.set_xlim(hist_range)
            ax.set_ylim(hist_range)
            
            plt.contourf(xx, yy, divided_kde, [0, 0.2, 1, 3], extend = 'max', colors = ['blue', 'white', 'red'])
            cbar = plt.colorbar(ticks = [0.1, 0.6, 2], drawedges = True)
            cbar.outline.set_color('black')
            cbar.outline.set_linewidth(2)
            cbar.ax.set_yticklabels(["<0.2: CR", "0.2-1", ">1: SR"])  # vertically oriented colorbar
            if contour:
                cset = ax.contour(xx, yy, divided_kde, [0, 0.2,  1, 3] , extend = 'max', colors = 'k')
                ax.clabel(cset, inline = 1, fontsize = 10)
                
            ax.set_xlabel(r"$m_{\mu\mu_{1}}$ [GeV]", loc = "right")
            ax.set_ylabel(r"$m_{\mu\mu_{2}}$ [GeV]", loc = "top")
            ax.set_title(r'Ratio $S/\sqrt{B}, m_{Z_{D}} = %s, m_{f_{D1}} = %s$ GeV' % (key.split("_")[1], key.split("_")[2]))
            
            if save:
                fig.savefig(dataDir + key + "/" + key + "_french_flag_method_%d.pdf" % self.kde_2Dmethod[key], bbox_inches = 'tight')

    '''
    def calc_total_sig_2D(self, samples, verbose = False, ret = True):
        if self.dataset == "bkg":
            print(colors.RED + "calc_total_sig_2D() can only be used for signal! Exiting...\n\n" + colors.ENDC)
            sys.exit()
        elif self.dataset == "sig":
            self.total_kde_sig = True
            self.total_pdf_2D = 0
            if verbose:
                temp_str    = ""
                num_samples = len(samples)
                cnt         = 0
                for key in samples:
                    if cnt < num_samples:
                        temp_str += key + ", "
                    elif cnt == num_samples:
                        temp_str += key
                    
                    cnt += 1
                print(colors.GREEN + "Calculating the total 2D KDE for the following signal datasets: %s", temp_str  + colors.ENDC)
                
            for key in samples:
                if self.kde_2Dmethod[key] == 0:
                    self.total_pdf_2D +=  self.kde_2D[key] * scale_factors_sig[key] * self.numEvents[key]
                elif self.kde_2Dmethod[key] == 1:
                    
                    self.total_pdf_2D = np.outer(self.total_pdf_1D[self.diMuonKey[0]], self.total_pdf_1D[self.diMuonKey[1]])
            
        if ret:
            return self.total_pdf_2D

    def plot_total_sig_kde_2D(self, samples, save = False):
        if self.total_kde_sig == None:
            print(colors.RED + "Must perform 2D KDE calculation using ccalc_total_sig_2D(). Exiting...\n\n" + colors.ENDC)
            sys.exit()
        if self.dataset == "bkg":
            print(colors.RED + "Total 2D signal KDE can only be plotted for signal samples. Exiting...\n\n" + colors.ENDC)
            sys.exit()
        elif self.dataset == "sig":
            if verbose:
                temp_str    = ""
                num_samples = len(samples)
                cnt         = 0
                for key in samples:
                    if cnt < num_samples:
                        temp_str += key + ", "
                    elif cnt == num_samples:
                        temp_str += key
                    
                    cnt += 1
                print(colors.GREEN + "Plotting the total 2D signal KDE for %s\n\n" % temp_str + colors.ENDC)
        
            xx, yy = np.mgrid[hist_range[0]:hist_range[1]:100j, hist_range[0]:hist_range[1]:100j]
            fig = plt.figure(figsize=(14, 10))
            ax  = fig.gca()
            ax.set_xlim(hist_range)
            ax.set_ylim(hist_range)
            plt.contourf(xx, yy, self.total_pdf_2D, cmap = 'coolwarm')
            plt.colorbar()
            ax.set_xlabel(r"$m_{\mu\mu_{1}}$ [GeV]", loc = "right")
            ax.set_ylabel(r"$m_{\mu\mu_{2}}$ [GeV]", loc = "top")

            total_fd_str = ""
            num_samples = len(samples)
            cnt         = 0
            for key in samples:
                if cnt < num_samples:
                    total_fd_str += key.split("_")[2] + ", "
                elif cnt == num_samples:
                    total_fd_str += key.split("_")[2]

                cnt += 1

            ax.set_title('Total Background for $m_{Z_{D}} = %s, m_{f_{D1}} = %s$' % (samples[0].split("_")[1], total_fd_str))
            plt.show()

            if save:
                if np.array(list(self.kde_2Dmethod.values())).all() == 0:
                    fig.savefig(dataDir + "_kde_total_2D_method_0.pdf", bbox_inches = 'tight')
                elif np.array(list(self.kde_2Dmethod.values())).all() == 1:
                    fig.savefig(dataDir + "_kde_total_2D_method_1.pdf", bbox_inches = 'tight')
                else:
                    print(colors.YELLOW + "Not all 2D KDEs performed using the same method, saving as " + dataDir + "_kde_total_background_2D_method_0and1.pdf" + colors.ENDC)
                    fig.savefig(dataDir + "_kde_total_background_2D_method_0and1.pdf", bbox_inches = 'tight')
                    
    '''