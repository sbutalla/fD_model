#!/bin/env python

'''
Various data processing tools for the fD model muon matching project and a
collection of utilities for matching gen level to sel level muons,
generating the possible permutations, invariant mass calculation,
and preparing the data for ML algorithms.

Functions of the class to be executed sequentially for the fD
Data Processing Tools block.

Stephen D. Butalla & Mehdi Rahmani
2021/06/01

========== Change Log ==========
S.D.B, 2022/06/07
- Added colors class
- Reformatted standalone functions into the
  process data class; functions of class
  must be executed sequentially---will fix
  this later
- Integrated functions from matchingUtilities 
  module

'''
import numpy as np
import pandas as pd
from math import sqrt
from random import randint
from tqdm import tqdm # Progress bar for loops
from uproot import open as openUp # Transfer ROOT data into np arrays
import numpy as np
import sys

################################################################
################################################################
################## fD Data Proessing Tools #####################
################################################################
################################################################

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

class processData:

    def __init__(self, dataset):
        if dataset not in ["mc", "bkg"]:
            print("Dataset type must be either 'mc' or 'bkg'.")
        else:
            self.dataset = dataset
        return None

    def extractData(self, rootFile, root_dir, ret = False, verbose = False):
        '''
        Accepts absolute (or relative) path to a ROOT file and
        returns simulated signal event data.
        '''
        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Extracting data" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        if verbose:
            print(colors.GREEN + "Opening file %s[%s]" % (rootFile, root_dir) + colors.ENDC)

        file = openUp(rootFile)
        #data = file['cutFlowAnalyzerPXBL4PXFL3;1/Events;1']
        data = file[root_dir]
        
        if verbose:
            print(colors.GREEN + "Extracting sel mu data (eta, phi, pT, and charge)" + colors.ENDC)

        selMu0_eta     = np.asarray(data['selMu0_eta'].array())
        selMu1_eta     = np.asarray(data['selMu1_eta'].array())
        selMu2_eta     = np.asarray(data['selMu2_eta'].array())
        selMu3_eta     = np.asarray(data['selMu3_eta'].array())
        self.selMu_eta = np.column_stack((selMu0_eta, selMu1_eta, selMu2_eta, selMu3_eta))

        selMu0_phi     = np.asarray(data['selMu0_phi'].array())
        selMu1_phi     = np.asarray(data['selMu1_phi'].array())
        selMu2_phi     = np.asarray(data['selMu2_phi'].array())
        selMu3_phi     = np.asarray(data['selMu3_phi'].array())
        self.selMu_phi = np.column_stack((selMu0_phi, selMu1_phi, selMu2_phi, selMu3_phi))

        selMu0_pt     = np.asarray(data['selMu0_pT'].array())
        selMu1_pt     = np.asarray(data['selMu1_pT'].array())
        selMu2_pt     = np.asarray(data['selMu2_pT'].array())
        selMu3_pt     = np.asarray(data['selMu3_pT'].array())
        self.selMu_pt = np.column_stack((selMu0_pt, selMu1_pt, selMu2_pt, selMu3_pt))
        
        selMu0_charge     = np.asarray(data['selMu0_charge'].array())
        selMu1_charge     = np.asarray(data['selMu1_charge'].array())
        selMu2_charge     = np.asarray(data['selMu2_charge'].array())
        selMu3_charge     = np.asarray(data['selMu3_charge'].array())
        self.selMu_charge = np.column_stack((selMu0_charge, selMu1_charge, selMu2_charge, selMu3_charge))

        if self.dataset == "bkg":
            pass
        else:
            if verbose:
                print(colors.GREEN + "Extracting gen mu data (eta, phi, pT, and charge)" + colors.ENDC)

            genA0Mu0_eta    = np.asarray(data['genA0Mu0_eta'].array())
            genA0Mu1_eta    = np.asarray(data['genA0Mu1_eta'].array())
            genA1Mu0_eta    = np.asarray(data['genA1Mu0_eta'].array())
            genA1Mu1_eta    = np.asarray(data['genA1Mu1_eta'].array())
            self.genAMu_eta = np.column_stack((genA0Mu0_eta, genA0Mu1_eta, genA1Mu0_eta, genA1Mu1_eta))

            genA0Mu0_phi    = np.asarray(data['genA0Mu0_phi'].array())
            genA0Mu1_phi    = np.asarray(data['genA0Mu1_phi'].array())
            genA1Mu0_phi    = np.asarray(data['genA1Mu0_phi'].array())
            genA1Mu1_phi    = np.asarray(data['genA1Mu1_phi'].array())
            self.genAMu_phi = np.column_stack((genA0Mu0_phi, genA0Mu1_phi, genA1Mu0_phi, genA1Mu1_phi))
            
            genA0Mu0_pt    = np.asarray(data['genA0Mu0_pt'].array())
            genA0Mu1_pt    = np.asarray(data['genA0Mu1_pt'].array())
            genA1Mu0_pt    = np.asarray(data['genA1Mu0_pt'].array())
            genA1Mu1_pt    = np.asarray(data['genA1Mu1_pt'].array())
            self.genAMu_pt = np.column_stack((genA0Mu0_pt, genA0Mu1_pt, genA1Mu0_pt, genA1Mu1_pt))

            genA0Mu0_charge    = np.asarray(data['genA0Mu0_charge'].array())
            genA0Mu1_charge    = np.asarray(data['genA0Mu1_charge'].array())
            genA1Mu0_charge    = np.asarray(data['genA1Mu0_charge'].array())
            genA1Mu1_charge    = np.asarray(data['genA1Mu1_charge'].array())
            self.genAMu_charge = np.column_stack((genA0Mu0_charge, genA0Mu1_charge, genA1Mu0_charge, genA1Mu1_charge))
        
        if ret:
            if self.dataset == "mc":
                if verbose:
                    print('Arrays returned: selMu_eta, selMu_phi, selMu_pt, selMu_charge, genAMu_eta, genAMu_phi, genAMu_pt, genAMu_charge')

                return self.selMu_eta, self.selMu_phi, self.selMu_pt, self.selMu_charge, self.genAMu_eta, self.genAMu_phi, self.genAMu_pt, self.genAMu_charge
            elif self.dataset == "bkg":
                if verbose:
                    print('Arrays returned: selMu_eta, selMu_phi, selMu_pt, selMu_charge')

                return self.selMu_eta, self.selMu_phi, self.selMu_pt, self.selMu_charge
        else:
            return None

    def prelimCuts(self, ret = False, verbose = False):

        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Applying preliminary cuts" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        if verbose:
            print("Determing the events that were not properly reconstructed or are outside of the geometric acceptance of the detector system (eta > 2.4)")

        badPhi       = np.unique(np.where(self.selMu_phi == -100)[0])
        badSelEta    = np.unique(np.where(abs(self.selMu_eta) > 2.4)[0])
        badSelpT     = np.unique(np.where(self.selMu_pt == -100)[0])
        badSelCharge = np.unique(np.where(np.sum(self.selMu_charge, axis = 1) != 0))

        if self.dataset == "mc":
            badGenAEta = np.unique(np.where(abs(self.genAMu_eta) > 2.4)[0])
        
        # Convert to lists so we can add without broadcasting problems
        badPhi     = list(badPhi)
        badSelEta  = list(badSelEta)
        badSelpT   = list(badSelpT)
        badCharge  = list(badSelCharge)

        if self.dataset == "mc":
            badGenAEta = list(badGenAEta)
        
        if self.dataset == "bkg":
            self.badEvents = sorted(np.unique(badPhi + badSelEta + badSelpT + badCharge))
        else:    
            self.badEvents = sorted(np.unique(badPhi + badGenAEta + badSelEta + badSelpT + badCharge)) # Add lists, return unique values, and sort to preserve order
        
        if verbose:
            print(25 * '*' +' CUT INFO ' + 25 * '*' + '\n')
            print('Total number of events failing reconstruction in phi: {}'.format(len(badPhi)))
            print('Total number of events with sel eta > 2.4: {}'.format(len(badSelEta)))

            if self.dataset == "mc":
                print('Total number of events with gen eta > 2.4: {}'.format(len(badGenAEta)))

            print('Total number of events with sel pT == -100: {}'.format(len(badSelpT)))
            print('Total number of events failing charge reconstruction: {}'.format(len(badCharge)))
            print('Total number of bad events: {}'.format(len(self.badEvents)))
            print('\n' + 60*'*')
     
        if ret:
            return self.badEvents
        else:
            return None

    def removeBadEvents(self, cut, ret = False, verbose = False):
        '''
        Removes bad events given the data and list of indices

        cut: string; which observable to apply the cut
        '''
        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Removing bad events" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        if verbose:
            print("Removing events that were not properly reconstructed or are outside of the geometric acceptance of the detector system.")

        if cut == "selMu_eta":
            self.selMu_etaCut = np.delete(self.selMu_eta, self.badEvents, axis=0)
            if ret:
                return self.selMu_etaCut
            else:
                return None

        elif cut == "selMu_phi":
            self.selMu_phiCut = np.delete(self.selMu_phi, self.badEvents, axis=0)
            return self.selMu_phiCut

        elif cut == "selMu_pt":
            self.selMu_ptCut = np.delete(self.selMu_pt, self.badEvents, axis=0)
            return self.selMu_ptCut

        elif cut == "selMu_charge":
            self.selMu_chargeCut = np.delete(self.selMu_charge, self.badEvents, axis=0)
            return self.selMu_chargeCut

        elif cut == "genAMu_eta":
            self.genAMu_etaCut = np.delete(self.genAMu_eta, self.badEvents, axis=0)
            return self.genAMu_etaCut

        elif cut == "genAMu_phi":
            self.genAMu_phiCut = np.delete(self.genAMu_phi, self.badEvents, axis=0)
            return self.genAMu_phiCut

        elif cut == "genAMu_pt":
            self.genAMu_ptCut = np.delete(self.genAMu_pt, self.badEvents, axis=0)
            return self.genAMu_ptCut

        elif cut == "genAMu_charge":
            self.genAMu_chargeCut = np.delete(self.genAMu_charge, self.badEvents, axis=0)
            return self.genAMu_chargeCut

        elif cut == "all":
            self.selMu_etaCut     = np.delete(self.selMu_eta, self.badEvents, axis=0)
            self.selMu_phiCut     = np.delete(self.selMu_phi, self.badEvents, axis=0)
            self.selMu_ptCut      = np.delete(self.selMu_pt, self.badEvents, axis=0)
            self.selMu_chargeCut  = np.delete(self.selMu_charge, self.badEvents, axis=0)

            if self.dataset == "mc":
                self.genAMu_etaCut    = np.delete(self.genAMu_eta, self.badEvents, axis=0)
                self.genAMu_phiCut    = np.delete(self.genAMu_phi, self.badEvents, axis=0)
                self.genAMu_ptCut     = np.delete(self.genAMu_pt, self.badEvents, axis=0)
                self.genAMu_chargeCut = np.delete(self.genAMu_charge, self.badEvents, axis=0)

            print("selMu_chargeCut shape: ", self.selMu_chargeCut.shape)
            if ret:
                if self.dataset == "mc":
                    if verbose:
                        print("Arrays returned: selMu_etaCut, selMu_phiCut, selMu_ptCut, selMu_chargeCut, genAMu_etaCut, genAMu_phiCut, genAMu_ptCut, genAMu_chargeCut")
                    
                    return self.selMu_etaCut, self.selMu_phiCut, self.selMu_ptCut, self.selMu_chargeCut, self.genAMu_etaCut, self.genAMu_phiCut, self.genAMu_ptCut, self.genAMu_chargeCut
                elif self.dataset == "bkg":
                    if verbose:
                        print("Arrays returned: selMu_etaCut, selMu_phiCut, selMu_ptCut, selMu_chargeCut")
                    
                    return self.selMu_etaCut, self.selMu_phiCut, self.selMu_ptCut, self.selMu_chargeCut
            else:
                return None

    def matchBkgMu(self, ret = False, verbose = False):
        numEvents = self.selMu_chargeCut.shape[0]
        self.min_dRgenFinal = np.ndarray((numEvents, 4))

        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "'Matching' background muons" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        if verbose:
            print("Pseudo-matching the four muons for background data")

        for event in range(numEvents):
            tempSelCharge = self.selMu_chargeCut[event, :] # 1 x 4
            self.min_dRgenFinal[event,] = np.array(np.where(self.selMu_chargeCut[event, :])) # 1 x 4
            if tempSelCharge[0] == tempSelCharge[1]:
                self.min_dRgenFinal[event, [0, 2]] = self.min_dRgenFinal[event, [2, 0]]

        if np.sum(self.min_dRgenFinal, axis = 1).all() == np.full((numEvents), 6.).all(): # check to make sure "matching" performed properly
            pass

        else:
            for event in range(numEvents):
                tempSelCharge = self.selMu_chargeCut[event, :] # 1 x 4
                self.min_dRgenFinal[event,] = np.array(np.where(selMu_chargeCut[event, :])) # 1 x 4
                if tempSelCharge[0] == tempSelCharge[1]:
                    self.min_dRgenFinal[event, [0, 2]] = self.min_dRgenFinal[event, [2, 0]]

            if np.sum(self.min_dRgenFinal, axis = 1).all() == np.full((numEvents), 6.).all(): # check to make sure "matching" performed properly
                print("Error matching the background dimuons!\nExiting...")
                sys.exit()


        if ret:
            return self.min_dRgenFinal
        else:
            return None

    def dRgenCalc(self, ret = False):
        '''
        Calculates the dR value between the generator level and reco level muons.
        To be used to determine if the muons are reconstructed properly.
        genAMu_eta: np.ndarray; reco level muon eta
        selMu_eta:  np.ndarray; generator level muon eta
        genAMu_phi: np.ndarray; reco level muon phi
        selMu_phi:  np.ndarray; generator level muon phi
        '''
        self.dRgen = np.ndarray((self.genAMu_etaCut.shape[0], 4, 4))

        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Calculating dR between gen level and reco level muons (MC)" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        for ii in tqdm(range(self.genAMu_etaCut.shape[0])): # for each event
            for jj in range(4): # dR for each gen muon
                for ll in range(4): #for each sel muon
                    self.dRgen[ii, jj, ll] = sqrt(pow(self.genAMu_etaCut[ii, jj] - self.selMu_etaCut[ii, ll], 2)+ pow(self.genAMu_phiCut[ii, jj] - self.selMu_phiCut[ii, ll], 2))
        if ret:
            return self.dRgen
        else:
            return None

    def SSM(self, ret = False, verbose = False, extraInfo = False):

        self.min_dRgen    = np.ndarray((self.genAMu_chargeCut.shape[0], 4))
        self.dRgenMatched = np.ndarray((self.genAMu_chargeCut.shape[0], 4))

        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Using tochastic sampling method (SSM) to match gen and reco level muons" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        for event in tqdm(range(self.selMu_chargeCut.shape[0])):
            if verbose:
                print ("Event: ", event)
                
            tempGenCharge = self.genAMu_chargeCut[event, :] # 1 x 4
            tempSelCharge = self.selMu_chargeCut[event, :] # 1 x 4

            # Match first randomly chosen muon 
            index      = randint(0, 3) # Generate random int in [0,3]
            chargeTemp = tempGenCharge[index] # Get charge corresponding to AXMuY
            
            genCharge  = np.array(np.where(self.genAMu_chargeCut[event,:] == chargeTemp)).reshape((2,)) # Select gen muons where charge (array of indices)
            selCharge  = np.array(np.where(self.selMu_chargeCut[event,:] == chargeTemp)).reshape((2,))# Select sel muons where charge is == AXMuY (array of indices)

            genChargeopo = np.array(np.where(self.genAMu_chargeCut[event,:] != chargeTemp)).reshape((2,)) # Select gen muons where charge (array of indices)
            selChargeopo = np.array(np.where(self.selMu_chargeCut[event,:] != chargeTemp)).reshape((2,))

            if verbose:
                print("sel chargeopo[0]: ", selChargeopo[0])
                print("sel chargeopo[1]: ", selChargeopo[1])
                print("sel charge[0]: ", selCharge[0])
                print("sel charge[1]: ", selCharge[1])
            
                print("gen charge[0]: ", genCharge[0])
                print("gen charge[1]: ", genCharge[1])
                print("gen chargeopo[0]: ", genChargeopo[0])
                print("gen charge[1]opo: ", genChargeopo[1])

            # Calculating mindR for each same charge gen muons
            min_dR0_index1 = np.array(np.minimum(self.dRgen[event,genCharge[0], selCharge[0]], self.dRgen[event, genCharge[0], selCharge[1]])).reshape((1,))
            min_dR0_index2 = np.array(np.minimum(self.dRgen[event,genCharge[1], selCharge[0]], self.dRgen[event, genCharge[1], selCharge[1]])).reshape((1,))
            min_dR0_index3 = np.array(np.minimum(self.dRgen[event,genChargeopo[0], selChargeopo[0]], self.dRgen[event, genChargeopo[0], selChargeopo[1]])).reshape((1,))
            min_dR0_index4 = np.array(np.minimum(self.dRgen[event,genChargeopo[1], selChargeopo[0]], self.dRgen[event, genChargeopo[1], selChargeopo[1]])).reshape((1,))

            # Calculating for the first gen muon
            if self.dRgen[event, genCharge[0], selCharge[0]] ==  min_dR0_index1:
                selIndex1 = selCharge[0]
                genIndex1 = genCharge[0]

            else: 
                selIndex1 = selCharge[1]
                genIndex1 = genCharge[0]


            self.dRgenMatched[event, 0] = self.dRgen[event, genIndex1, selIndex1]

            temp = np.delete(selCharge, np.where(selCharge == selIndex1)) 

            genIndex2 = genCharge[1]
            selIndex2 = temp[0]


            self.dRgenMatched[event, 1] = self.dRgen[event, genIndex2, selIndex2]


            if self.dRgen[event,genChargeopo[0], selChargeopo[0]] ==  min_dR0_index3:
                selIndex3 = selChargeopo[0]
                genIndex3 = genChargeopo[0]

            else: 
                selIndex3 = selChargeopo[1]
                genIndex3 = genChargeopo[0]


            self.dRgenMatched[event, 2] = self.dRgen[event, genIndex3, selIndex3]
            
            tempopo = np.delete(selChargeopo, np.where(selChargeopo == selIndex3))

            genIndex4 = genChargeopo[1]
            selIndex4 = tempopo[0]  

            self.dRgenMatched[event, 3] = self.dRgen[event, genIndex4, selIndex4]

            genInd = np.array((genIndex1, genIndex2, genIndex3, genIndex4))
            selInd = np.array((selIndex1, selIndex2, selIndex3, selIndex4))
            
            for muon in range(4):
                self.min_dRgen[event, genInd[muon]] = selInd[muon]
            
            if verbose:
                print("sel muon: ", selIndex1, ", mached with gen: ", genIndex1)
                print("other sel muon: ", selIndex2, ", mached with other gen: ", genIndex2)
                print("opposite charge sel muon: ", selIndex3, ", mached with opposite charge gen: ", genIndex3)
                print("other opposite charge sel muon: ", selIndex4, ", mached with other opposite charge gen: ", genIndex4)
            
            if extraInfo:
                dEtaMatched  = np.ndarray((self.dRgen.shape[0], 4))
                dPhiMatched  = np.ndarray((self.dRgen.shape[0], 4))
                dPtMatched   = np.ndarray((self.dRgen.shape[0], 4))
            
                
                for ii in range(4):
                    dEtaMatched[event, ii] = self.genAMu_etaCut[event, genInd[ii]] - self.selMu_etaCut[event, selInd[ii]]
                    dPhiMatched[event, ii] = self.genAMu_phiCut[event, genInd[ii]] - self.selMu_phiCut[event, selInd[ii]]
                    #dPtMatched[event, ii] = genpTFinal[event, genInd[ii]] - selpTFinal[event, selInd[ii]]

        
        if verbose == False and extraInfo == False:
            if ret:
                return self.min_dRgen, self.dRgenMatched 
            else:
                return None

        if verbose == True and extraInfo == True:
            if ret:
                return self.min_dRgen, self.dRgenMatched, dEtaMatched, dPhiMatched #, dPtMatched
            else:
                return None

        if verbose == False and extraInfo == True:
            if ret:
                return dEtaMatched, dPhiMatched #, dPtMatched
            else:
                return None

    def dRcut(self, dRcut, cut, ret = False, verbose = False):
        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Cutting events that do not match the criterion: dR < %.2f" % dRcut + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        self.dRcut_events = np.unique(np.where(self.dRgenMatched > dRcut)[0])

        if cut == "selMu_eta":
            self.selMu_etaFinal = np.delete(self.selMu_etaCut, self.dRcut_events, axis=0)
            return self.selMu_etaFinal

        elif cut == "selMu_phi":
            self.selMu_phiFinal  = np.delete(self.selMu_phiCut, self.dRcut_events, axis=0)
            return self.selMu_phiFinal 

        elif cut == "selMu_pt":
            self.selMu_ptFinal  = np.delete(self.selMu_ptCut, self.dRcut_events, axis=0)
            return self.selMu_ptFinal 

        elif cut == "selMu_charge":
            self.selMu_chargeFinal = np.delete(self.selMu_chargeCut, self.dRcut_events, axis=0)
            return self.selMu_chargeFinal 

        elif cut == "genAMu_eta":
            self.genAMu_etaFinal = np.delete(self.genAMu_etaCut, self.dRcut_events, axis=0)
            return self.genAMu_etaFinal 

        elif cut == "genAMu_phi":
            self.genAMu_phiFinal = np.delete(self.genAMu_phiCut, self.dRcut_events, axis=0)
            return self.genAMu_phiFinal 

        elif cut == "genAMu_pt":
            self.genAMu_ptFinal  = np.delete(self.genAMu_ptCut, self.dRcut_events, axis=0)
            return self.genAMu_ptFinal 

        elif cut == "genAMu_charge":
            self.genAMu_chargeFinal = np.delete(self.genAMu_chargeCut, self.dRcut_events, axis=0)
            return self.genAMu_chargeFinal 

        elif cut == "min_dRgen":
            self.min_dRgenFinal = np.delete(self.min_dRgen, self.dRcut_events, axis=0)
            return self.min_dRgenFinal 

        elif cut ==  "dRgenMatched":
            self.dRgenMatched = np.delete(self.dRgenMatched, self.dRcut_events, axis=0)
            return self.dRgenMatched

        elif cut == "all":
            self.selMu_etaFinal     = np.delete(self.selMu_etaCut, self.dRcut_events, axis=0)
            self.selMu_phiFinal     = np.delete(self.selMu_phiCut, self.dRcut_events, axis=0)
            self.selMu_ptFinal      = np.delete(self.selMu_ptCut, self.dRcut_events, axis=0)
            self.selMu_chargeFinal  = np.delete(self.selMu_chargeCut, self.dRcut_events, axis=0)
            self.genAMu_etaFinal    = np.delete(self.genAMu_etaCut, self.dRcut_events, axis=0)
            self.genAMu_phiFinal    = np.delete(self.genAMu_phiCut, self.dRcut_events, axis=0)
            self.genAMu_ptFinal     = np.delete(self.genAMu_ptCut, self.dRcut_events, axis=0)
            self.genAMu_chargeFinal = np.delete(self.genAMu_chargeCut, self.dRcut_events, axis=0)
            self.min_dRgenFinal     = np.delete(self.min_dRgen, self.dRcut_events, axis=0)
            self.dRgenMatched       = np.delete(self.dRgenMatched, self.dRcut_events, axis=0)

            # Clean up memory
            del self.selMu_etaCut
            del self.selMu_phiCut
            del self.selMu_ptCut
            del self.selMu_chargeCut
            del self.genAMu_etaCut
            del self.genAMu_phiCut
            del self.genAMu_ptCut
            del self.genAMu_chargeCut

            if ret:
                if verbose:
                    print("Arrays returned: selMu_etaCut, selMu_phiCut, selMu_ptCut, selMu_chargeCut, genAMu_etaCut, genAMu_phiCut, genAMu_ptCut, genAMu_chargeCut")
            
                return self.selMu_etaFinal, self.selMu_phiFinal, self.selMu_ptFinal, self.selMu_chargeFinal, self.genAMu_etaFinal, self.genAMu_phiFinal, self.genAMu_ptFinal , self.genAMu_chargeFinal, self.min_dRgenFinal, self.dRgenMatched 
            else:
                return None
        else:
            print(colors.YELLOW + "Not a valid option; choose from the list:\nselMu_eta\nselMu_phi\nselMu_pt\nselMu_charge\ngenAMu_eta\ngenAMu_phi\ngenAMu_pt\ngenAMu_charge\n")
    
    def permutations(self, ret = False, verbose = False):
        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Generating all permutations" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        if self.dataset == "bkg":
            print("Renaming sel mu arrays")
            # Rename to keep arrays consistent (no dR cut for bkg data)
            self.selMu_etaFinal    = self.selMu_etaCut 
            self.selMu_phiFinal    = self.selMu_phiCut
            self.selMu_ptFinal     = self.selMu_ptCut
            self.selMu_chargeFinal = self.selMu_chargeCut

            # Clean up memory
            del self.selMu_etaCut
            del self.selMu_phiCut
            del self.selMu_ptCut
            del self.selMu_chargeCut


        if verbose:
            print("Calculating all permutations of the 4 muons\n\n")
            print("Calculating the wrong permutations")    

        #### WRONG PERMUTATIONS ####
        numEvents      = self.min_dRgenFinal.shape[0]
        self.wrongPerm = np.ndarray((numEvents, 2, self.min_dRgenFinal.shape[1]))
        self.allPerm   = np.ndarray((numEvents, self.wrongPerm.shape[1] + 1, 4))

        for event in tqdm(range(numEvents)):

            self.wrongPerm[event, 0] = self.min_dRgenFinal[event, :]
            self.wrongPerm[event, 1] = self.min_dRgenFinal[event, :]
            
            correctPerm = np.array(self.min_dRgenFinal[event, :], dtype=int) # Array of indices of matched data
            correctSelChargeOrder = self.selMu_chargeFinal[event, correctPerm]  # Correct order of charges (matched)
            pos = np.array(np.where(correctSelChargeOrder == 1)).reshape((2,))  # indices of positive sel muons

            if verbose:
                print("Event ", event)
                print("Correct permutation: ", correctPerm)  
                print("Correct sel muon charge order", correctSelChargeOrder)
                #print("pos", pos)

            self.wrongPerm[event, 0, pos[0]] = self.min_dRgenFinal[event, pos[1]]
            self.wrongPerm[event, 0, pos[1]] = self.min_dRgenFinal[event, pos[0]]
            
            neg = np.array(np.where(correctSelChargeOrder == -1)).reshape((2,))

            self.wrongPerm[event, 1, neg[0]] = self.min_dRgenFinal[event, neg[1]]
            self.wrongPerm[event, 1, neg[1]] = self.min_dRgenFinal[event, neg[0]]
        
        if verbose:
            print("Calculating all permutations")  

        for event in tqdm(range(self.min_dRgenFinal.shape[0])):
            self.allPerm[event, 0, :] = self.min_dRgenFinal[event, :] # correct permutation
            self.allPerm[event, 1, :] = self.wrongPerm[event, 0, :] # wrong permutation 0
            self.allPerm[event, 2, :] = self.wrongPerm[event, 1, :] # wrong permutation 1
        
        self.allPerm = self.allPerm.astype(int) 
        
        if ret:
            if verbose:
                print("Arrays returned: wrongPerm, allPerm")

            return self.wrongPerm, self.allPerm
        else:
            return None

    def invMassCalc(self, ret = False, verbose = False):
        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Calculating the invariant mass for all permutations" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        num = self.min_dRgenFinal.shape[0]
        self.invariantMass = np.ndarray((num, 3, 3))
        #invariantMass = np.ndarray((selMupTFinal.shape[0], permutationLabels.shape[0], permutationLabels.shape[1])
        
        for event in tqdm(range(num)): # Loop over events
            
            # Correct permutation indices
            A0_c = np.copy(self.min_dRgenFinal[event, 0:2])
            A1_c = np.copy(self.min_dRgenFinal[event, 2:4])
            
            # Incorrect permutation indices
            A0_w0 = np.copy(self.wrongPerm[event, 0, 0:2])
            A1_w0 = np.copy(self.wrongPerm[event, 0, 2:4])
            
            A0_w1 = np.copy(self.wrongPerm[event, 1, 0:2])
            A1_w1 = np.copy(self.wrongPerm[event, 1, 2:4])

            indicesA_c = np.column_stack((A0_c, A1_c))
            indicesA_c = indicesA_c.astype(int)
            
            indicesA_w0 = np.column_stack((A0_w0, A1_w0))
            indicesA_w0 = indicesA_w0.astype(int)
            
            indicesA_w1 = np.column_stack((A0_w1, A1_w1))
            indicesA_w1 = indicesA_w1.astype(int)
            
            # Place labels in invariant mass array
            self.invariantMass[event, 0, 2] = 1 # Correct permutation
            self.invariantMass[event, 1, 2] = 0 # Incorrect permutation
            self.invariantMass[event, 2, 2] = 0 # Incorrect permutation
        
            ##### CORRECT PERMUTATION #####
            # Calculation for A0
            self.invariantMass[event, 0, 0] = np.sqrt( (2 * self.selMu_ptFinal[event, indicesA_c[0, 0]] *  self.selMu_ptFinal[event, indicesA_c[1, 0]]) *   
                                                                                                   (np.cosh(self.selMu_etaFinal[event, indicesA_c[0, 0]] - self.selMu_etaFinal[event, indicesA_c[1, 0]]) - 
                                                                                                    np.cos(self.selMu_phiFinal[event, indicesA_c[0, 0]] - self.selMu_phiFinal[event, indicesA_c[1, 0]])))
            # Calculation for A1 
            self.invariantMass[event, 0, 1] = np.sqrt( (2 * self.selMu_ptFinal[event, indicesA_c[0, 1]] *  self.selMu_ptFinal[event, indicesA_c[1, 1]]) *   
                                                                                                   (np.cosh(self.selMu_etaFinal[event, indicesA_c[0, 1]] - self.selMu_etaFinal[event, indicesA_c[1, 1]]) - 
                                                                                                    np.cos(self.selMu_phiFinal[event, indicesA_c[0, 1]] - self.selMu_phiFinal[event, indicesA_c[1, 1]])))    
            
            ##### WRONG PERMUTATIONS #####
            # Calculation for A0
            self.invariantMass[event, 1, 0] = np.sqrt( (2 * self.selMu_ptFinal[event, indicesA_w0[0, 0]] *  self.selMu_ptFinal[event, indicesA_w0[1, 0]]) *   
                                                                                                   (np.cosh(self.selMu_etaFinal[event, indicesA_w0[0, 0]] - self.selMu_etaFinal[event, indicesA_w0[1, 0]]) - 
                                                                                                    np.cos(self.selMu_phiFinal[event, indicesA_w0[0, 0]] - self.selMu_phiFinal[event, indicesA_w0[1, 0]])))
            # Calculation for A1 
            self.invariantMass[event, 1, 1] = np.sqrt( (2 * self.selMu_ptFinal[event, indicesA_w0[0, 1]] *  self.selMu_ptFinal[event, indicesA_w0[1, 1]]) *   
                                                                                                   (np.cosh(self.selMu_etaFinal[event, indicesA_w0[0, 1]] - self.selMu_etaFinal[event, indicesA_w0[1, 1]]) - 
                                                                                                    np.cos(self.selMu_phiFinal[event, indicesA_w0[0, 1]] - self.selMu_phiFinal[event, indicesA_w0[1, 1]])))    
            
             # Calculation for A0
            self.invariantMass[event, 2, 0] = np.sqrt( (2 * self.selMu_ptFinal[event, indicesA_w1[0, 0]] *  self.selMu_ptFinal[event, indicesA_w1[1, 0]]) *   
                                                                                                   (np.cosh(self.selMu_etaFinal[event, indicesA_w1[0, 0]] - self.selMu_etaFinal[event, indicesA_w1[1, 0]]) - 
                                                                                                    np.cos(self.selMu_phiFinal[event, indicesA_w1[0, 0]] - self.selMu_phiFinal[event, indicesA_w1[1, 0]])))
            # Calculation for A1 
            self.invariantMass[event, 2, 1] = np.sqrt( (2 * self.selMu_ptFinal[event, indicesA_w1[0, 1]] *  self.selMu_ptFinal[event, indicesA_w1[1, 1]]) *   
                                                                                                   (np.cosh(self.selMu_etaFinal[event, indicesA_w1[0, 1]] -self.selMu_etaFinal[event, indicesA_w1[1, 1]]) - 
                                                                                                    np.cos(self.selMu_phiFinal[event, indicesA_w1[0, 1]] - self.selMu_phiFinal[event, indicesA_w1[1, 1]])))

        if ret:
            return self.invariantMass
        else:
            return None

    def dR_diMu(self, ret = False, verbose = False):
        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Calculating dPhi and dR for all permutations" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        num = self.min_dRgenFinal.shape[0]
        self.diMu_dR = np.ndarray((num, 3, 2))
        self.dPhi = np.ndarray((num, 3, 2))

        for event in tqdm(range(num)):
        ## correct dR A0
            A0_c = np.copy(self.min_dRgenFinal[event, 0:2])
            A1_c = np.copy(self.min_dRgenFinal[event, 2:4])

            # Incorrect permutation indices
            A0_w0 = np.copy(self.wrongPerm[event, 0, 0:2])
            A1_w0 = np.copy(self.wrongPerm[event, 0, 2:4])

            A0_w1 = np.copy(self.wrongPerm[event, 1, 0:2])
            A1_w1 = np.copy(self.wrongPerm[event, 1, 2:4])

            indicesA_c = np.column_stack((A0_c, A1_c))
            indicesA_c = indicesA_c.astype(int)

            indicesA_w0 = np.column_stack((A0_w0, A1_w0))
            indicesA_w0 = indicesA_w0.astype(int)

            indicesA_w1 = np.column_stack((A0_w1, A1_w1))
            indicesA_w1 = indicesA_w1.astype(int)

            self.dPhi[event, 0, 0] = self.selMu_phiFinal[event, indicesA_c[0, 0]]  - self.selMu_phiFinal[event, indicesA_c[1, 0]]
            self.dPhi[event, 0, 1] = self.selMu_phiFinal[event, indicesA_c[0, 1]]  - self.selMu_phiFinal[event, indicesA_c[1, 1]]
            self.dPhi[event, 1, 0] = self.selMu_phiFinal[event, indicesA_w0[0, 0]] - self.selMu_phiFinal[event, indicesA_w0[1, 0]]
            self.dPhi[event, 1, 1] = self.selMu_phiFinal[event, indicesA_w0[0, 1]] - self.selMu_phiFinal[event, indicesA_w0[1, 1]]
            self.dPhi[event, 2, 0] = self.selMu_phiFinal[event, indicesA_w1[0, 0]] - self.selMu_phiFinal[event, indicesA_w1[1, 0]]
            self.dPhi[event, 2, 1] = self.selMu_phiFinal[event, indicesA_w1[0, 1]] - self.selMu_phiFinal[event, indicesA_w1[1, 1]]


            ## correct dR A0
            self.diMu_dR[event, 0, 0] = sqrt(pow(self.selMu_etaFinal[event, indicesA_c[0, 0]] - self.selMu_etaFinal[event, indicesA_c[1, 0]],2)+ pow(self.dPhi[event,0,0], 2))
            ## correct dR A1
            self.diMu_dR[event, 0 ,1] = sqrt(pow(self.selMu_etaFinal[event, indicesA_c[0, 1]] - self.selMu_etaFinal[event, indicesA_c[1, 1]],2)+ pow(self.dPhi[event,0,1], 2))

            ## wrong0 dR A0
            self.diMu_dR[event, 1, 0] = sqrt(pow(self.selMu_etaFinal[event, indicesA_w0[0, 0]] - self.selMu_etaFinal[event, indicesA_w0[1, 0]],2)+ pow(self.dPhi[event,1,0], 2))
            ## wrong0 dR A1
            self.diMu_dR[event, 1, 1] = sqrt(pow(self.selMu_etaFinal[event, indicesA_w0[0, 1]] - self.selMu_etaFinal[event, indicesA_w0[1, 1]],2)+ pow(self.dPhi[event,1,1], 2))

            ## wrong1 dR A0
            self.diMu_dR[event, 2, 0] = sqrt(pow(self.selMu_etaFinal[event, indicesA_w1[0, 0]] - self.selMu_etaFinal[event, indicesA_w1[1, 0]],2)+ pow(self.dPhi[event,2,0], 2))
            ## wrong1 dR A1
            self.diMu_dR[event, 2, 1] = sqrt(pow(self.selMu_etaFinal[event, indicesA_w1[0, 1]] - self.selMu_etaFinal[event, indicesA_w1[1, 1]],2)+ pow(self.dPhi[event,2,1], 2))
        if ret:   
            return self.diMu_dR, self.dPhi
        else:
            return None

    def fillAndSort(self, sort = True, sep = True, ret = True, pandas = True, total = False):

        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Filling/sorting the final array for background data" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        events   = self.min_dRgenFinal.shape[0]
        EventNum = np.ndarray((events, 3)) 
        counter  = 0

        for event in range(events):
            for ii in range(3):
                EventNum[event, 0] = counter  

            counter += 1

        EventNum = EventNum.astype(int)
        
        
        # All data plus event numbers
        dataframe = np.ndarray((events, 3, 24)) #3199 * 3 * 22

        for event in tqdm(range(self.selMu_ptFinal.shape[0])):

            for permutation in range(self.invariantMass.shape[1]):

                EventNum_temp  = np.copy(EventNum[event, permutation]).reshape((1,))
                selpT_temp     = np.copy(self.selMu_ptFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                selEta_temp    = np.copy(self.selMu_etaFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                selPhi_temp    = np.copy(self.selMu_phiFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                selCharge_temp = np.copy(self.selMu_chargeFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                invMass_temp   = np.copy(self.invariantMass[event, permutation, :]) # 1 x 3  
                dPhi_temp      = np.copy(self.dPhi[event, permutation, :])
                diMu_dR_temp   = np.copy(self.diMu_dR[event, permutation, :]) # 1 x 2
                
                dataframe[event, permutation, :]= np.concatenate((selpT_temp, selEta_temp, selPhi_temp, selCharge_temp, dPhi_temp, diMu_dR_temp, EventNum_temp, invMass_temp)) # 1 x 19


        #flatten the df
        dataframe_shaped = np.reshape(dataframe, (self.selMu_ptFinal.shape[0] * self.invariantMass.shape[1], 24))
        if self.dataset == "mc":
            return dataframe_shaped

        elif self.dataset == "bkg":
            if sep:
                X_data = dataframe_shaped[:, 0:23]   
                Y_data = dataframe_shaped[:, 20:24]

                if pandas:
                    self.X_df = pd.DataFrame(X_data, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3", "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1", "dRA0", "dRA1", "event", "invMassA0", "invMassA1"])
                    self.Y_df = pd.DataFrame(Y_data, columns=['event', 'invmA0', 'invmA1', 'pair'])

                    if sort:
                        self.X_df_sorted = self.X_df.sort_values('event')

            if total:
                self.total_df = pd.DataFrame(dataframe_shaped, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3", "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "event", "invMassA0", "invMassA1", "Label"])

            if ret:
                print("Returning sorted X data and Y data")
                return self.X_df_sorted, self.Y_df

    def fillFinalArray(self, numbering = True, perm = True, diMu_dRBool = True, pandas = False, ret = True, verbose = False):
        print("\n\n")
        print(60 * '*')
        print(colors.GREEN + "Filling the final array for MC data" + colors.ENDC)
        print(60 * '*')
        #print("\n\n")

        if verbose:
            if pandas:
                print("Filling final array/dataframe for use with XGBoost\n")

        if diMu_dRBool == True:
            dataframe = np.ndarray((self.selMu_ptFinal.shape[0], self.invariantMass.shape[1], 23))
        else:
            dataframe = np.ndarray((self.selMu_ptFinal.shape[0], self.invariantMass.shape[1], 19))
        #print(dataframe.shape)
        
        if perm == False and diMu_dRBool == True:
            for event in tqdm(range(self.selMu_ptFinal.shape[0])):
                for permutation in range(self.invariantMass.shape[1]):
                    # Remove row of data for each sel muon
                    selMu_ptTemp     = np.copy(self.selMu_ptFinal[event, :]) # 1 x 4
                    selMu_etaTemp    = np.copy(self.selMu_etaFinal[event, :]) # 1 x 4
                    selMu_phiTemp    = np.copy(self.selMu_phiFinal[event, :]) # 1 x 4
                    selMu_chargeTemp = np.copy(self.selMu_chargeFinal[event, :]) # 1 x 4
                    invMass_temp     = np.copy(self.invariantMass[event, permutation, :]) # 1 x 3 
                    dPhi_temp        = np.copy(self.dPhi[event, permutation, :])
                    diMu_dR_temp     = np.copy(self.diMu_dR[event, permutation, :]) # 1 x 2
                
                    dataframe[event, permutation, :] = np.concatenate((selMu_ptTemp, selMu_etaTemp, selMu_phiTemp, selMu_chargeTemp, dPhi_temp, diMu_dR_temp, invMass_temp))

        if perm == True and diMu_dRBool == False:
            for event in tqdm(range(self.selMu_ptFinal.shape[0])):
                for permutation in range(self.invariantMass.shape[1]):
                    selMu_ptTemp     = np.copy(self.selMu_ptFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                    selMu_etaTemp    = np.copy(self.selMu_etaFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                    selMu_phiTemp    = np.copy(self.selMu_phiFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                    selMu_chargeTemp = np.copy(self.selMu_chargeFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                    invMass_temp     = np.copy(self.invariantMass[event, permutation, :]) # 1 x 3 

                    dataframe[event, permutation, :] = np.concatenate((selMu_ptTemp, selMu_etaTemp, selMu_phiTemp, selMu_chargeTemp, invMass_temp))
        
        if perm == True and diMu_dRBool == True:
            eventNum = np.ndarray((self.dRgenFinal.shape[0], 3)) #3199 * 3 * 22
            counter  = 0

            for event in range(self.dRgenFinal.shape[0]):

                EventNum[event, 0] = counter
                EventNum[event, 1] = counter
                EventNum[event, 2] = counter
                counter += 1

            eventNum = eventNum.astype(int)

            for event in tqdm(range(self.selMu_ptFinal.shape[0])):
                
                for permutation in range(self.invariantMass.shape[1]):
                    eventNum_temp    = np.copy(EventNum[event, permutation]).reshape((1,))
                    selMu_ptTemp     = np.copy(self.selMu_ptFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                    selMu_etaTemp    = np.copy(self.selMu_etaFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                    selMu_phiTemp    = np.copy(self.selMu_phiFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                    selMu_chargeTemp = np.copy(self.selMu_chargeFinal[event, self.allPerm[event, permutation]]) # 1 x 4
                    invMass_temp     = np.copy(self.invariantMass[event, permutation, :]) # 1 x 3 
                    dPhi_temp        = np.copy(self.dPhi[event, permutation, :])
                    diMu_dR_temp     = np.copy(self.diMu_dR[event, permutation, :]) # 1 x 2

                    dataframe[event, permutation, :] = np.concatenate((selMu_ptTemp, selMu_etaTemp, selMu_phiTemp, selMu_chargeTemp, dPhi_temp, diMu_dR_temp, eventNum_temp, invMass_temp))

        if perm == False and diMu_dRBool == False:
            for event in tqdm(range(self.selMu_ptFinal.shape[0])):
                for permutation in range(self.invariantMass.shape[1]):
                    # Remove row of data for each sel muon
                    selMu_ptTemp     = np.copy(self.selMu_ptFinal[event, :]) # 1 x 4
                    selMu_etaTemp    = np.copy(self.selMu_etaFinal[event, :]) # 1 x 4
                    selMu_phiTemp    = np.copy(self.selMu_phiFinal[event, :]) # 1 x 4
                    selMu_chargeTemp = np.copy(self.selMu_chargeFinal[event, :]) # 1 x 4
                    invMass_temp     = np.copy(self.invariantMass[event, permutation, :]) # 1 x 3 
                    
                    dataframe[event, permutation, :] = np.concatenate((selMu_ptTemp, selMu_etaTemp, selMu_phiTemp, selMu_chargeTemp, invMass_temp)) # 1 x 19
        

        dataframe_shaped = np.reshape(dataframe, (self.selMu_ptCut.shape[0] * self.invariantMass.shape[1], 24))
        total_df = pd.DataFrame(dataframe_shaped, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
          "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "event", "invMassA0",
          "invMassA1", "pair"])

        
        '''
        if diMu_dRBool == True:
            self.dataframe_shaped = np.reshape(dataframe, (self.selMu_ptFinal.shape[0] * self.invariantMass.shape[1], 23))
        else:
            self.dataframe_shaped = np.reshape(dataframe, (self.selMu_ptFinal.shape[0] * self.invariantMass.shape[1], 19))
        #print('Shape of output data: {}'.format(dataframe_shaped.shape))
        
        if pandas == True:
            columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
              "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dRA0", "dRA1", "invMassA0",
              "invMassA1","label"]
            self.df = pd.DataFrame(data=self.dataframe_shaped, columns=columns)
            
            if ret:
                return self.dataframe_shaped, self.df
            else:
                return None
        else:
            if ret:
                return self.dataframe_shaped
            else:
            return None

        '''
    #def plot(self):
    
