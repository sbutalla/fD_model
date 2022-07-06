#!/bin/env python

'''
Various plotting tools the fD model muon matching project

Stephen D. Butalla & Mehdi Rahmani
2021/06/01

'''

from uproot import open as openUp # Transfer ROOT data into np arrays
import numpy as np

################################################################
################################################################
################## fD Data Proessing Tools #####################
################################################################
################################################################

class processData:

    def __init__(self):
        return None

    def extractData(self, rootFile, root_dir):
        '''
        Accepts absolute (or relative) path to a ROOT file and
        returns simulated signal event data.
        '''

        file = openUp(rootFile)
        #data = file['cutFlowAnalyzerPXBL4PXFL3;1/Events;1']
        data = file[root_dir]
        
        selMu0_eta = np.asarray(data['selMu0_eta'].array())
        selMu1_eta = np.asarray(data['selMu1_eta'].array())
        selMu2_eta = np.asarray(data['selMu2_eta'].array())
        selMu3_eta = np.asarray(data['selMu3_eta'].array())
        self.selMu_eta  = np.column_stack((selMu0_eta, selMu1_eta, selMu2_eta, selMu3_eta))

        selMu0_phi = np.asarray(data['selMu0_phi'].array())
        selMu1_phi = np.asarray(data['selMu1_phi'].array())
        selMu2_phi = np.asarray(data['selMu2_phi'].array())
        selMu3_phi = np.asarray(data['selMu3_phi'].array())
        self.selMu_phi  = np.column_stack((selMu0_phi, selMu1_phi, selMu2_phi, selMu3_phi))

        selMu0_pt = np.asarray(data['selMu0_pT'].array())
        selMu1_pt = np.asarray(data['selMu1_pT'].array())
        selMu2_pt = np.asarray(data['selMu2_pT'].array())
        selMu3_pt = np.asarray(data['selMu3_pT'].array())
        self.selMu_pt  = np.column_stack((selMu0_pt, selMu1_pt, selMu2_pt, selMu3_pt))
        
        selMu0_charge = np.asarray(data['selMu0_charge'].array())
        selMu1_charge = np.asarray(data['selMu1_charge'].array())
        selMu2_charge = np.asarray(data['selMu2_charge'].array())
        selMu3_charge = np.asarray(data['selMu3_charge'].array())
        self.selMu_charge  = np.column_stack((selMu0_charge, selMu1_charge, selMu2_charge, selMu3_charge))

        genA0Mu0_eta = np.asarray(data['genA0Mu0_eta'].array())
        genA0Mu1_eta = np.asarray(data['genA0Mu1_eta'].array())
        genA1Mu0_eta = np.asarray(data['genA1Mu0_eta'].array())
        genA1Mu1_eta = np.asarray(data['genA1Mu1_eta'].array())
        self.genAMu_eta   = np.column_stack((genA0Mu0_eta, genA0Mu1_eta, genA1Mu0_eta, genA1Mu1_eta))

        genA0Mu0_phi = np.asarray(data['genA0Mu0_phi'].array())
        genA0Mu1_phi = np.asarray(data['genA0Mu1_phi'].array())
        genA1Mu0_phi = np.asarray(data['genA1Mu0_phi'].array())
        genA1Mu1_phi = np.asarray(data['genA1Mu1_phi'].array())
        self.genAMu_phi = np.column_stack((genA0Mu0_phi, genA0Mu1_phi, genA1Mu0_phi, genA1Mu1_phi))
        
        genA0Mu0_pt = np.asarray(data['genA0Mu0_pt'].array())
        genA0Mu1_pt = np.asarray(data['genA0Mu1_pt'].array())
        genA1Mu0_pt = np.asarray(data['genA1Mu0_pt'].array())
        genA1Mu1_pt = np.asarray(data['genA1Mu1_pt'].array())
        self.genAMu_pt   = np.column_stack((genA0Mu0_pt, genA0Mu1_pt, genA1Mu0_pt, genA1Mu1_pt))

        genA0Mu0_charge = np.asarray(data['genA0Mu0_charge'].array())
        genA0Mu1_charge = np.asarray(data['genA0Mu1_charge'].array())
        genA1Mu0_charge = np.asarray(data['genA1Mu0_charge'].array())
        genA1Mu1_charge = np.asarray(data['genA1Mu1_charge'].array())
        self.genAMu_charge   = np.column_stack((genA0Mu0_charge, genA0Mu1_charge, genA1Mu0_charge, genA1Mu1_charge))
        
        print('Arrays returned: selMu_eta, selMu_phi, selMu_pt, selMu_charge, genAMu_eta, genAMu_phi, genAMu_pt, genAMu_charge')

        return self.selMu_eta, self.selMu_phi, self.selMu_pt, self.selMu_charge, self.genAMu_eta, self.genAMu_phi, self.genAMu_pt, self.genAMu_charge

    def prelimCuts(self, verbose = True):

        badPhi = np.unique(np.where(self.selMu_phi == -100)[0])
        badSelEta = np.unique(np.where(abs(self.selMu_eta) > 2.4)[0])
        badGenAEta = np.unique(np.where(abs(self.genAMu_eta) > 2.4)[0])
        badSelpT = np.unique(np.where(self.selMu_pt == -100)[0])
        badSelCharge = np.unique(np.where(np.sum(self.selMu_charge, axis = 1) != 0))
        
        # Convert to lists so we can add without broadcasting problems
        badPhi    = list(badPhi)
        badSelEta = list(badSelEta)
        badGenAEta = list(badGenAEta)
        badSelpT  = list(badSelpT)
        badCharge = list(badSelCharge)
        
        self.badEvents = sorted(np.unique(badPhi + badGenAEta + badSelEta + badSelpT + badCharge)) # Add lists, return unique values, and sort to preserve order
        
        if verbose == True:
            print('********** CUT INFO **********\n')
            print('Total number of events failing reconstruction in phi: {}'.format(len(badPhi)))
            print('Total number of events with sel eta > 2.4: {}'.format(len(badSelEta)))
            print('Total number of events with gen eta > 2.4: {}'.format(len(badGenAEta)))
            print('Total number of events with sel pT == -100: {}'.format(len(badSelpT)))
            print('Total number of events failing charge reconstruction: {}'.format(len(badCharge)))
            print('Total number of bad events: {}'.format(len(self.badEvents)))
            print('\n ******************************')
            
        return self.badEvents

    def removeBadEvents(self, cut):
        '''
        Removes bad events given the data and list of indices

        cut: Which observable to apply the cut
        '''
        if cut == "selMu_eta":
            self.selMu_etaCut = np.delete(self.selMu_eta, self.badEvents, axis=0)
            return self.selMu_etaCut
        elif cut == "selMu_phi":
            self.selMu_phiCut = np.delete(self.selMu_phi, self.badEvents, axis=0)
        elif cut == "all":
            self.selMu_etaCut = np.delete(self.selMu_eta, self.badEvents, axis=0)
            self.selMu_phiCut = np.delete(self.selMu_phi, self.badEvents, axis=0)
            

    def dRcut(self, dRgenMatched, cut):
        return np.unique(np.where(dRgenMatched > cut)[0])




    ################################################################
    ################################################################
    ############## Background Data Proessing Tools #################
    ################################################################
    ################################################################

    def extractBkgData(self, rootFile):
        '''
        Accepts absolute (or relative) path to a ROOT file and
        returns background event data.
        '''

        file = openUp(rootFile)
        data = file['cutFlowAnalyzerPXBL4PXFL3;1/Events;1']

        selMu0_eta = np.asarray(data['selMu0_eta'].array())
        selMu1_eta = np.asarray(data['selMu1_eta'].array())
        selMu2_eta = np.asarray(data['selMu2_eta'].array())
        selMu3_eta = np.asarray(data['selMu3_eta'].array())
        selMu_eta  = np.column_stack((selMu0_eta, selMu1_eta, selMu2_eta, selMu3_eta))

        selMu0_phi = np.asarray(data['selMu0_phi'].array())
        selMu1_phi = np.asarray(data['selMu1_phi'].array())
        selMu2_phi = np.asarray(data['selMu2_phi'].array())
        selMu3_phi = np.asarray(data['selMu3_phi'].array())
        selMu_phi  = np.column_stack((selMu0_phi, selMu1_phi, selMu2_phi, selMu3_phi))

        selMu0_pt = np.asarray(data['selMu0_pT'].array())
        selMu1_pt = np.asarray(data['selMu1_pT'].array())
        selMu2_pt = np.asarray(data['selMu2_pT'].array())
        selMu3_pt = np.asarray(data['selMu3_pT'].array())
        selMu_pt  = np.column_stack((selMu0_pt, selMu1_pt, selMu2_pt, selMu3_pt))

        selMu0_charge = np.asarray(data['selMu0_charge'].array())
        selMu1_charge = np.asarray(data['selMu1_charge'].array())
        selMu2_charge = np.asarray(data['selMu2_charge'].array())
        selMu3_charge = np.asarray(data['selMu3_charge'].array())
        selMu_charge  = np.column_stack((selMu0_charge, selMu1_charge, selMu2_charge, selMu3_charge))

        genA0Mu0_eta = np.asarray(data['genA0Mu0_eta'].array())
        genA0Mu1_eta = np.asarray(data['genA0Mu1_eta'].array())
        genA1Mu0_eta = np.asarray(data['genA1Mu0_eta'].array())
        genA1Mu1_eta = np.asarray(data['genA1Mu1_eta'].array())
        genAMu_eta   = np.column_stack((genA0Mu0_eta, genA0Mu1_eta, genA1Mu0_eta, genA1Mu1_eta))

        genA0Mu0_phi = np.asarray(data['genA0Mu0_phi'].array())
        genA0Mu1_phi = np.asarray(data['genA0Mu1_phi'].array())
        genA1Mu0_phi = np.asarray(data['genA1Mu0_phi'].array())
        genA1Mu1_phi = np.asarray(data['genA1Mu1_phi'].array())
        genAMu_phi   = np.column_stack((genA0Mu0_phi, genA0Mu1_phi, genA1Mu0_phi, genA1Mu1_phi))

        genA0Mu0_pt = np.asarray(data['genA0Mu0_pt'].array())
        genA0Mu1_pt = np.asarray(data['genA0Mu1_pt'].array())
        genA1Mu0_pt = np.asarray(data['genA1Mu0_pt'].array())
        genA1Mu1_pt = np.asarray(data['genA1Mu1_pt'].array())
        genAMu_pt   = np.column_stack((genA0Mu0_pt, genA0Mu1_pt, genA1Mu0_pt, genA1Mu1_pt))

        genA0Mu0_charge = np.asarray(data['genA0Mu0_charge'].array())
        genA0Mu1_charge = np.asarray(data['genA0Mu1_charge'].array())
        genA1Mu0_charge = np.asarray(data['genA1Mu0_charge'].array())
        genA1Mu1_charge = np.asarray(data['genA1Mu1_charge'].array())
        genAMu_charge   = np.column_stack((genA0Mu0_charge, genA0Mu1_charge, genA1Mu0_charge, genA1Mu1_charge))

        diMuonC = np.asarray(data['diMuonC_FittedVtx_m'].array())

        diMuonF = np.asarray(data['diMuonF_FittedVtx_m'].array())

        return selMu_eta, selMu_phi, selMu_pt, selMu_charge, genAMu_eta, genAMu_phi, genAMu_pt, genAMu_charge, diMuonC, diMuonF


    def prelimBkgCuts(diMuonC, diMuonF, selMu_phi, selMu_eta, selMu_pt, selMu_charge, verbose = True):

        badInvMassMuC = np.unique(np.where(diMuonC == -1000)[0])
        badInvMassMuF = np.unique(np.where(diMuonF == -1000)[0])
        badSelPhi = np.unique(np.where(selMu_phi  == -100)[0])
        badSelEta = np.unique(np.where(abs(selMu_eta) > 2.4)[0])
        badSelpT = np.unique(np.where(selMu_pt  == -100)[0])
        badSelCharge = np.unique(np.where(np.sum(selMu_charge, axis = 1) != 0))

        badInvMassMuC = list(badInvMassMuC)
        badInvMassMuF = list(badInvMassMuF)
        badSelPhi     = list(badSelPhi)
        badSelEta     = list(badSelEta)
        badSelpT      = list(badSelpT)
        badSelCharge  = list(badSelCharge)
        
        badEvents = sorted(np.unique(badSelPhi + badSelEta + badSelpT + badSelCharge + badInvMassMuC + badInvMassMuF))
            
        if verbose == True:
            print('*' * 48)
            print('********* CUT INFO FOR BACKGROUND DATA *********\n')
            print('*' * 48)
            print('Total number of events failing invariant mass reconstruction (diMuon C): {}'.format(len(badInvMassMuC)))
            print('Total number of events failing invariant mass reconstruction (diMuon F): {}'.format(len(badInvMassMuF)))
            print('Total number of events with bad phi: {}'.format(len(badSelPhi)))
            print('Total number of events with |eta| > 2.4: {}'.format(len(badSelEta)))
            print('Total number of events with sel pT == -100: {}'.format(len(badSelpT)))
            print('Total number of events failing charge reconstruction: {}'.format(len(badSelCharge)))
            print('Total number of bad events: {}'.format(len(badEvents)))
            print('\n ******************************')
            
        return badEvents 


