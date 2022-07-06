#!/bin/env python

'''
Various plotting tools the fD model muon matching project

Stephen D. Butalla & Mehdi Rahmani
2021/06/01

'''

from uproot import * # Transfer ROOT data into np arrays
from uproot import open as openUp
import numpy as np

# Accepts path to root file and extracts data

def extractData(rootFile):

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
    genAMu_phi = np.column_stack((genA0Mu0_phi, genA0Mu1_phi, genA1Mu0_phi, genA1Mu1_phi))
    
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
    
    print('Arrays returned: selMu_eta, selMu_phi, selMu_pt, selMu_charge, genAMu_eta, genAMu_phi, genAMu_pt, genAMu_charge')

    return selMu_eta, selMu_phi, selMu_pt, selMu_charge, genAMu_eta, genAMu_phi, genAMu_pt, genAMu_charge

def prelimCuts(selMu_phi, selMu_eta, selMu_pt, selMu_charge, genAMu_eta, verbose = True):

    badPhi = np.unique(np.where(selMu_phi == -100)[0])
    badSelEta = np.unique(np.where(abs(selMu_eta) > 2.4)[0])
    badGenAEta = np.unique(np.where(abs(genAMu_eta) > 2.4)[0])
    badSelpT = np.unique(np.where(selMu_pt == -100)[0])
    badSelCharge = np.unique(np.where(np.sum(selMu_charge, axis = 1) != 0))
    
    # Convert to lists so we can add without broadcasting problems
    badPhi    = list(badPhi)
    badSelEta = list(badSelEta)
    badGenAEta = list(badGenAEta)
    badSelpT  = list(badSelpT)
    badCharge = list(badSelCharge)
    
    badEvents = sorted(np.unique(badPhi + badGenAEta + badSelEta + badSelpT + badCharge)) # Add lists, return unique values, and sort to preserve order
    
    if verbose == True:
        print('********** CUT INFO **********\n')
        print('Total number of events failing reconstruction in phi: {}'.format(len(badPhi)))
        print('Total number of events with sel eta > 2.4: {}'.format(len(badSelEta)))
        print('Total number of events with gen eta > 2.4: {}'.format(len(badGenAEta)))
        print('Total number of events with sel pT == -100: {}'.format(len(badSelpT)))
        print('Total number of events failing charge reconstruction: {}'.format(len(badCharge)))
        print('Total number of bad events: {}'.format(len(badEvents)))
        print('\n ******************************')
        
    return badEvents

def removeBadEvents(badEvents, data):
    '''
    Removes bad events given the data and list of indices

    badEvents:  list of indices to remove
    data:      
    '''
    return np.delete(data, badEvents, axis=0)

def dRcut(dRgenMatched, cut):
    return np.unique(np.where(dRgenMatched > cut)[0])

