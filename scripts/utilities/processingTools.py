#!/bin/env python

'''
Various plotting tools the fD model muon matching project

Stephen D. Butalla & Mehdi Rahmani
2021/06/01

'''

from uproot import open as openUp # Transfer ROOT data into np arrays
import numpy as np
from tqdm import tqdm

################################################################
################################################################
################## fD Data Proessing Tools #####################
################################################################
################################################################

def extractData(rootFile):
    '''
    Accepts absolute (or relative) path to a ROOT file and
    returns simulated signal event data.
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

################################################################
################################################################
############## Background Data Proessing Tools #################
################################################################
################################################################

def extractBkgData(rootFile):
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
    badSelCharge     = list(badSelCharge)
    
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

def invMassCalc(min_dRgenFinal, wrongPerm, selpTFinal, selEtaFinal, selPhiFinal):
    
    invariantMass = np.ndarray((min_dRgenFinal.shape[0], 3, 3))
    #invariantMass = np.ndarray((selMupTFinal.shape[0], permutationLabels.shape[0], permutationLabels.shape[1])
    
    for event in tqdm(range(min_dRgenFinal.shape[0])): # Loop over events
        
        # Correct permutation indices
        A0_c = np.copy(min_dRgenFinal[event, 0:2])
        A1_c = np.copy(min_dRgenFinal[event, 2:4])
        
        # Incorrect permutation indices
        A0_w0 = np.copy(wrongPerm[event, 0, 0:2])
        A1_w0 = np.copy(wrongPerm[event, 0, 2:4])
        
        A0_w1 = np.copy(wrongPerm[event, 1, 0:2])
        A1_w1 = np.copy(wrongPerm[event, 1, 2:4])

        indicesA_c = np.column_stack((A0_c, A1_c))
        indicesA_c = indicesA_c.astype(int)
        
        indicesA_w0 = np.column_stack((A0_w0, A1_w0))
        indicesA_w0 = indicesA_w0.astype(int)
        
        indicesA_w1 = np.column_stack((A0_w1, A1_w1))
        indicesA_w1 = indicesA_w1.astype(int)
        
        # Place labels in invariant mass array
        invariantMass[event, 0, 2] = 1 # Correct permutation
        invariantMass[event, 1, 2] = 0 # Incorrect permutation
        invariantMass[event, 2, 2] = 0 # Incorrect permutation
    
        ##### CORRECT PERMUTATION #####
        # Calculation for A0
        invariantMass[event, 0, 0] = np.sqrt( (2 * selpTFinal[event, indicesA_c[0, 0]] *  selpTFinal[event, indicesA_c[1, 0]]) *   
                                                                                               (np.cosh(selEtaFinal[event, indicesA_c[0, 0]] - selEtaFinal[event, indicesA_c[1, 0]]) - 
                                                                                                np.cos(selPhiFinal[event, indicesA_c[0, 0]] - selPhiFinal[event, indicesA_c[1, 0]])))
        # Calculation for A1 
        invariantMass[event, 0, 1] = np.sqrt( (2 * selpTFinal[event, indicesA_c[0, 1]] *  selpTFinal[event, indicesA_c[1, 1]]) *   
                                                                                               (np.cosh(selEtaFinal[event, indicesA_c[0, 1]] - selEtaFinal[event, indicesA_c[1, 1]]) - 
                                                                                                np.cos(selPhiFinal[event, indicesA_c[0, 1]] - selPhiFinal[event, indicesA_c[1, 1]])))    
        
        ##### WRONG PERMUTATIONS #####
        # Calculation for A0
        invariantMass[event, 1, 0] = np.sqrt( (2 * selpTFinal[event, indicesA_w0[0, 0]] *  selpTFinal[event, indicesA_w0[1, 0]]) *   
                                                                                               (np.cosh(selEtaFinal[event, indicesA_w0[0, 0]] - selEtaFinal[event, indicesA_w0[1, 0]]) - 
                                                                                                np.cos(selPhiFinal[event, indicesA_w0[0, 0]] - selPhiFinal[event, indicesA_w0[1, 0]])))
        # Calculation for A1 
        invariantMass[event, 1, 1] = np.sqrt( (2 * selpTFinal[event, indicesA_w0[0, 1]] *  selpTFinal[event, indicesA_w0[1, 1]]) *   
                                                                                               (np.cosh(selEtaFinal[event, indicesA_w0[0, 1]] - selEtaFinal[event, indicesA_w0[1, 1]]) - 
                                                                                                np.cos(selPhiFinal[event, indicesA_w0[0, 1]] - selPhiFinal[event, indicesA_w0[1, 1]])))    
        
         # Calculation for A0
        invariantMass[event, 2, 0] = np.sqrt( (2 * selpTFinal[event, indicesA_w1[0, 0]] *  selpTFinal[event, indicesA_w1[1, 0]]) *   
                                                                                               (np.cosh(selEtaFinal[event, indicesA_w1[0, 0]] - selEtaFinal[event, indicesA_w1[1, 0]]) - 
                                                                                                np.cos(selPhiFinal[event, indicesA_w1[0, 0]] - selPhiFinal[event, indicesA_w1[1, 0]])))
        # Calculation for A1 
        invariantMass[event, 2, 1] = np.sqrt( (2 * selpTFinal[event, indicesA_w1[0, 1]] *  selpTFinal[event, indicesA_w1[1, 1]]) *   
                                                                                               (np.cosh(selEtaFinal[event, indicesA_w1[0, 1]] - selEtaFinal[event, indicesA_w1[1, 1]]) - 
                                                                                                np.cos(selPhiFinal[event, indicesA_w1[0, 1]] - selPhiFinal[event, indicesA_w1[1, 1]])))    
        
    
    
    return invariantMass

def dR_diMu(min_dRgenFinal, wrongPerm):

    diMu_dR = np.ndarray((min_dRgenFinal.shape[0], 3, 2))
    for event in range (min_dRgenFinal.shape[0]):
    ## correct dR A0
        A0_c = np.copy(min_dRgenFinal[event, 0:2])
        A1_c = np.copy(min_dRgenFinal[event, 2:4])
        
        # Incorrect permutation indices
        A0_w0 = np.copy(wrongPerm[event, 0, 0:2])
        A1_w0 = np.copy(wrongPerm[event, 0, 2:4])
        
        A0_w1 = np.copy(wrongPerm[event, 1, 0:2])
        A1_w1 = np.copy(wrongPerm[event, 1, 2:4])

        indicesA_c = np.column_stack((A0_c, A1_c))
        indicesA_c = indicesA_c.astype(int)
        
        indicesA_w0 = np.column_stack((A0_w0, A1_w0))
        indicesA_w0 = indicesA_w0.astype(int)
        
        indicesA_w1 = np.column_stack((A0_w1, A1_w1))
        indicesA_w1 = indicesA_w1.astype(int)
        
        
        ## correct dR A0
        diMu_dR[event,0, 0] = math.sqrt(pow(selEtaFinal[event, indicesA_c[0, 0]] - selEtaFinal[event, indicesA_c[1, 0]],2)+ pow(selPhiFinal[event, indicesA_c[0, 0]]-selPhiFinal[event, indicesA_c[1, 0]],2))
        ## correct dR A1
        diMu_dR[event,0 ,1] = math.sqrt(pow(selEtaFinal[event, indicesA_c[0, 1]] - selEtaFinal[event, indicesA_c[1, 1]],2)+ pow(selPhiFinal[event, indicesA_c[0, 1]]-selPhiFinal[event, indicesA_c[1, 1]],2))

        ## wrong0 dR A0
        diMu_dR[event, 1, 0] = math.sqrt(pow(selEtaFinal[event, indicesA_w0[0, 0]] - selEtaFinal[event, indicesA_w0[1, 0]],2)+ pow(selPhiFinal[event, indicesA_w0[0, 0]]-selPhiFinal[event, indicesA_w0[1, 0]],2))
        ## wrong0 dR A1
        diMu_dR[event,1, 1] = math.sqrt(pow(selEtaFinal[event, indicesA_w0[0, 1]] - selEtaFinal[event, indicesA_w0[1, 1]],2)+ pow(selPhiFinal[event, indicesA_w0[0, 1]]-selPhiFinal[event, indicesA_w0[1, 1]],2))

        ## wrong1 dR A0
        diMu_dR[event,2, 0] = math.sqrt(pow(selEtaFinal[event, indicesA_w1[0, 0]] - selEtaFinal[event, indicesA_w1[1, 0]],2)+ pow(selPhiFinal[event, indicesA_w1[0, 0]]-selPhiFinal[event, indicesA_w1[1, 0]],2))
         ## wrong1 dR A1
        diMu_dR[event,2, 1] = math.sqrt(pow(selEtaFinal[event, indicesA_w1[0, 1]] - selEtaFinal[event, indicesA_w1[1, 1]],2)+ pow(selPhiFinal[event, indicesA_w1[0, 1]]-selPhiFinal[event, indicesA_w1[1, 1]],2))

    return diMu_dR

