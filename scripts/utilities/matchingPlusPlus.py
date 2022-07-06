#!/bin/env python
import numpy as np
import pandas as pd
from math import sqrt
from random import randint
from tqdm import tqdm # Progress bar for loops

'''

Collection of utilities for matching gen level to sel level muons,
generating the possible permutations, invariant mass calculation,
and preparing the data for ML algorithms

S.D. Butalla & M. Rahmani
2021/06/06

'''
class match:
    def __init__(self, genAMu_eta, selMu_eta, genAMu_phi, selMu_phi, genAMu_charge, selMu_charge):
        self.genAMu_eta    = genAMu_eta
        self.selMu_eta     = selMu_eta
        self.genAMu_phi    = genAMu_phi
        self.selMu_phi     = selMu_phi
        self.genAMu_charge = genAMu_charge
        self.selMu_charge  = selMu_charge

    def dRgenCalc(self):
        '''
        Calculates the dR value between the generator level and reco level muons.
        To be used to determine if the muons are reconstructed properly.
        genAMu_eta: np.ndarray; reco level muon eta
        selMu_eta:  np.ndarray; generator level muon eta
        genAMu_phi: np.ndarray; reco level muon phi
        selMu_phi:  np.ndarray; generator level muon phi
        '''
        self.dRgen = np.ndarray((self.genAMu_eta.shape[0], 4, 4))

        for ii in tqdm(range(self.genAMu_eta.shape[0])): # for each event
            for jj in range(4): # dR for each gen muon
                for ll in range(4): #for each sel muon
                    self.dRgen[ii, jj, ll] = sqrt(pow(self.genAMu_eta[ii, jj] - self.selMu_eta[ii, ll], 2)+ pow(self.genAMu_phi[ii, jj] - self.selMu_phi[ii, ll], 2))
        
        return self.dRgen

    def SSM(self, verbose = False, extraInfo = False):

        self.min_dRgen    = np.ndarray((self.genAMu_charge.shape[0], 4))
        self.dRgenMatched = np.ndarray((self.genAMu_charge.shape[0], 4))

        for event in tqdm(range(self.selMu_charge.shape[0])):
            if verbose:
                print ("Event: ", event)
                
            tempGenCharge = self.genAMu_charge[event, :] # 1 x 4
            tempSelCharge = self.selMu_charge[event, :] # 1 x 4

            # Match first randomly chosen muon 
            index      = randint(0, 3) # Generate random int in [0,3]
            chargeTemp = tempGenCharge[index] # Get charge corresponding to AXMuY
            
            genCharge  = np.array(np.where(self.genAMu_charge[event,:] == chargeTemp)).reshape((2,)) # Select gen muons where charge (array of indices)
            selCharge  = np.array(np.where(self.selMu_charge[event,:] == chargeTemp)).reshape((2,))# Select sel muons where charge is == AXMuY (array of indices)

            genChargeopo = np.array(np.where(self.genAMu_charge[event,:] != chargeTemp)).reshape((2,)) # Select gen muons where charge (array of indices)
            selChargeopo = np.array(np.where(self.selMu_charge[event,:] != chargeTemp)).reshape((2,))

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
                    dEtaMatched[event, ii] = self.genAMu_eta[event, genInd[ii]] - self.selMu_eta[event, selInd[ii]]
                    dPhiMatched[event, ii] = self.genAMu_phi[event, genInd[ii]] - self.selMu_phi[event, selInd[ii]]
                    #dPtMatched[event, ii] = genpTFinal[event, genInd[ii]] - selpTFinal[event, selInd[ii]]

        
        if verbose == False and extraInfo == False:
            return self.min_dRgen, self.dRgenMatched 

        if verbose == True and extraInfo == True:
            return self.min_dRgen, self.dRgenMatched, dEtaMatched, dPhiMatched #, dPtMatched

        if verbose == False and extraInfo == True:
            return dEtaMatched, dPhiMatched #, dPtMatched

    def dRcut(self, cut):
            return np.unique(np.where(self.dRgenMatched > cut)[0])

    def Wpermutation(min_dRgenFinal, selChargeFinal, genChargeFinal):

        wrongPerm = np.ndarray((min_dRgenFinal.shape[0], 2, min_dRgenFinal.shape[1]))

                
        for event in tqdm(range(min_dRgenFinal.shape[0])):

        #     print('Event ', event)
            wrongPerm[event, 0]   =  min_dRgenFinal[event, :]
            wrongPerm[event, 1]   =  min_dRgenFinal[event, :]
            
            correctPerm = np.array(min_dRgenFinal[event, :], dtype=int) # Array of indices of matched data
        #     print("correctPerm", correctPerm)
            
            correctSelChargeOrder = selChargeFinal[event, correctPerm]  # Correct order of charges (matched)
        #     print("correctSelChargeOrder", correctSelChargeOrder)
            
            pos = np.array(np.where(correctSelChargeOrder == 1)).reshape((2,))  # indices of positive sel muons
        #     print("pos", pos)
            
            

            wrongPerm[event, 0, pos[0]] = min_dRgenFinal[event, pos[1]]
            wrongPerm[event, 0, pos[1]] = min_dRgenFinal[event, pos[0]]
            

            neg = np.array(np.where(correctSelChargeOrder == -1)).reshape((2,))
        #     print("neg", neg)
            
        #          print("neg", neg)
            # Swap sel muons 
            wrongPerm[event, 1, neg[0]] = min_dRgenFinal[event, neg[1]]
            wrongPerm[event, 1, neg[1]] = min_dRgenFinal[event, neg[0]]
            
        #     print("wrongPerm", wrongPerm)

        return wrongPerm






    def allPerms(min_dRgen, wrongPerm):
        allPerm = np.ndarray((min_dRgen.shape[0], wrongPerm.shape[1] + 1, 4))
        for event in tqdm(range(min_dRgen.shape[0])):
            allPerm[event, 0, :] = min_dRgen[event, :] # correct permutation
            allPerm[event, 1, :] = wrongPerm[event, 0, :] # wrong permutation 0
            allPerm[event, 2, :] = wrongPerm[event, 1, :] # wrong permutation 1
        
        allPerm = allPerm.astype(int) 
       
        return allPerm

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

    def dR_diMu(min_dRgenFinal, wrongPerm, selEtaFinal, selPhiFinal):

        diMu_dR = np.ndarray((min_dRgenFinal.shape[0], 3, 2))
        dPhi = np.ndarray((min_dRgenFinal.shape[0], 3, 2))
        for event in tqdm(range(min_dRgenFinal.shape[0])):
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

            dPhi[event, 0, 0] = selPhiFinal[event, indicesA_c[0, 0]]-selPhiFinal[event, indicesA_c[1, 0]]
            dPhi[event, 0, 1] = selPhiFinal[event, indicesA_c[0, 1]]-selPhiFinal[event, indicesA_c[1, 1]]
            dPhi[event, 1, 0] = selPhiFinal[event, indicesA_w0[0, 0]]-selPhiFinal[event, indicesA_w0[1, 0]]
            dPhi[event, 1, 1] = selPhiFinal[event, indicesA_w0[0, 1]]-selPhiFinal[event, indicesA_w0[1, 1]]
            dPhi[event, 2, 0] = selPhiFinal[event, indicesA_w1[0, 0]]-selPhiFinal[event, indicesA_w1[1, 0]]
            dPhi[event, 2, 1] = selPhiFinal[event, indicesA_w1[0, 1]]-selPhiFinal[event, indicesA_w1[1, 1]]


                    ## correct dR A0
            diMu_dR[event,0, 0] = sqrt(pow(selEtaFinal[event, indicesA_c[0, 0]] - selEtaFinal[event, indicesA_c[1, 0]],2)+ pow(dPhi[event,0,0], 2))
            ## correct dR A1
            diMu_dR[event,0 ,1] = sqrt(pow(selEtaFinal[event, indicesA_c[0, 1]] - selEtaFinal[event, indicesA_c[1, 1]],2)+ pow(dPhi[event,0,1], 2))

            ## wrong0 dR A0
            diMu_dR[event, 1, 0] = sqrt(pow(selEtaFinal[event, indicesA_w0[0, 0]] - selEtaFinal[event, indicesA_w0[1, 0]],2)+ pow(dPhi[event,1,0], 2))
            ## wrong0 dR A1
            diMu_dR[event,1, 1] = sqrt(pow(selEtaFinal[event, indicesA_w0[0, 1]] - selEtaFinal[event, indicesA_w0[1, 1]],2)+ pow(dPhi[event,1,1], 2))

            ## wrong1 dR A0
            diMu_dR[event,2, 0] = sqrt(pow(selEtaFinal[event, indicesA_w1[0, 0]] - selEtaFinal[event, indicesA_w1[1, 0]],2)+ pow(dPhi[event,2,0], 2))
             ## wrong1 dR A1
            diMu_dR[event,2, 1] = sqrt(pow(selEtaFinal[event, indicesA_w1[0, 1]] - selEtaFinal[event, indicesA_w1[1, 1]],2)+ pow(dPhi[event,2,1], 2))
            
        return diMu_dR, dPhi


    def fillFinalArray(selpTFinal, selEtaFinal, selPhiFinal, selChargeFinal, allPerm, invariantMass, dPhi, diMu_dR, perm = False, diMu_dRBool = False, pandas = False):
        if diMu_dRBool == True:
            dataframe = np.ndarray((selpTFinal.shape[0], invariantMass.shape[1], 23))
        else:
            dataframe = np.ndarray((selpTFinal.shape[0], invariantMass.shape[1], 19))
        print(dataframe.shape)
        
        if perm == False and diMu_dRBool == True:
            for event in tqdm(range(selpTFinal.shape[0])):
                for permutation in range(invariantMass.shape[1]):
                    # Remove row of data for each sel muon
                    selpT_temp     = np.copy(selpTFinal[event, :]) # 1 x 4
                    selEta_temp    = np.copy(selEtaFinal[event, :]) # 1 x 4
                    selPhi_temp    = np.copy(selPhiFinal[event, :]) # 1 x 4
                    selCharge_temp = np.copy(selChargeFinal[event, :]) # 1 x 4
                    invMass_temp   = np.copy(invariantMass[event, permutation, :]) # 1 x 3 
                    diMu_dR_temp = np.copy(diMu_dR[event, permutation, :]) # 1 x 2
                
                    dataframe[event, permutation, :] = np.concatenate((selpT_temp, selEta_temp, selPhi_temp, selCharge_temp, diMu_dR_temp, invMass_temp))

        if perm == True and diMu_dRBool == False:
            for event in tqdm(range(selpTFinal.shape[0])):
                for permutation in range(invariantMass.shape[1]):
                    selpT_temp     = np.copy(selpTFinal[event, allPerm[event, permutation]]) # 1 x 4
                    selEta_temp    = np.copy(selEtaFinal[event, allPerm[event, permutation]]) # 1 x 4
                    selPhi_temp    = np.copy(selPhiFinal[event, allPerm[event, permutation]]) # 1 x 4
                    selCharge_temp = np.copy(selChargeFinal[event, allPerm[event, permutation]]) # 1 x 4
                    invMass_temp   = np.copy(invariantMass[event, permutation, :]) # 1 x 3 

                    dataframe[event, permutation, :] = np.concatenate((selpT_temp, selEta_temp, selPhi_temp, selCharge_temp, invMass_temp))
        
        if perm == True and diMu_dRBool == True:
            n = 0
            for event in tqdm(range(selpTFinal.shape[0])):
                n += 1
                for permutation in range(invariantMass.shape[1]):
                               # Remove row of data for each sel muon
                    selpT_temp     = np.copy(selpTFinal[event, allPerm[event, permutation]]) # 1 x 4
                    selEta_temp    = np.copy(selEtaFinal[event, allPerm[event, permutation]]) # 1 x 4
                    selPhi_temp    = np.copy(selPhiFinal[event, allPerm[event, permutation]]) # 1 x 4
                    selCharge_temp = np.copy(selChargeFinal[event, allPerm[event, permutation]]) # 1 x 4
                    invMass_temp   = np.copy(invariantMass[event, permutation, :]) # 1 x 3
                    dPhi_temp      = np.copy(dPhi[event, permutation, :])
                    diMu_dR_temp   = np.copy(diMu_dR[event, permutation, :]) # 1 x 2

                    dataframe[event, permutation, :] = np.concatenate((selpT_temp, selEta_temp, selPhi_temp, selCharge_temp, dPhi_temp, diMu_dR_temp, invMass_temp))
            print("number of events",n)        

        if perm == False and diMu_dRBool == False:
            for event in tqdm(range(selpTFinal.shape[0])):
                for permutation in range(invariantMass.shape[1]):
                    # Remove row of data for each sel muon
                    selpT_temp     = np.copy(selpTFinal[event, :]) # 1 x 4
                    selEta_temp    = np.copy(selEtaFinal[event, :]) # 1 x 4
                    selPhi_temp    = np.copy(selPhiFinal[event, :]) # 1 x 4
                    selCharge_temp = np.copy(selChargeFinal[event, :]) # 1 x 4
                    invMass_temp   = np.copy(invariantMass[event, permutation, :]) # 1 x 3 


                    
                    dataframe[event, permutation, :] = np.concatenate((selpT_temp, selEta_temp, selPhi_temp, selCharge_temp, invMass_temp)) # 1 x 19
        
        if diMu_dRBool == True:
            dataframe_shaped = np.reshape(dataframe, (selpTFinal.shape[0] * invariantMass.shape[1], 23))
        else:
            dataframe_shaped = np.reshape(dataframe, (selpTFinal.shape[0] * invariantMass.shape[1], 19))
        print('Shape of output data: {}'.format(dataframe_shaped.shape))
        
        if pandas == True:
            columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
              "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dRA0", "dRA1", "invMassA0",
              "invMassA1","label"]
            df = pd.DataFrame(data=dataframe_shaped, columns=columns)
            
            return dataframe_shaped, df
        else:
            return dataframe_shaped