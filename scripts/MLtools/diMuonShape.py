import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tqdm import tqdm # Progress bar for loops
import pandas as pd

def sort(dRgenFinal, allPerm, selMu_ptCut, selMu_etaCut, selMu_phiCut, selMu_chargeCut, invariantMass, dPhi, diMu_dR):

    EventNum    = np.ndarray((dRgenFinal.shape[0], 3)) #3199 * 3 * 22
    counter = 0

    for event in range(dRgenFinal.shape[0]):

        EventNum[event, 0] = counter # 
        EventNum[event, 1] = counter # 
        EventNum[event, 2] = counter # 
        counter = counter+1

    EventNum = EventNum.astype(int)
    
    
    ## adding event number to the data frame
    dataframe    = np.ndarray((dRgenFinal.shape[0], 3,24)) #3199 * 3 * 22

    for event in tqdm(range(selMu_ptCut.shape[0])):

        for permutation in range(invariantMass.shape[1]):

            EventNum_temp   = np.copy(EventNum[event, permutation]).reshape((1,))
            selpT_temp     = np.copy(selMu_ptCut[event, allPerm[event, permutation]]) # 1 x 4
            selEta_temp    = np.copy(selMu_etaCut[event, allPerm[event, permutation]]) # 1 x 4
            selPhi_temp    = np.copy(selMu_phiCut[event, allPerm[event, permutation]]) # 1 x 4
            selCharge_temp = np.copy(selMu_chargeCut[event, allPerm[event, permutation]]) # 1 x 4
            invMass_temp   = np.copy(invariantMass[event, permutation, :]) # 1 x 3  
            dPhi_temp      = np.copy(dPhi[event, permutation, :])
            diMu_dR_temp   = np.copy(diMu_dR[event, permutation, :]) # 1 x 2
            dataframe[event, permutation, :]= np.concatenate((selpT_temp, selEta_temp, selPhi_temp, selCharge_temp, dPhi_temp, diMu_dR_temp,EventNum_temp,invMass_temp)) # 1 x 19


    ## flattning the df
    dataframe_shaped = np.reshape(dataframe, (selMu_ptCut.shape[0] * invariantMass.shape[1], 24))
    total_df = pd.DataFrame(dataframe_shaped, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
          "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "event", "invMassA0",
          "invMassA1","Label"])
    pd.set_option("max_rows", 7)
    
    return dataframe_shaped



	# 	## puting x,y test and train in pandas 
	# X = dataframe_shaped[:, 0:21]   
	# Y = dataframe_shaped[:, 18:22]


def shape(X_train, X_test, y_train, y_test):
        
    Xtrain = pd.DataFrame(X_train, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
              "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "event", "invMassA0",
              "invMassA1"])

    Ytrain = pd.DataFrame(y_train, columns=['event','invmA0','invmA1','pair'])


    Ytrain = Ytrain.drop(['invmA0', 'invmA1'], axis = 1)



    Xtest =  pd.DataFrame(X_test, columns = ["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3",
              "selPhi0", "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0", "dPhi1","dRA0", "dRA1", "event", "invMassA0",
              "invMassA1"])

    Ytest = pd.DataFrame(y_test, columns=['event','invmA0','invmA1','pair'])
    Ytest = Ytest.drop(['invmA0', 'invmA1'], axis = 1)



        ## removing event coloum from df pd 
    Ytrain = Ytrain.drop(['event'], axis = 1)
    Ytest = Ytest.drop(['event'], axis = 1)

    ## xgb and fitting 
    model = XGBClassifier(n_jobs = -1,use_label_encoder=False,eval_metric='logloss')
    model.fit(Xtrain, Ytrain)

    ## merging Xtrain and Xtest
    totalDF_X1 = pd.concat([Xtrain, Xtest], axis=0)

    ## sorting the X based on event number 
    sorted_X = totalDF_X1.sort_values('event')

    ## predicting for ALL X
    total_predY = model.predict(sorted_X)


    # reshaping the 
    total = total_predY.reshape(-1, 3)

        # # sorted_X
    arr = total_predY
    sorted_X['Perdict'] = arr.tolist()
    print("sorted data frame: ")
    sorted_X.head(n=10)


        # selecting rows based on condition
    #here we only have correct match for each event 
    correct_pair= sorted_X[sorted_X['Perdict']==1] 
    print("sorted data frame for correct pairs: ")
    correct_pair.head(n=10)
    # print('\nResult dataframe :\n', rslt_df)

        ## selecting the wrong matches for each event 
    wrong_pair= sorted_X[sorted_X['Perdict']==0] 
    print("sorted data frame for wrong pairs: ")
    wrong_pair.head(n=10)

    
    
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

        ## correct formation of di-muons: ivariant mass 2D
    ax = correct_pair.plot(x='invMassA0', y='invMassA1', kind='scatter', color='darkred', ax=axes[0])
    ax.set_ylim(0, 80)
    ax.set_xlim(0, 80)
    ax.set_xlabel(r'$m_{\mu\mu_{1}}$[GeV]', loc='right')
    ax.set_ylabel(r'$m_{\mu\mu_{2}}$[GeV]', loc='top')
    ax.set_title('Correct Pair')




        ## wrong formation of di-muons: ivariant mass 2D
    ax = wrong_pair.plot(x='invMassA0', y='invMassA1', kind='scatter', color='darkred', ax=axes[1])
    # result.plot(x='invMassA0', y='invMassA1', figsize=(10,10), kind='scatter',ax=ax, color='red')
    ax.set_ylim(0, 250)
    ax.set_xlim(0, 250)
    ax.set_xlabel(r'$m_{\mu\mu_{1}}$[GeV]', loc='right')
    ax.set_ylabel(r'$m_{\mu\mu_{2}}$[GeV]', loc='top')
    ax.set_title('Wrong Pair')
    
    fig = ax.get_figure()
    fig.savefig("Figures/2DInvMass_dPhiCor.pdf", bbox_inches='tight')


    
    
    
    fig, axess = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    
    ax = correct_pair['invMassA0'].plot.hist(bins=100, alpha=0.8, range=(0, 30), color='darkred' , ax=axess[0,0])
    ax.set_xlabel(r'$m_{\mu\mu_{1}}$[GeV]', loc='right')
    ax.set_ylabel('Numbrt of Events', loc='top')
    # ax.set_ylim(0, 200)
    # ax.set_xlim(0, 30)
    ax.set_title('Correct Pair')



    ax =  correct_pair['invMassA1'].plot.hist(bins=100, alpha=0.8,  range=(0, 30), color='darkred' , ax=axess[0,1])
    ax.set_xlabel(r'$m_{\mu\mu_{2}}$[GeV]', loc='right')
    ax.set_ylabel('Numbrt of Events', loc='top')
    # ax.set_xlim(0, 30)
    ax.set_title('Correct Pair')



    ax =  wrong_pair['invMassA0'].plot.hist(bins=100, alpha=0.8, range=(0, 180),color='darkred', ax=axess[1,0])
    ax.set_xlabel(r'$m_{\mu\mu_{1}}$[GeV]', loc='right')
    ax.set_ylabel('Numbrt of Events', loc='top')
    # ax.set_xlim(0, 180)
    ax.set_title('Wrong Pair')



    ax =  wrong_pair['invMassA1'].plot.hist(bins=100, alpha=0.8, range=(0, 180), color='darkred',  ax=axess[1,1])
    ax.set_xlabel(r'$m_{\mu\mu_{2}}$[GeV]', loc='right')
    # ax.set_xlim(0, 180)
    ax.set_ylabel('Numbrt of Events', loc='top')
    ax.set_title('Wrong Pair') 

    fig = ax.get_figure()
    fig.savefig("Figures/1DInvMass_dPhiCor.pdf", bbox_inches='tight')

    
    return sorted_X, correct_pair, wrong_pair


