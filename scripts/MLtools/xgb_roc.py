# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# from matplotlib import plt
from xgboost import XGBClassifier

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def xgb_roc(finalDataShaped, testSize):
	# generate 2 class dataset
	X = finalDataShaped[:, 0:(finalDataShaped.shape[1] - 1)]   
	y = finalDataShaped[:, (finalDataShaped.shape[1] - 1)].astype(int)
	# split into train/test sets
	trainX, testX, trainy, testy = train_test_split(X, y, test_size = testSize, random_state=2)
	# generate a no skill prediction (majority class)
	ns_probs = [0 for _ in range(len(testy))]
	# fit a model
	model = XGBClassifier(n_jobs = -1,use_label_encoder=False,eval_metric='logloss')
	model.fit(trainX, trainy)
	# predict probabilities
	lr_probs = model.predict_proba(testX)
	# keep probabilities for the positive outcome only
	lr_probs = lr_probs[:, 1]
	# calculate scores
	ns_auc = roc_auc_score(testy, ns_probs)
	lr_auc = roc_auc_score(testy, lr_probs)
	# summarize scores
	print('No Skill: ROC AUC=%.3f' % (ns_auc))
	print('Logistic: ROC AUC=%.3f' % (lr_auc))
	# calculate roc curves
	ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
	lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
	# plot the roc curve for the model
	fig, ax = plt.subplots(figsize=(8,8))
	ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
	ax.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
	# axis labels
	plt.xlabel('False Positive Rate', loc='right')
	plt.ylabel('True Positive Rate',loc='top')
	# show the legend
	plt.legend()
	# save the plot
	plt.savefig('Figures/XGB_ROC_dPhiCor.pdf', bbox_inches='tight')
	# show the plot
	plt.show()

	return