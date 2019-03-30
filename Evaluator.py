from DataHandler import DataHandler
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


class Evaluator:

    def __init__(self, config, test_loader, test_handler):
        self.config = config
        self.test_loader = test_loader
        self.test_handler = test_handler

    def build_dataset(self,negative_ratio):
        key = self.config["learning_param"].get("eval_key", "all")
        if(key == "MedSurg"):
            self.build_dataset_for_MedSurg(negative_ratio)
        else:
            self. build_dataset_for_all(negative_ratio)

    def build_dataset_for_MedSurg(self, negative_ratio):
        n_positive = self.test_handler.CPA_table.shape[0]
        n_negative = int(n_positive * negative_ratio)
        NonCPA_target, row_id = self.test_handler.sample_negative_CPA(n_negative,True)
        neg_MedSurg = self.test_handler.target_table["MedSurg"].iloc[row_id]
        neg_MedSurg.reset_index(drop=True, inplace = True)
        pos_MedSurg = self.test_handler.target_table["MedSurg"].loc[self.test_handler.target_table["CPA"] == True]
        pos_MedSurg.reset_index(drop=True, inplace = True)
        
        #compile_Meg
        MedCPA = self.test_handler.CPA_table.loc[pos_MedSurg == "medicine", :]
        MedNonCPA = NonCPA_target.loc[neg_MedSurg == "medicine", :]
        med_n_pos = MedCPA.shape[0]
        med_n_neg = MedNonCPA.shape[0]
        med_y = np.concatenate([np.ones(med_n_pos),np.zeros(med_n_neg)],axis=0)
        med_X = self.test_handler.load_feature(pd.concat([MedCPA, MedNonCPA],axis=0))

        #compile_Surg
        SurgCPA = self.test_handler.CPA_table.loc[pos_MedSurg == "surgery", :]
        SurgNonCPA = NonCPA_target.loc[neg_MedSurg == "surgery", :]
        surg_n_pos = SurgCPA.shape[0]
        surg_n_neg = SurgNonCPA.shape[0]
        surg_y = np.concatenate([np.ones(surg_n_pos),np.zeros(surg_n_neg)],axis=0)
        surg_X = self.test_handler.load_feature(pd.concat([SurgCPA, SurgNonCPA],axis=0))

        self.data_list = [(med_X, med_y), (surg_X, surg_y)]


    def build_dataset_for_all(self, negative_ratio):
        X_test,y_test = self.test_handler.build_dataset(negative_ratio)
        self.data_list = [(X_test, y_test)]
        print(y_test)
        
    def evaluate(self, models):
        res = []
        for X,y_test in self.data_list:
            y_scores = np.array([mdl.predict_proba(X) for mdl in models])
            y_score = np.mean(y_scores,axis=0,keepdims=False)[:,1]
            res.append(self.cal_metrics(y_score, y_test))
        return np.array(res)

    def cal_metrics(self, y_score, y_test):
        nData = y_test.shape[0]
        nCPA = np.sum(y_test)
        n_nonCPA = np.sum(1-y_test)
        fpr, tpr, therehsolds = roc_curve(y_test, y_score)
        auc_score = auc(fpr, tpr)
        idx = np.argmax(fpr+tpr)
        youden_index = therehsolds[idx]
        tp = np.sum(y_test[y_score > youden_index])
        fp = np.sum(1.0 - y_test[y_score > youden_index])
        fn = np.sum(y_test[y_score <= youden_index])
        tn = np.sum(1.0 - y_test[y_score <= youden_index])
        accuracy = (tp+tn) / (tp+tn+fp+fn)
        PPV = tp / (tp + fp)
        NPV = tn / (tn + fn)
        sensitivity = tp /  (tp + fn)
        specificity =   tn / (tn + fp)
        f_val = (2.0*tp) / (2.0*tp + fp + fn)
        res = np.array([nData, nCPA, n_nonCPA, accuracy, PPV, NPV, sensitivity, specificity, f_val, auc_score])
        return res 

