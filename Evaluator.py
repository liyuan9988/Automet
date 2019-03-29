from DataHandler import DataHandler
import numpy as np
from sklearn.metrics import roc_curve, auc


class Evaluator:

    def __init__(self, config, test_loader, test_handler):
        self.config = config
        self.test_loader = test_loader
        self.test_handler = test_handler

    def build_dataset(self, negative_ratio):
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

