import numpy as np
np.random.seed(42)
from utils import get_all_CPA, build_dataset, expand_CPA
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

#load data
def train_one(ids):
    n_timepoint = 3
    n_day = 1
    pos_idx = get_all_CPA()
    n_pos = pos_idx.shape[0]
    pos_idx_train, pos_idx_test = train_test_split(pos_idx, random_state=ids)
    pos_idx_train = expand_CPA(pos_idx_train, n_timepoint)
    pos_idx_test = expand_CPA(pos_idx_test, n_timepoint)
    #train
    print("train")
    model_list = []
    
    for i in range(1):
        print("train %d"%i)
        X,y = build_dataset(pos_idx_train ,vital = True, labo = True,    n_timepoint=n_timepoint, nDay=n_day)
        print(X.shape)
        mdl = RandomForestClassifier(n_estimators=100)
        model_list.append(mdl.fit(X,y))
        print(mdl.feature_importances_)
        
    
    #test
    print("test")
    X,y_test = build_dataset(pos_idx_test,vital = True, labo = True , n_negative=10000,  n_timepoint=n_timepoint,nDay=n_day)
    print(np.mean(X[y_test == 0,0]))
    print(np.mean(X[y_test == 1,0]))
    np.save("test_data.npy",np.c_[X,y_test])
    y_scores = np.array([mdl.predict_proba(X) for mdl in model_list])
    y_score = np.mean(y_scores,axis=0,keepdims=False)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score[:,1])
    return auc(fpr["micro"], tpr["micro"])
    
if __name__ == "__main__":
    #with Pool(1) as p:
    #    res = np.array(p.map(train_one, range(100)))
    res = np.array([train_one(i) for i in range(100)])
    np.save("res_all.npy",res)