import numpy as np
np.random.seed(42)
from utils import get_all_CPA, build_dataset, expand_CPA, bulid_target_table
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

#load data
def train_one(ids):
    n_timepoint = 3
    n_day = 2
    n_model = 10

    #train_path = "/Users/liyuanxu/Dropbox/automet/data/train/"
    #test_path = "/Users/liyuanxu/Dropbox/automet/data/test/"
    train_path = "./data/train_190221/"
    test_path = "./data/test_190221/"
    train_target_table = bulid_target_table(train_path)
    test_target_table = bulid_target_table(test_path)

    pos_idx_train = get_all_CPA(train_target_table)
    pos_idx_train = expand_CPA(pos_idx_train, n_timepoint)
    #train
    print("train")
    model_list = []
    
    for i in range(n_model):
        print("train %d"%i)
        X,y = build_dataset(train_path, train_target_table, pos_idx_train ,vital = True, labo = False,    n_timepoint=n_timepoint, nDay=n_day)
        print(X.shape)
        mdl = RandomForestClassifier(n_estimators=100)
        model_list.append(mdl.fit(X,y))
        print(mdl.feature_importances_)
        
    
    #test
    print("test")
    pos_idx_test = get_all_CPA(test_target_table)
    pos_idx_test = expand_CPA(pos_idx_test, n_timepoint)
    X,y_test = build_dataset(test_path, test_target_table,pos_idx_test, vital = True, labo = False , n_negative=10000,  n_timepoint=n_timepoint,nDay=n_day)
    print(np.mean(X[y_test == 0,0]))
    print(np.mean(X[y_test == 1,0]))
    y_scores = np.array([mdl.predict_proba(X) for mdl in model_list])
    y_score = np.mean(y_scores,axis=0,keepdims=False)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score[:,1])
    return auc(fpr["micro"], tpr["micro"])
    
if __name__ == "__main__":
    with Pool(10) as p:
        res = np.array(p.map(train_one, range(10)))
    #res = np.array([train_one(i) for i in range(100)])
    np.save("res_sub.npy",res)
