import numpy as np
np.random.seed(42)
from DataHandler import DataHandler
from DataLoader import DataLoader
from FeatureBuilder import FeatureBuilder
from SeisitsuLoader import SeisitsuLoader
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import click
import json
import os
#load data

def train_one(config,train_loader, train_handler, X_test, y_test, rand_seed):
    model_list = []
    np.random.seed(rand_seed)
    for i in range(config["learning_param"]["n_models"]):
        print("train %d"%i)
        X,y = train_handler.build_dataset(config["learning_param"]["neg_train_ratio"])
        mdl = RandomForestClassifier(**config["model_param"])
        model_list.append(mdl.fit(X,y))
        
    y_scores = np.array([mdl.predict_proba(X_test) for mdl in model_list])
    y_score = np.mean(y_scores,axis=0,keepdims=False)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score[:,1])
    return auc(fpr["micro"], tpr["micro"])

@click.command()
@click.argument("config_name")
@click.option('--nParallel', '-t', default=1)
def train(config_name, nparallel):
    with open(config_name, "r") as f:
        config = json.load(f)    
    train_loader = DataLoader(config, config["learning_param"]["train_csv_root"])
    train_handler = DataHandler(train_loader,config)
    test_loader = DataLoader(config, config["learning_param"]["test_csv_root"])
    test_handler = DataHandler(test_loader,config)
    X_test,y_test = test_handler.build_dataset(100.0)
    n_repeat = config["learning_param"]["n_repeat"]
    with Pool(processes=nparallel) as p:
        multiple_results = [p.apply_async(train_one, (config,train_loader, train_handler, X_test, y_test, i)) for i in range(n_repeat)]
        res = np.array([tmp.get() for tmp in multiple_results])
    config["mean_auc"] = np.mean(res)
    config["var_auc"] = np.var(res)
    
    with open(config_name[:-5]+".res.json", "w") as f:
        json.dump(config, f)
     
    
def main():
    train()

if __name__ == "__main__":
    main()
