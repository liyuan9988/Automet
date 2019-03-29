import numpy as np
np.random.seed(42)
from DataHandler import DataHandler
from DataLoader import DataLoader
from FeatureBuilder import FeatureBuilder
from SeisitsuLoader import SeisitsuLoader
from Evaluator import Evaluator
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import click
import json
import os
#load data

def train_one(config,train_loader, train_handler, evaluator, rand_seed):
    model_list = []
    np.random.seed(rand_seed)
    for i in range(config["learning_param"]["n_models"]):
        print("train %d"%i)
        X,y = train_handler.build_dataset(config["learning_param"]["neg_train_ratio"])
        mdl = RandomForestClassifier(**config["model_param"])
        model_list.append(mdl.fit(X,y))

    return evaluator.evaluate(model_list)    
    
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
    evaluator = Evaluator(config, test_loader, test_handler)
    evaluator.build_dataset(100.0)
    n_repeat = config["learning_param"]["n_repeat"]
    with Pool(processes=nparallel) as p:
        multiple_results = [p.apply_async(train_one, (config,train_loader, train_handler, evaluator, i)) for i in range(n_repeat)]
        res = np.array([tmp.get() for tmp in multiple_results])
    
    averages = np.mean(res, axis = 0, keepdims=False)
    variances = np.var(res, axis = 0, keepdims=False)
    config["results"] = dict()
    config["results"]["nData_mean"] = list(averages[:,0])
    config["results"]["nData_var"] = list(variances[:,0])
    config["results"]["nCPA_mean"] = list(averages[:,1])
    config["results"]["nCPA_var"] = list(variances[:,1])
    config["results"]["n_nonCPA_mean"] = list(averages[:,2])
    config["results"]["n_nonCPA_var"] = list(variances[:,2])
    config["results"]["accuracy_mean"] = list(averages[:,3])
    config["results"]["accuracy_var"] = list(variances[:,3])
    config["results"]["PPV_mean"] = list(averages[:,4])
    config["results"]["PPV_var"] = list(variances[:,4])
    config["results"]["NPV_mean"] = list(averages[:,5])
    config["results"]["NPV_var"] = list(variances[:,5])
    config["results"]["sensitivity_mean"] = list(averages[:,6])
    config["results"]["sensitivity_var"] = list(variances[:,6])
    config["results"]["specificity_mean"] = list(averages[:,7])
    config["results"]["specificity_var"] = list(variances[:,7])
    config["results"]["f_val_mean"] = list(averages[:,8])
    config["results"]["f_val_var"] = list(variances[:,8])
    config["results"]["auc_score_mean"] = list(averages[:,9])
    config["results"]["auc_score_var"] = list(variances[:,9])
    
    
    
    with open(config_name[:-5]+".res.json", "w") as f:
        json.dump(config, f)
     
    
def main():
    train()

if __name__ == "__main__":
    main()
