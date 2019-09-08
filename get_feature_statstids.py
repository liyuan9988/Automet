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
import pandas as pd
#load data
    
def get_col_names(configs):
    res = ["date_diff","emergency","age","prev_CPA","sex","bmi"]
    features = configs["feature"]
    for ind_param in features:
        name = ind_param["name"]
        feature_type = ind_param["feature_type"]
        if(feature_type == "vital"):
            default = configs["Vital_setting"]
        elif(feature_type == "labo"):
            default = configs["Labo_setting"]
        n_timepoints = ind_param.get("n_timepoints", default["n_timepoints"])
        for i in range(n_timepoints):
            res.append("{}_{}".format(name,i+1))
    return res


@click.command()
@click.argument("config_name")
@click.option('--train', 'dataset', flag_value='train',
              default=True)
@click.option('--test', 'dataset', flag_value='test')
@click.option('--nParallel', '-t', default=1)
def train(config_name, dataset, nparallel):
    with open(config_name, "r") as f:
        config = json.load(f)  
    if dataset == "train": 
        csv_root =  config["learning_param"]["train_csv_root"]
        target_table = config["learning_param"].get("train_target_table_file", "seishitu_codeblue_wo_future.csv")
    elif dataset == "test":
        csv_root =  config["learning_param"]["test_csv_root"]
        target_table = config["learning_param"].get("test_target_table_file", "seishitu_codeblue_wo_future.csv")
    loader = DataLoader(config, csv_root)
    handler = DataHandler(loader, config, target_table)
    X,y = handler.build_dataset(100.0,nparallel)
    names = get_col_names(config)
    CPA_mean = np.mean(X[y==1], axis=0)
    CPA_var = np.var(X[y==1], axis=0)
    nCPA_mean = np.mean(X[y==0], axis=0)
    nCPA_var = np.var(X[y==0], axis=0)
    table = pd.DataFrame(dict(names=names,
                              CPA_mean = CPA_mean,
                              CPA_var = CPA_var,
                              nCPA_mean = nCPA_mean,
                              nCPA_var= nCPA_var))
    table.to_csv(dataset+"_"+config_name[:-4]+"csv",index = False)
    
def main():
    train()

if __name__ == "__main__":
    main()
