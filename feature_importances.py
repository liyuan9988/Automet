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

def train_one(config,train_loader, train_handler, rand_seed):
    model_list = []
    np.random.seed(rand_seed)
    for i in range(config["learning_param"]["n_models"]):
        print("train %d"%i)
        X,y = train_handler.build_dataset(config["learning_param"]["neg_train_ratio"])
        mdl = RandomForestClassifier(**config["model_param"])
        model_list.append(mdl.fit(X,y))
    
    return np.array([mdl.feature_importances_ for mdl in model_list]).mean(axis=0)

@click.command()
@click.argument("config_name")
def train(config_name):
    with open(config_name, "r") as f:
        config = json.load(f)    
    train_loader = DataLoader(config, config["learning_param"]["train_csv_root"])
    train_target_table = config["learning_param"].get("train_target_table_file", "seishitu_codeblue_wo_future.csv")
    train_handler = DataHandler(train_loader,config, train_target_table)
    importances = train_one(config,train_loader, train_handler, 42)
    np.save(config_name[:-5]+".feature_importances.npy", importances)     
    
def main():
    train()

if __name__ == "__main__":
    main()
