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

def train_one(config,train_loader, train_handler, rand_seed):
    model_list = []
    np.random.seed(rand_seed)
    for i in range(config["learning_param"]["n_models"]):
        print("train %d"%i)
        X,y = train_handler.build_dataset(config["learning_param"]["neg_train_ratio"])
        mdl = RandomForestClassifier(**config["model_param"])
        model_list.append(mdl.fit(X,y))

    return model_list
    
@click.command()
@click.argument("config_name")
@click.option('--nParallel', '-t', default=1)
def train(config_name, nparallel):
    with open(config_name, "r") as f:
        config = json.load(f)   
    train_csv_root =  config["learning_param"]["train_csv_root"]
    train_target_table = config["learning_param"].get("train_target_table_file", "seishitu_codeblue_wo_future.csv")
    train_loader = DataLoader(config, train_csv_root)
    train_handler = DataHandler(train_loader,config, train_target_table)
    test_loader = DataLoader(config, config["learning_param"]["test_csv_root"])
    test_target_table = config["learning_param"].get("test_target_table_file", "seishitu_codeblue_wo_future.csv")
    test_handler = DataHandler(test_loader,config,test_target_table)
    evaluator = Evaluator(config, test_loader, test_handler)
    print("building_test_data")
    evaluator.build_dataset(100.0, nparallel)
    n_repeat = config["learning_param"]["n_repeat"]
    with Pool(processes=nparallel) as p:
        models = [p.apply_async(train_one, (config, train_loader, train_handler, i)) for i in range(n_repeat)]
        res = np.array([evaluator.predict(model.get()) for model in models])
      
    res_file = config_name[:-5]+".predict.npy"
    np.save(res_file, res)
     
    
def main():
    train()

if __name__ == "__main__":
    main()
