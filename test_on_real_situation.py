import os
import json
import click
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
from Evaluator import Evaluator
from SeisitsuLoader import SeisitsuLoader
from FeatureBuilder import FeatureBuilder
from DataLoader import DataLoader
from DataHandler import DataHandler
import numpy as np
np.random.seed(42)
import joblib
# load data


def train_one(config, train_loader, train_handler, rand_seed):
    model_list = []
    np.random.seed(rand_seed)
    for i in range(config["learning_param"]["n_models"]):
        print("train %d" % i)
        X, y = train_handler.build_dataset(config["learning_param"]["neg_train_ratio"])
        mdl = RandomForestClassifier(**config["model_param"])
        model_list.append(mdl.fit(X, y))

    return model_list


@click.command()
@click.argument("config_name")
@click.option('--nParallel', '-t', default=1)
def train(config_name, nparallel):
    with open(config_name, "r") as f:
        config = json.load(f)
    train_csv_root = config["learning_param"]["train_csv_root"]
    train_target_table = config["learning_param"].get("train_target_table_file", "seishitu_codeblue_wo_future.csv")
    train_loader = DataLoader(config, train_csv_root)
    train_handler = DataHandler(train_loader, config, train_target_table)
    test_loader = DataLoader(config, config["learning_param"]["test_csv_root"])
    test_target_table = config["learning_param"].get("test_target_table_file", "seishitu_codeblue_wo_future.csv")
    test_handler = DataHandler(test_loader, config, test_target_table)
    evaluator = Evaluator(config, test_loader, test_handler)
    print("building_test_data")
    #evaluator.build_dataset(100.0, nparallel)
    mdl_dir_name = config_name[:-5]+"/models/"
    if(os.path.isdir(mdl_dir_name)):
        model_list = []
        n_model = config["learning_param"]["n_models"]
        for i in range(n_model):
            model_list.append(joblib.load(mdl_dir_name+"RF%d.pickle"%i))
        print("model loaded")
    else:
        model_list = train_one(config, train_loader, train_handler, 0)
        os.makedirs(config_name[:-5]+"/models/", exist_ok=True)
        for idx, model in enumerate(model_list):
            joblib.dump(model, config_name[:-5]+"/models/RF%d.pickle"%idx)
    print("model prepared")
    evaluator.simulate_on_all_negative(model_list, config_name[:-5]+"/models/", nparallel)
    

def main():
    train()


if __name__ == "__main__":
    main()
