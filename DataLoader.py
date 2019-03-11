import pandas as pd
import numpy as np



class DataLoader:

    def __init__(self, params):
        self.csv_root = params["learning_param"]["csv_root"]
        self.file_list = ["vitals/bp.csv",
                          "vitals/hr.csv",
                          "vitals/rr.csv",
                          "vitals/saturation.csv",
                          "vitals/temp.csv",
                          "vitals/urine.csv",
                          "CBC_Coag_Chemo/cbc.csv",
                          "CBC_Coag_Chemo/coag.csv",
                          "CBC_Coag_Chemo/Chemo.csv"
                          ]
        self.table_dict = dict()
        for file in self.file_list:
            self.add_table(file) 

    def add_table(self, file_name):
        print("loading %s ..."%file_name)
        table = pd.read_csv(self.csv_root + file_name)
        table["date_Einstein"] = pd.to_datetime(table["date_Einstein"])
        self.table_dict[file_name] = table

    def __getitem__(self, key):
        return self.table_dict[key]


if __name__ == "__main__":
    import json
    with open("./config.json", "r") as f:
        config = json.load(f)
    loader = DataLoader(config)
    print(loader["vitals/urine.csv"])
        