import pandas as pd
import numpy as np
from joblib import Parallel, delayed

class SeisitsuLoader:

    def __init__(self, csv_path):
        self.table = pd.read_csv(csv_path)
        self.table["enter_Einstein"] = pd.to_datetime(self.table["enter_Einstein"])
        self.table["discharge_Einstein"] = pd.to_datetime(self.table["discharge_Einstein"])
        self.table["date_codeB"] = pd.to_datetime(self.table["date_codeB"])
        self.table["emergency"] = pd.to_numeric(self.table["emergency"])
        #self.Med = pd.get_dummies(self.table["MedSurg"]).values[:,0]
        #self.Med[np.isnan(self.Med)] = 0 #NAはMedとする
        self.Sex = pd.get_dummies(self.table["sex"]).values[:,0]
        self.Bmi = pd.to_numeric(self.table["BMI"])
        self.Bmi[np.isnan(self.Bmi)] = np.nanmean(self.Bmi) #NAは平均でimputation
    
    def query_core(self, id_hashed, target_date):
        flgs = self.table["id_hashed"]==int(id_hashed)
        date = pd.to_datetime(target_date)
        flgs = np.logical_and(flgs, date >= self.table["enter_Einstein"])
        flgs = np.logical_and(flgs, date <= self.table["discharge_Einstein"])
        if(np.sum(flgs)==0):
            print(date)
            print(id_hashed)
        idx_list = np.where(flgs)[0]
        prev_CPA = self.check_prev_CPA(idx_list, date)
        idx = idx_list[0]
        date_diff = (date - self.table["enter_Einstein"].iloc[idx]).days
        emergency = self.table["emergency"].iloc[idx]
        age = self.table["age"].iloc[idx]
        sex = self.Sex[idx]
        bmi = self.Bmi[idx]
        #med_sug = self.Med[idx]
        res = np.array([date_diff,emergency,age,prev_CPA,sex,bmi])
        return res
    
    def query_all(self, id_hashed_list, target_date_list):
        res = [self.query_core(id_hashed, target_date) for id_hashed, target_date in zip(id_hashed_list, target_date_list)]
        return np.array(res)

    def check_prev_CPA(self, idx_list, target_date):
        for idx in idx_list:
            if(self.table["CPA"].iloc[idx]):
                code_B = self.table["date_codeB"].iloc[idx]
                if(code_B < target_date):
                    return 1
        return 0




if __name__ == "__main__":
    #dl = DataLoader("~/Dropbox/automet/data/cleaned_data_csv/vitals/bp.csv", "sBP",timepoints=3)
    dl = SeisitsuLoader("~/Dropbox/automet/data/train/seishitu_codeblue_wo_future.csv")
    print(dl.table.shape)
    print(dl.query_core("151793","1927-02-25"))
    print(dl.query_all(["151793","151793"],["1927-02-24","1927-02-25"]))
    print(dl.query_all(["5718331","5718331","5718331"],["1960-07-18","1960-07-19","1960-07-20"]))
