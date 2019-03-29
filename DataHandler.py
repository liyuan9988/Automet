import numpy as np
import pandas as pd
from FeatureBuilder import FeatureBuilder
from SeisitsuLoader import SeisitsuLoader

class DataHandler:

    def __init__(self, dataloader, config):
        self.config = config
        self.dataloader = dataloader  
        self.build_feature_builders()
        self.bulid_target_table() 

    def build_feature_builders(self):
        self.feature_list = []
        for feature_opt in self.config["feature"]:
            builder = FeatureBuilder(feature_opt, self.config, self.dataloader)
            self.feature_list.append(builder)

    def bulid_target_table(self):
        path = self.dataloader.csv_root
        file_name = self.config["learning_param"].get("target_table_file", "seishitu_codeblue_wo_future_small.csv")
        self.seisitsu = SeisitsuLoader(path+file_name)  
        self.target_table = pd.read_csv(path+file_name)
        self.target_table["enter_Einstein"] = pd.to_datetime(self.target_table["enter_Einstein"])
        self.target_table["discharge_Einstein"] = pd.to_datetime(self.target_table["discharge_Einstein"])
        self.target_table["date_codeB"] = pd.to_datetime(self.target_table["date_codeB"])
        self.CPA_table = self.target_table.loc[self.target_table["CPA"] == True, :]
        self.CPA_table = self.CPA_table.loc[:,["id_hashed","date_codeB","CPAtp"]]
        self.CPA_table.columns = ["id_hashed","target_date", "timepoint"]
        self.CPA_table= self.CPA_table.reset_index(drop=True)
        #expand_CPA when predicting ahead outcomes
        self.expand_CPA(self.config["learning_param"]["max_target_timepoint_gap"])
    
    def cal_date_and_timepoint(self, date, timepoint, diff_timepoint):
        n_timepoints_per_day = self.config["learning_param"]["n_target_timepoints_per_day"]
        final_time = timepoint + diff_timepoint
        diff_date = (final_time - 1) // n_timepoints_per_day
        final_time = (final_time - 1) % n_timepoints_per_day
        finalDate = date + pd.to_timedelta("%sday"%diff_date)
        return finalDate, final_time + 1   

    def expand_CPA(self, n_timepoint): 
        res = [self.CPA_table]
        for i in range(1,n_timepoint):
            table = self.CPA_table.copy()
            for j in range(table.shape[0]):
                date = table["target_date"].iloc[j]
                timepoint = table["timepoint"].iloc[j]
                date, timepoint = self.cal_date_and_timepoint(date, timepoint, -i)
                table.loc[j,"target_date"] = date
                table.loc[j,"timepoint"] = timepoint
            res.append(table)
        rep_tmp = pd.concat(res, axis = 0)
        rep_tmp = rep_tmp.reset_index(drop=True)
        rep_tmp["timepoint"] = rep_tmp["timepoint"].astype(int)
        self.CPA_table = rep_tmp

    def check_exist_in_CPA(self, id_hashed_candi, date_candi, timepoint_candi):
        flg = (self.CPA_table["id_hashed"] == id_hashed_candi)
        if(np.sum(flg)==0):
            return False
        flg = np.logical_and(flg, self.CPA_table["target_date"] == date_candi)
        if(np.sum(flg)==0):
            return False
        flg = np.logical_and(flg, self.CPA_table["timepoint"] == timepoint_candi)
        if(np.sum(flg)==0):
            return False
        else:
            return True

    #get n samples of Non-CPA data (table with id_hashed, data_codeB)
    def sample_negative_CPA(self, n):
        id_hashed = []
        target_date = []
        timepoint = []
        duration = np.empty(self.target_table.shape[0],dtype=int)
        for i in range(self.target_table.shape[0]):
            start = self.target_table["enter_Einstein"].iloc[i]
            end = self.target_table["discharge_Einstein"].iloc[i]
            duration[i] = (end - start).days + 1
        weight = np.array(duration,dtype=float)/np.sum(duration)
        while(len(id_hashed)<n):
            idx = np.random.choice(self.target_table.shape[0], p=weight)
            diff = np.random.choice(duration[idx])
            start = self.target_table["enter_Einstein"].iloc[idx]
            date = start + pd.to_timedelta("%ddays"%diff)
            id_hashed_candi = self.target_table["id_hashed"].iloc[idx]
            timepoint_candi = np.random.choice(self.config["learning_param"]["n_target_timepoints_per_day"])  
            if(not self.check_exist_in_CPA(id_hashed_candi, date, timepoint_candi)):
                id_hashed.append(id_hashed_candi)
                target_date.append(date)
                timepoint.append(timepoint_candi)
        return pd.DataFrame({"id_hashed":id_hashed, "target_date":target_date,
                            "timepoint":timepoint})
    
    def load_feature(self, table):
        res = []
        res.append(self.seisitsu.query_all(table["id_hashed"],table["target_date"]))
        for builder in self.feature_list:
            res.append(builder.query_table(table))
        return np.concatenate(res,axis=1)

    def build_dataset(self, n_neg_train_ratio=1.0):
        n_positive = self.CPA_table.shape[0]
        n_negative = int(n_positive * n_neg_train_ratio)
        y = np.concatenate([np.ones(n_positive),np.zeros(n_negative)],axis=0)
        neg_id = self.sample_negative_CPA(n_negative)
        all_id = pd.concat([self.CPA_table, neg_id],axis=0)
        X = self.load_feature(all_id)
        return X, y
    
if __name__ == "__main__":
    import json
    from DataLoader import DataLoader
    with open("./config.json", "r") as f:
        config = json.load(f)
    
    train_loader = DataLoader(config, config["learning_param"]["train_csv_root"])
    train_handler = DataHandler(train_loader,config)
    train_handler.CPA_table.to_csv("tmp1.csv")
    (train_handler.sample_negative_CPA(10)).to_csv("tmp2.csv")
