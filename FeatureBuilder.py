import numpy as np
import pandas as pd
from DataLoader import DataLoader
from math import ceil

class FeatureBuilder:

    def __init__(self, ind_param, org_params, data_loader):
        self.name = ind_param["name"]
        self.file_name = ind_param["file_name"]
        self.feature_type = ind_param["feature_type"]
        if(self.feature_type == "vital"):
            default = org_params["Vital_setting"]
        elif(self.feature_type == "labo"):
            default = org_params["Labo_setting"]
        else:
            print("Invalid Feature Type %s"%self.feature_type)
            raise ValueError
        
        self.n_timepoints = ind_param.get("n_timepoints", default["n_timepoints"])
        self.n_timepoints_per_day = ind_param.get("n_timepoints_per_day", default["n_timepoints_per_day"])
        self.n_offset_timepoints = ind_param.get("n_offset_timepoints",default["n_offset_timepoints"])
        self.na_methods = ind_param.get("NA_methods",default["NA_methods"])
        self.na_breaks =  ind_param.get("NA_breaks",default["NA_methods"])
        self.n_target_timepoints_per_day = org_params["learning_param"]["n_target_timepoints_per_day"]
        if(self.na_methods == "Imputation"):
            self.search_for_imputation = ind_param.get("search_for_imputation",default["search_for_imputation"])
        else:
            self.search_for_imputation = 0
        self.table = data_loader[self.file_name]

    def cal_timepoint_diff(self, s_date, s_time, e_date, e_time):
        diff = (e_date - s_date).dt.days.values
        return diff*self.n_timepoints_per_day + (e_time-s_time)

    def translate_target_time_to_feature_time(self, target_time):
        tmp = target_time * self.n_timepoints_per_day
        res = tmp / self.n_target_timepoints_per_day
        return int(ceil(res))
        

    def lookup_subtable(self, id_hashed, target_day, target_time):
        sub_table = self.table.loc[(self.table["id_hashed"]==int(id_hashed)),:]
        target_time = self.translate_target_time_to_feature_time(target_time)
        assert target_time <= self.n_timepoints_per_day
        if("timepoint" in sub_table.columns):
            t_diff = self.cal_timepoint_diff(sub_table["date_Einstein"], sub_table["timepoint"],target_day, target_time)
        else:
            t_diff = self.cal_timepoint_diff(sub_table["date_Einstein"], 1,target_day, target_time)
        flgs = (t_diff < self.n_timepoints + self.n_offset_timepoints + self.search_for_imputation) 
        flgs = np.logical_and(flgs, t_diff >= self.n_offset_timepoints)
        sub_table = sub_table.loc[flgs,self.name]
        t_diff = t_diff[flgs]
        return sub_table, t_diff

    #idx_table.columns = [id_hashed, target_day, target_time]
    def query_table(self, idx_table, res = None):
        res = np.empty((idx_table.shape[0], self.n_timepoints+ self.search_for_imputation))
        res[:] = np.nan
        for i in range(idx_table.shape[0]):
            id_hashed = idx_table["id_hashed"].values[i]
            target_date = idx_table["target_date"].values[i]
            timepoint = idx_table["timepoint"].values[i]
            sub_table, t_diff = self.lookup_subtable(id_hashed,target_date,timepoint)
            col_id = -(t_diff-self.n_offset_timepoints)-1
            res[i,col_id] = sub_table.values
        return self.fill_na(res)
    
    def fill_na(self, table):
        if(self.na_methods == "Imputation"):
            return self.fill_na_imputation(table)
        elif(self.na_methods == "Binary"):
            return self.fill_na_binary(table)
        elif(self.na_methods == "Categorical"):
            return self.fill_na_cat(table)
        else:
            print("invalid na methods %s"%self.na_methods)
            raise ValueError
    
    def fill_na_imputation(self, table):
        mis_val = self.table[self.name].mean()
        tmp = np.array(table[:,0])
        tmp[np.isnan(tmp)] = mis_val
        for i in range(table.shape[1]):
            flg = np.logical_not(np.isnan(table[:,i]))
            tmp[flg] = table[flg,i] 
            table[:,i]  =  tmp
        return table[:,self.search_for_imputation:]

    def fill_na_binary(self,table):
        return np.array(np.isnan(table), dtype=int, copy = False)
    
    def fill_na_cat(self, table):
        nRow, nCol = table.shape
        nData =  nRow * nCol
        binned_data = pd.cut(table.reshape(nData), bins = self.na_breaks)
        return binned_data.codes.reshape(nRow, nCol)


if __name__ == "__main__":
    import json
    with open("./config.json", "r") as f:
        config = json.load(f)
    loader = DataLoader(config, config["learning_param"]["train_csv_root"])
    ind0 = config["feature"][0]
    ind1 = config["feature"][7]
    fb = FeatureBuilder(ind0, config, loader)
    fb1 = FeatureBuilder(ind1, config, loader)
    print(fb.n_target_timepoints_per_day)
    print(fb.lookup_subtable(1075937, pd.to_datetime("1973-12-20"),3))
    print(fb1.lookup_subtable(1063622, pd.to_datetime("1961-05-03"),3))
    
    idx_table = pd.DataFrame.from_dict({"id_hashed" : ["151793","151793"], 
                              "target_date": pd.to_datetime(["1927-02-24","1927-02-25"]),
                              "timepoint":[1,1]})
    print(idx_table)
    print(fb1.query_table(idx_table))
    