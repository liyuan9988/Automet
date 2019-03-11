import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def call_query(loader, id_hashed, target_date, timestep, target_col, nDay):
    return loader.query_core(id_hashed, target_date, timestep, target_col, nDay)

class DataLoader:

    def __init__(self, csv_path):
        self.table = pd.read_csv(csv_path)
        self.table["date_Einstein"] = pd.to_datetime(self.table["date_Einstein"])
    
    #look up dates
    def lookup_subtable(self, id_hashed, target_date, nDay):
        sub_table = self.table.loc[self.table["id_hashed"]==int(id_hashed),:]
        diffs = pd.to_datetime(target_date) - sub_table["date_Einstein"]
        flgs = np.logical_and(diffs >= pd.to_timedelta("1day"), 
                              diffs <= pd.to_timedelta("%ddays"%nDay)) 
        sub_table = sub_table.loc[flgs]
        diffs = diffs[flgs].dt.days.values
        return sub_table, diffs

    def query_all_table(self, idx_table, target_col, nDay = 7, fill_missing=True):
        id_hashed_list = idx_table["id_hashed"]
        target_date_list = idx_table["target_date"]
        timepoint_list = idx_table["timepoint"]
        return self.query_all(id_hashed_list, target_date_list, timepoint_list, target_col, nDay, fill_missing)

    def query_all(self, id_hashed_list, target_date_list, timepoint_list, target_col,nDay = 7, fill_missing=True):
        if(isinstance(target_col, str)):
            target_col = [target_col]
        res = [self.query_core(id_hashed, target_date, timepoint, target_col, nDay) for id_hashed, target_date, timepoint in zip(id_hashed_list, target_date_list, timepoint_list)]
        
        if(fill_missing):
            return self.fill_missing(np.array(res),target_col)
        return np.array(res)

    def query_core(self, id_hashed, target_date, timepoint, target_col, nDay = 7):
        raise NotImplementedError
    
    def fill_missing(self, res, target_col):
        raise NotImplementedError




class VitalDataLoader(DataLoader):

    def __init__(self, csv_path, timepoints=3):
        super().__init__(csv_path)
        self.timepoints = timepoints
        
    def query_core(self, id_hashed, target_date, timepoint, target_col, nDay = 7):
        sub_table, diffs = self.lookup_subtable(id_hashed, target_date, nDay+1)
        n_target = len(target_col)
        res = np.empty(nDay*self.timepoints*n_target)
        offset = 0
        for j in target_col:
            res_tmp = np.array([np.nan for i in range(nDay*self.timepoints)])
            for i in range(len(diffs)):
                timestep = (sub_table["timepoint"].iloc[i])
                idx = (self.timepoints-timestep)+self.timepoints*(diffs[i]-1) - timepoint
                if(idx >= 0 and idx < len(res_tmp)):
                    res_tmp[idx] = sub_table[j].iloc[i]
            res_tmp = res_tmp[::-1]
            res[offset:offset+nDay*self.timepoints] = res_tmp
            offset += nDay*self.timepoints
        return res

    
    def fill_missing(self, res, target_col):
        Delta = res.shape[1]//len(target_col)
        offset=0
        for one_col in target_col:
            res_tmp = np.array(res[:,offset:offset+Delta])
            mis_val = self.table[one_col].mean()
            tmp = np.array(res_tmp[:,0])
            tmp[np.isnan(tmp)] = mis_val
            for i in range(res_tmp.shape[1]):
                flg = np.logical_not(np.isnan(res_tmp[:,i]))
                tmp[flg] = res_tmp[flg,i] 
                res_tmp[:,i]  =  tmp
            res[:,offset:offset+Delta] = res_tmp 
            offset += Delta
        return res



class LaboDataLoader(DataLoader):

    def __init__(self, csv_path):
        super().__init__(csv_path)

    #returns the newest date within $nDay days    
    def query_core(self, id_hashed, target_date, timepoint, target_col, nDay = 7):
        if(isinstance(target_col, str)):
            target_col = [target_col]
        sub_table, _ = self.lookup_subtable(id_hashed, target_date, nDay)
        sub_table = sub_table.sort_values("date_Einstein",ascending = False)
        res = np.empty(len(target_col))
        res[:] = np.nan
        for i,one_target in enumerate(target_col):
            if(np.isnan(res[i])):
                for j in range(sub_table.shape[0]):
                    res[i] = sub_table[one_target].iloc[j]
        return res

    def fill_missing(self, res, target_col):
        mis_val = self.table[target_col].mean()
        if(len(target_col)==1):
            res[np.isnan(res)] = mis_val
        for i in range(res.shape[0]):
            res[i, np.isnan(res[i])] = mis_val[np.isnan(res[i])]
        return res





if __name__ == "__main__":
    dl1 = LaboDataLoader("~/Dropbox/automet/data/cleaned_data_csv/CBC_Coag_Chemo/cbc.csv")
    dl2 = VitalDataLoader("~/Dropbox/automet/data/cleaned_data_csv/vitals/bp.csv",timepoints=3)
    print(dl1.query_all(["151793","151793"],["1927-02-24","1927-02-25"],[0,0],"wbc"))
    print(dl2.query_all(["151793"],["1927-02-24"],[0],["dBP"]))
    print(dl2.query_all(["151793"],["1927-02-24"],[1],["dBP"]))