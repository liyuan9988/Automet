import numpy as np
import pandas as pd

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
        self.n_timepoints_per_day = ind_param.get("n_timepoints_per_day", default["n_timepoints_per_day"]),
        self.n_offset_timepoints = ind_param.get("n_offset_timepoints",default["n_offset_timepoints"])
        self.na_methods = ind_param.get("NA_methods",default["NA_methods"])
        self.na_breaks =  ind_param.get("NA_breaks",default["NA_methods"])
        self.table = data_loader[self.file_name].loc[:,["id_hashed", "date_Einstein", self.name]]


    def cal_date_and_timepoint(self, date, timepoint, diff_timepoint):
        final_time = timepoint + diff_timepoint
        diff_date = final_time // self.n_timepoints_per_day
        final_time = final_time % self.n_timepoints_per_day
        finalDate = date + pd.to_timedelta("%sday"%diff_date)
        return finalDate, final_time
        

    def lookup_subtable(self, id_hashed, target_date, nDay):
        sub_table = self.table.loc[self.table["id_hashed"]==int(id_hashed),:]
        diffs = pd.to_datetime(target_date) - sub_table["date_Einstein"]
        flgs = np.logical_and(diffs >= pd.to_timedelta("1day"), 
                              diffs <= pd.to_timedelta("%ddays"%nDay)) 
        sub_table = sub_table.loc[flgs]
        diffs = diffs[flgs].dt.days.values
        return sub_table, diffs

if __name__ == "__main__":
    
