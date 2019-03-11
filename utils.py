import pandas as pd
import numpy as np
from SeisitsuLoader import SeisitsuLoader
from DataLoader import VitalDataLoader, LaboDataLoader


def bulid_target_table(path):
    target_table = pd.read_csv(path+"seishitu_codeblue_wo_future.csv")
    target_table["enter_Einstein"] = pd.to_datetime(target_table["enter_Einstein"])
    target_table["discharge_Einstein"] = pd.to_datetime(target_table["discharge_Einstein"])
    target_table["date_codeB"] = pd.to_datetime(target_table["date_codeB"])
    return target_table

def expand_CPA(table, n_timepoint=1):
    rep_tmp = pd.concat([table for i in range(n_timepoint)], axis = 0)
    rep_tmp = rep_tmp.reset_index(drop=True)
    time_points = np.concatenate([i*np.ones(table.shape[0],dtype=int) for i in range(n_timepoint)])
    res = pd.concat([rep_tmp, pd.Series(time_points)], axis = 1)
    res = res.rename(columns = {0:"timepoint"})
    return res

#get all CPA data (table with id_hased, date_codeB, timepoint)
def get_all_CPA(target_table):
    tmp = target_table.loc[target_table["CPA"] == True,["id_hashed","date_codeB"]]
    tmp.columns = ["id_hashed","target_date"]
    tmp = tmp.reset_index(drop=True)
    return tmp
    
#get n samples of Non-CPA data (table with id_hashed, data_codeB)
def sample_negative_CPA(n, target_table, n_timepoints =1):
    id_hashed = []
    target_date = []
    timepoint = []
    duration = np.empty(target_table.shape[0],dtype=int)
    for i in range(target_table.shape[0]):
        start = target_table["enter_Einstein"].iloc[i]
        end = target_table["discharge_Einstein"].iloc[i]
        if(target_table["CPA"].iloc[i]):
            duration[i] = (end - start).days
        else:
            duration[i] = (end - start).days + 1
    weight = np.array(duration,dtype=float)/np.sum(duration)
    while(len(id_hashed)<n):
        idx = np.random.choice(target_table.shape[0], p=weight)
        diff = np.random.choice(duration[idx])
        start = target_table["enter_Einstein"].iloc[idx]
        date = start + pd.to_timedelta("%ddays"%diff)
        if(target_table["CPA"].iloc[idx] and
           date >= target_table["date_codeB"].iloc[idx]):
           date = date + pd.to_timedelta("1day")
        id_hashed.append(target_table["id_hashed"].iloc[idx])
        target_date.append(date)
        timepoint.append(np.random.choice(n_timepoints))
    return pd.DataFrame({"id_hashed":id_hashed, "target_date":target_date,
                         "timepoint":timepoint})

#build feature matrix from table with id_hashed, data_codeB
def load_feature(path, idx_table, seisitsu = True, vital = True, labo = True, nDay = 7, fill_missing = True):
    res = []
    if(seisitsu):
        print("load seisitsu...")
        dl = SeisitsuLoader(path+"seishitu_codeblue_wo_future.csv")
        res.append(dl.query_all(idx_table["id_hashed"],idx_table["target_date"]))
    
    if(vital):
        print("load blood pressure...")
        dl = VitalDataLoader(path+"vitals/bp.csv", timepoints=3)
        res.append(dl.query_all_table(idx_table,["sBP","dBP"], nDay = nDay, fill_missing=fill_missing))
        print("load heart rate...")
        dl = VitalDataLoader(path+"vitals/hr.csv", timepoints=3)
        res.append(dl.query_all_table(idx_table,"hr", nDay = nDay, fill_missing=fill_missing))
        print("load rr")
        dl = VitalDataLoader(path+"vitals/rr.csv", timepoints=3)
        res.append(dl.query_all_table(idx_table,"rr",  nDay = nDay, fill_missing=fill_missing))
        print("load saturation")
        dl = VitalDataLoader(path+"vitals/saturation.csv", timepoints=3)
        res.append(dl.query_all_table(idx_table,"saturation", nDay = nDay, fill_missing=fill_missing))
        print("load temp")
        dl = VitalDataLoader(path+"vitals/temp.csv", timepoints=3)
        res.append(dl.query_all_table(idx_table,"temp", nDay = nDay, fill_missing=fill_missing))
        print("load urine")
        dl = VitalDataLoader(path+"vitals/urine.csv", timepoints=3)
        res.append(dl.query_all_table(idx_table,"urine", nDay = nDay, fill_missing=fill_missing))

    if(labo):
        print("load cbc")
        dl = LaboDataLoader(path+"CBC_Coag_Chemo/cbc.csv")
        cbc_list = ["wbc","rbc","hb","hct","mcv","mch","mchc","plt","rdw"]
        res.append(dl.query_all_table(idx_table,cbc_list, nDay = nDay, fill_missing=fill_missing))
        print("load chemo")
        dl = LaboDataLoader(path+"CBC_Coag_Chemo/chemo.csv")
        chemo_list = ["tp","alb","ast","alt","ld","ALP","gGTP","t_bil","i_bil","ck","ck_mb","bun","cre","eGFR","sodium","potassium","chrol","glu","crp","calcium","magnesium","bnp"]
        res.append(dl.query_all_table(idx_table,chemo_list, nDay = nDay, fill_missing=fill_missing))
        print("load coag")
        dl = LaboDataLoader(path+"CBC_Coag_Chemo/coag.csv")
        coag_list = ["pt_inr","pt","pt_sec","aptt"]
        res.append(dl.query_all_table(idx_table,coag_list, nDay = nDay, fill_missing=fill_missing))
    return np.concatenate(res,axis=1)

def build_dataset(path, target_table, pos_id, n_negative=None, seisitsu = True, vital = True, labo = True, n_timepoint=1, nDay =7):
    n_positive = pos_id.shape[0]
    if(n_negative is None):
        n_negative = n_positive
    y = np.concatenate([np.ones(n_positive),np.zeros(n_negative)],axis=0)
    neg_id = sample_negative_CPA(n_negative, target_table, n_timepoint)
    all_id = pd.concat([pos_id,neg_id],axis=0)
    X = load_feature(path, all_id,seisitsu, vital, labo, nDay)
    return X, y
    



if __name__ == "__main__":
    from SeisitsuLoader import SeisitsuLoader
    table = bulid_target_table("/Users/liyuanxu/Dropbox/automet/data/train/")
    res = expand_CPA(get_all_CPA(table),3)
    print(expand_CPA(res,3))
    print(res.iloc[-100:,:])
    print(res.columns)
    #print(load_feature(res))
