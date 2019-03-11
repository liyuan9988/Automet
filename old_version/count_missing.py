import numpy as np
from utils import get_all_CPA, load_feature, expand_CPA, sample_negative_CPA
from DataLoader import VitalDataLoader

def count_missing_CPA(nDay, n_timepoint, dl, col_name):
    pos_idx = get_all_CPA()
    pos_idx_table = expand_CPA(pos_idx, n_timepoint)
    res = dl.query_all_table(pos_idx_table, [col_name], nDay = nDay, fill_missing=False)
    return np.sum(np.isnan(res))/(res.shape[0]*res.shape[1])


def count_missing_non_CPA(nDay, n_timepoint, dl, col_name, neg_idx_table):
    res = dl.query_all_table(neg_idx_table, [col_name], nDay = nDay, fill_missing=False)
    return np.sum(np.isnan(res))/(res.shape[0]*res.shape[1])

def return_one_row_CPA(dl, col_name):
    res = np.empty(21)
    i = 0
    for nDay in [7,6,5,4,3,2,1]:
        for n_timepoint in [3,2,1]:
            res[i] = count_missing_CPA(nDay, n_timepoint, dl, col_name)
            i+=1
            print(i)
    return res

def return_one_row_non_CPA(dl, col_name, negative_table_list):
    res = np.empty(21)
    i = 0
    for nDay in [7,6,5,4,3,2,1]:
        for n_timepoint in [2,1,0]:
            res[i] = count_missing_non_CPA(nDay, n_timepoint, dl, col_name, negative_table_list[n_timepoint])
            i+=1
            print(i)
    return res


if __name__ == "__main__":
    path = "~/Dropbox/automet/data/cleaned_data_csv/"
    result = []
    negative_table_list = [sample_negative_CPA(1000, i) for i in [1,2,3]]
    dl = VitalDataLoader(path+"vitals/bp.csv", timepoints=3)
    result.append(return_one_row_CPA(dl, "sBP"))
    result.append(return_one_row_non_CPA(dl, "sBP",negative_table_list))
    result.append(return_one_row_CPA(dl, "dBP"))
    result.append(return_one_row_non_CPA(dl, "dBP",negative_table_list))
    print("bp ended")

    dl = VitalDataLoader(path+"vitals/hr.csv", timepoints=3)
    result.append(return_one_row_CPA(dl, "hr"))
    result.append(return_one_row_non_CPA(dl, "hr",negative_table_list))
    print("hr ended")

    dl = VitalDataLoader(path+"vitals/saturation.csv", timepoints=3)
    result.append(return_one_row_CPA(dl, "saturation"))
    result.append(return_one_row_non_CPA(dl, "saturation",negative_table_list))
    print("saturation ended")

    
    dl = VitalDataLoader(path+"vitals/rr.csv", timepoints=3)
    result.append(return_one_row_CPA(dl, "rr"))
    result.append(return_one_row_non_CPA(dl, "rr",negative_table_list))
    print("rr ended")

    dl = VitalDataLoader(path+"vitals/temp.csv", timepoints=3)
    result.append(return_one_row_CPA(dl, "temp"))
    result.append(return_one_row_non_CPA(dl, "temp",negative_table_list))
    print("temp ended")

    dl = VitalDataLoader(path+"vitals/urine.csv", timepoints=3)
    result.append(return_one_row_CPA(dl, "urine"))
    result.append(return_one_row_non_CPA(dl, "urine",negative_table_list))
    print("loaded")
    
    result = np.array(result)
    np.savetxt("bar.csv", result, delimiter=",", fmt='%.2f')



    


