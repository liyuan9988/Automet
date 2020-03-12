from DataHandler import DataHandler
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from math import ceil


class Evaluator:

    def __init__(self, config, test_loader, test_handler):
        self.config = config
        self.test_loader = test_loader
        self.test_handler = test_handler

    def build_dataset(self, negative_ratio, nprocess=1):
        key = self.config["learning_param"].get("eval_key", "all")
        icu = self.config["learning_param"].get("icu_path", None)
        n_positive = self.test_handler.CPA_table.shape[0]
        n_negative = int(n_positive * negative_ratio)
        print("n_negative = %d" % n_negative)
        if(key == "MedSurg"):
            self.build_dataset_for_MedSurg(negative_ratio, nprocess)
        elif(key == "ICU"):
            self.build_dataset_for_ICU(negative_ratio, icu, nprocess)
        else:
            self.build_dataset_for_all(negative_ratio, nprocess)

    def lookup_for_ICU(self, icu_table, target_id, target_day, target_time, delta_time):
        flg = (icu_table["id_hashed"] == target_id)
        true_target_day = target_day - pd.to_timedelta("%d day" % ((target_time-delta_time)//self.config["learning_param"]["n_target_timepoints_per_day"]))
        flg = np.logical_and(flg, true_target_day >= icu_table["enter_ICU_Einstein"])
        flg = np.logical_and(flg, true_target_day <= icu_table["discharge_ICU_Einstein"])
        return np.sum(flg) > 0

    def build_dataset_for_ICU(self, negative_ratio, icu, nprocess):
        n_positive = self.test_handler.CPA_table.shape[0]
        n_negative = int(n_positive * negative_ratio)
        NonCPA_target = self.test_handler.sample_negative_CPA(n_negative, False)
        icu_table = pd.read_csv(icu)
        icu_table["enter_ICU_Einstein"] = pd.to_datetime(icu_table["enter_ICU_Einstein"])
        icu_table["discharge_ICU_Einstein"] = pd.to_datetime(icu_table["discharge_ICU_Einstein"])
        vital_length = self.config["Vital_setting"]["n_timepoints"]
        vital_timepoints_per_day = self.config["Vital_setting"]["n_timepoints_per_day"]
        target_timepoints_per_day = self.config["learning_param"]["n_target_timepoints_per_day"]
        delta_time = int(ceil(vital_length / vital_timepoints_per_day * target_timepoints_per_day))
        non_cpa_icu_flgs = NonCPA_target.apply(lambda x: self.lookup_for_ICU(icu_table, x["id_hashed"], x["target_date"], x["timepoint"], delta_time), axis=1)
        cpa_icu_flgs = self.test_handler.CPA_table.apply(lambda x: self.lookup_for_ICU(icu_table, x["id_hashed"], x["target_date"], x["timepoint"], delta_time), axis=1)

        # compile_Meg
        icuCPA = self.test_handler.CPA_table.loc[cpa_icu_flgs, :]
        icuNonCPA = NonCPA_target.loc[non_cpa_icu_flgs, :]
        icu_n_pos = icuCPA.shape[0]
        icu_n_neg = icuNonCPA.shape[0]
        if(icu_n_neg == 0):
            print((icu_n_pos, icu_n_neg))
            raise ValueError("No negative data in icu")
        icu_y = np.concatenate([np.ones(icu_n_pos), np.zeros(icu_n_neg)], axis=0)
        icu_X = self.test_handler.load_feature(pd.concat([icuCPA, icuNonCPA], axis=0), nprocess)

        # compile_Surg
        wardCPA = self.test_handler.CPA_table.loc[~cpa_icu_flgs, :]
        wardNonCPA = NonCPA_target.loc[~non_cpa_icu_flgs, :]
        ward_n_pos = wardCPA.shape[0]
        ward_n_neg = wardNonCPA.shape[0]
        if(ward_n_neg == 0):
            print((ward_n_pos, ward_n_neg))
            raise ValueError("No negative data in ward")
        ward_y = np.concatenate([np.ones(ward_n_pos), np.zeros(ward_n_neg)], axis=0)
        ward_X = self.test_handler.load_feature(pd.concat([wardCPA, wardNonCPA], axis=0), nprocess)

        self.data_list = [(icu_X, icu_y), (ward_X, ward_y)]

    def build_dataset_for_MedSurg(self, negative_ratio, nprocess):
        n_positive = self.test_handler.CPA_table.shape[0]
        n_negative = int(n_positive * negative_ratio)
        NonCPA_target, row_id = self.test_handler.sample_negative_CPA(n_negative, True)
        neg_MedSurg = self.test_handler.target_table["MedSurg"].iloc[row_id]
        neg_MedSurg.reset_index(drop=True, inplace=True)
        pos_MedSurg = self.test_handler.target_table["MedSurg"].loc[self.test_handler.target_table["CPA"] == True]
        pos_MedSurg.reset_index(drop=True, inplace=True)

        # compile_Meg
        MedCPA = self.test_handler.CPA_table.loc[pos_MedSurg == "medicine", :]
        MedNonCPA = NonCPA_target.loc[neg_MedSurg == "medicine", :]
        med_n_pos = MedCPA.shape[0]
        med_n_neg = MedNonCPA.shape[0]
        med_y = np.concatenate([np.ones(med_n_pos), np.zeros(med_n_neg)], axis=0)
        med_X = self.test_handler.load_feature(pd.concat([MedCPA, MedNonCPA], axis=0), nprocess)

        # compile_Surg
        SurgCPA = self.test_handler.CPA_table.loc[pos_MedSurg == "surgery", :]
        SurgNonCPA = NonCPA_target.loc[neg_MedSurg == "surgery", :]
        surg_n_pos = SurgCPA.shape[0]
        surg_n_neg = SurgNonCPA.shape[0]
        surg_y = np.concatenate([np.ones(surg_n_pos), np.zeros(surg_n_neg)], axis=0)
        surg_X = self.test_handler.load_feature(pd.concat([SurgCPA, SurgNonCPA], axis=0), nprocess)

        self.data_list = [(med_X, med_y), (surg_X, surg_y)]

    def build_dataset_for_all(self, negative_ratio, nprocess):
        X_test, y_test = self.test_handler.build_dataset(negative_ratio, nprocess)
        self.data_list = [(X_test, y_test)]
        print(y_test)

    def evaluate(self, models):
        res = []
        for X, y_test in self.data_list:
            y_scores = np.array([mdl.predict_proba(X) for mdl in models])
            y_score = np.mean(y_scores, axis=0, keepdims=False)[:, 1]
            res.append(self.cal_metrics(y_score, y_test))
        return np.array(res)

    def predict(self, models, data_id=0):
        X_test, y_test = self.data_list[data_id]
        predict = np.array([mdl.predict_proba(X_test) for mdl in models])
        predict = np.mean(predict, axis=0, keepdims=False)[:, 1]
        return np.array([predict, y_test])

    def simulate_on_all_negative(self, models, data_folder, n_process=1):
        non_CPA_table = self.test_handler.get_negative_CPA()
        CPA_table = self.test_handler.CPA_table
        all_table = pd.concat([CPA_table, non_CPA_table], axis=0).reset_index(drop=True)
        y = np.concatenate([np.ones(CPA_table.shape[0]), np.zeros(non_CPA_table.shape[0])], axis=0)
        all_table_tmp = pd.concat([all_table, pd.DataFrame(y,columns=["true_label"])], axis=1)
        all_table_tmp.to_csv(data_folder + "/all_idx.csv", index=False)
        print("id table built")
        X = self.test_handler.load_feature(all_table, n_process)
        np.save(data_folder + "/feature_table.npy",X)
        y_scores = np.array([mdl.predict_proba(X) for mdl in models])
        y_score = np.mean(y_scores, axis=0, keepdims=False)[:, 1]
        all_table_tmp = pd.concat([all_table_tmp, pd.DataFrame(y_score, columns=["predict_label"])], axis=1)
        all_table_tmp.to_csv(data_folder + "/all_result.csv", index=False)
        res = self.cal_metrics(y_score, y)
        np.save(data_folder + "/metrics.npy", res)
        fpr, tpr, thresholds = roc_curve(y, y_score)
        threshold_info = pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=thresholds))
        threshold_info.to_csv(data_folder + "/threshold_info.csv", index=False)



    def cal_metrics(self, y_score, y_test):
        nData = y_test.shape[0]
        nCPA = np.sum(y_test)
        n_nonCPA = np.sum(1-y_test)
        fpr, tpr, therehsolds = roc_curve(y_test, y_score)
        auc_score = auc(fpr, tpr)
        idx = np.argmax(tpr - fpr)
        youden_index = therehsolds[idx]
        tp = np.sum(y_test[y_score > youden_index])
        fp = np.sum(1.0 - y_test[y_score > youden_index])
        fn = np.sum(y_test[y_score <= youden_index])
        tn = np.sum(1.0 - y_test[y_score <= youden_index])
        accuracy = (tp+tn) / (tp+tn+fp+fn)
        PPV = tp / (tp + fp)
        NPV = tn / (tn + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f_val = (2.0*tp) / (2.0*tp + fp + fn)
        res = np.array([nData, nCPA, n_nonCPA, accuracy, PPV, NPV, sensitivity, specificity, f_val, auc_score, youden_index, tp, fp, fn, tn])
        return res
