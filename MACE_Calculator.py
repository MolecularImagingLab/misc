#!/usr/bin/env python

import pandas as pd
import numpy as np
dataframe = pd.read_csv('/home/lauri/Documents/mace_sample.csv')

""" Reverse item scoring. """
def reverse_item_fx(dframe, indx):
    for i in range(-18,1):
        inx = indx+i
        dframe.iloc[:,inx] = abs(1-dframe.iloc[:,inx])
    return dframe

""" Checks if number of missing values are acceptable based on cutoffs defined in mace_calculate(). """
def mace_cutoff_fx(matrix1, cutoff):
    for item in range(0,len(matrix1)):
        if cnt[item] < cutoff:
            cnt[item] = 0
        else:
            cnt[item] = 1
    return cnt

""" Mace item scoring. """
def mace_age_fx(dframe, indx, cutoff, scoring, min_acceptable):
    x = np.array([0])
    repeats = len(dframe.index)*19*2
    out_mat = np.repeat(x, repeats)
    out_mat = np.reshape(out_mat, [len(dframe.index), 19*2])
    df = pd.DataFrame(data=out_mat)
    j = 0
    for i in range(-18,1):    
        j = -j +1
        k = -j +19
        inx = indx+i
        hold = dataframe.iloc[:,inx]
        lnt = (hold.apply(lambda x:(hold.count(axis=1))).iloc[:,0]).to_numpy()
        cnt = ((hold.apply(lambda x:(hold.isna().sum(axis=1)))).iloc[:,0]).to_numpy()
        for item in range(0,9):
            if lnt[item] < min_acceptable:
                cnt[item] = np.nan
        mace = mace_cutoff_fx(cnt, cutoff)
        qz = np.array2string(mace)
        mace_sum = scoring[(cnt+1)]
        df.iloc[:,j] = mace
        df.iloc[:,k] = mace_sum
    return df

""" Column indices to form MACE subscores. """
def mace_indices():
    sex_ab_indx = np.array([230,249,268,363,382,686,705]) #c("Sex_comment_Childhood","Fondled_Childhood","Touch_them_Childhood","o_touch_them","o_intercourse_Childhood","Peer_forced_sex_Childhood","Peer_sex_not_want_Childhood")
    pva_indx = np.array([21,40,59,78]) #c("Swore_Childhood","Hurtful_Childhood","Afraid_Childhood","Leave_Childhood")
    nvea_indx = np.array([97,762,781,914,933,952]) #c("Closet_Childhood","P_diff_please_Childhood","P_no_time_Childhood","Adult_resposibility_Childhood","Financial_pressure_Childhood","Kept_secrets_Childhood")
    phys_ab_indx = np.array([116,135,154,173,192,211]) #c("Pushed_Childhood","Hit_Childhood","Hit_med_Childhood","Spank_buttocks_Childhood","Spanked_bare_Childhood","Spanked_strap_Childhood")
    wipv_indx = np.array([401,420,439,458,477]) #c("Adults_push_m_Childhood","Adults_hit_m_Childhood","Adults_hit_med_m_Childhood","Adults_push_f_Childhood","Adults_hit_f_Childhood")
    viol_sib_indx = np.array([287,306,325,344])	#c("Hit_sib_Childhood","Hit_sib_med_Childhood","Sex_comment_sib_Childhood","Fondled_sib_Childhood")
    peer_emot_indx = np.array([496,515,534,553,572]) #c("peer_swore_Childhood","Peer_hurtful_Childhood","Peer_Rumors_Childhood","Peer_Excluded_Childhood","Peer_afraid_Childhood")
    peer_phys_indx = np.array([591,610,629,648,667]) #c("Peer_threat_money_Childhood","Peer_forced_Childhood","Peer_pushed_Childhood","Peer_hit_Childhood","Peer_hit_med_Childhood")
    emot_negl_indx = np.array([724,743,800,819,990]) #c("M_unavail_poor_Childhood","F_unavail_poor_Childhood","R_P_loved_you_Childhood","R_P_special_Childhood","R_Fam_strength_Childhood")
    em_negl_rev = np.array([800,819,990])
    phys_negl_indx = np.array([838,857,876,895,971]) #c("R_P_protect_Childhood","R_P_ER_Childhood","No_food_Childhood","Dirty_clothes_Childhood","R_Looked_out_each_other_Childhood")
    phys_negl_rev = np.array([838,857,971])
    spank_indx = np.array([173,192,211])
    look_out = np.array(range(953,970))
    protect = np.array(range(820,837))
    er_md = np.array(range(839,856))
    ever_indx = np.concatenate([sex_ab_indx,pva_indx,nvea_indx,phys_ab_indx,wipv_indx,viol_sib_indx,peer_emot_indx,peer_phys_indx,emot_negl_indx,phys_negl_indx])
    ever_indx = np.sort(ever_indx)
    scoring_indices = {'sex_ab_indx':sex_ab_indx,'pva_indx':pva_indx,'nvea_indx':nvea_indx,'phys_ab_indx':phys_ab_indx,
                      'wipv_indx':wipv_indx, 'viol_sub_indx':viol_sib_indx, 'peer_emot_indx':peer_emot_indx, 'peer_phys_indx':peer_phys_indx,
                      'emot_negl_indx':emot_negl_indx, 'em_negl_rev':em_negl_rev, 'phys_negl_indx':phys_negl_indx, 'phys_negl_rev':phys_negl_rev,
                      'spank_indx':spank_indx, 'look_out':look_out, 'protect':protect, 'er_md':er_md, 'ever_indx':ever_indx}
    return scoring_indices

""" MACE item response theory (IRT) values based on Teicher & Parigger (2015). """
def mace_irt_values():
    sex_ab_scoring = np.array([0, 1.84063982566719, 3.57167393405874, 4.86193482048058, 6.05874975634753, 7.35545905210534, 8.67772952605267, 10])
    pva_scoring = np.array([0, 2.5, 5, 7.5, 10])
    nvea_scoring = np.array([0, 1.57203586201709, 3.07318170239233, 4.30017268552622, 5.66888216594181, 7.74694728216119, 10])
    phys_ab_scoring = np.array([0, 1.97350025645546, 3.80610533906172, 5.07761998873815, 6.34283867353684, 8.10910025721665, 10])
    wipv_scoring = np.array([0, 2.16265962002852, 4.24162614623762, 6.04358509621965, 7.99291014858454, 10])
    viol_sib_scoring = np.array([0, 2.5, 5, 7.5, 10])
    peer_emot_scoring = np.array([0, 2.01491889140193, 3.94389919669921, 5.65181683198404, 7.76456257878893, 10])
    peer_phys_scoring = np.array([0, 2.31610945680412, 4.49115107318588, 6.18181081484338, 8.05096066393912, 10])
    emot_negl_scoring = np.array([0, 2.06604852199299, 4.06369282630551, 5.86280636891976, 7.89386012994807, 10])
    phys_negl_scoring = np.array([0, 2.16341492872456, 4.2156227144729, 5.92172308914546, 7.91150096415818, 10])
    scoring_vals = {'sex_ab_scoring':sex_ab_scoring, 'pva_scoring':pva_scoring, 'nvea_scoring':nvea_scoring, 'phys_ab_scoring':phys_ab_scoring, 
                    'wipv_scoring':wipv_scoring, 'viol_sib_scoring':viol_sib_scoring, 'peer_emot_scoring':peer_emot_scoring, 
                    'peer_phys_scoring':peer_phys_scoring, 'emot_negl_scoring':emot_negl_scoring, 'phys_negl_scoring':phys_negl_scoring}
    for score in scoring_vals.keys():
        scoring_vals[score] = np.round(scoring_vals[score])
    return scoring_vals

""" MACE scoring function. """
def mace_calculate(dataframe, start_col):
    x = mace_irt_values()
    cutoffs = np.array([2,3,4,4,2,1,4,2,2,2])
    min_acceptable = np.array([5,3,4,4,3,2,4,4,4,4])
    min_acc_intrafam = 5
    min_acc_total = 8
    a = mace_indices()
    num_cols=len(dataframe.columns) - start_col
    list_cols = [0,1]
    list_cols.extend(list(range(start_col,num_cols+1)))
    ever_hold = dataframe.iloc[: , list_cols].copy()
    # Calculate results
    # Sexual Abuse Scores
    sex_ab_hold = mace_age_fx(ever_hold, a['sex_ab_indx'], cutoffs[0], x['sex_ab_scoring'], min_acceptable[0])
    sex_ab_mace = sex_ab_hold.iloc[:,18]
    sex_ab_sum = sex_ab_hold.iloc[:,37]
    final_out = sex_ab_hold

""" Main function to process 52 question MACE, total of 988 items. """
def mace_execute(dataframe, start_col):
    start_col=start_col
    dataframe=dataframe
    num_cols=len(dataframe.columns) - start_col
    if num_cols != 988:
        print("Dataframe has",num_cols,"columns for scoring. Only 988 columns valid for scoring.")
    else:
        mace_calculate(dataframe, start_col)
        print("Dataframe has 988 columns for scoring. Calculating...")

