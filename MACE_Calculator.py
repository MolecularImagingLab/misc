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
def mace_cutoff_fx(cnt, cutoff):
    for item in range(0,len(cnt)):
        if cnt[item] < cutoff:
            cnt[item] = 0
        else:
            cnt[item] = 1
    return cnt

""" Mace item scoring. """
def mace_age_fx(dframe, indx, cutoff, scoring, min_acceptable):
    out_mat = np.zeros([dframe.shape[0],19*2])
    df = pd.DataFrame(data=out_mat)
    j = 0
    for i in range(-18,1):    
        j = -j +1
        k = -j +19
        inx = indx+i
        hold = dataframe.iloc[:,inx]
        lnt = (hold.count(axis=1)).to_numpy()
        cnt = (hold.sum(axis=1, skipna=True)).to_numpy(dtype='float32')
        for item in range(0,len(dframe.index)):
            if lnt[item] < min_acceptable:
                cnt[item] = np.nan
        mace = mace_cutoff_fx(cnt, cutoff)
        qz = np.array2string(mace)
        for c in cnt:
            c = int(c)
            mace_sum = scoring[c] #c+1 in r-code
        df.iloc[:,j] = mace
        df.iloc[:,k] = mace_sum
    return df

""" Column indices to form MACE subscores. """
def mace_indices(): # index+1 in r-code
    sex_ab_indx = np.array([229,248,267,362,381,685,704]) #c("Sex_comment_Childhood","Fondled_Childhood","Touch_them_Childhood","o_touch_them","o_intercourse_Childhood","Peer_forced_sex_Childhood","Peer_sex_not_want_Childhood")
    pva_indx = np.array([20,39,58,77]) #c("Swore_Childhood","Hurtful_Childhood","Afraid_Childhood","Leave_Childhood")
    nvea_indx = np.array([96,761,780,913,932,951]) #c("Closet_Childhood","P_diff_please_Childhood","P_no_time_Childhood","Adult_resposibility_Childhood","Financial_pressure_Childhood","Kept_secrets_Childhood")
    phys_ab_indx = np.array([115,134,153,172,191,210]) #c("Pushed_Childhood","Hit_Childhood","Hit_med_Childhood","Spank_buttocks_Childhood","Spanked_bare_Childhood","Spanked_strap_Childhood")
    wipv_indx = np.array([400,419,438,457,476]) #c("Adults_push_m_Childhood","Adults_hit_m_Childhood","Adults_hit_med_m_Childhood","Adults_push_f_Childhood","Adults_hit_f_Childhood")
    viol_sib_indx = np.array([286,305,324,343])	#c("Hit_sib_Childhood","Hit_sib_med_Childhood","Sex_comment_sib_Childhood","Fondled_sib_Childhood")
    peer_emot_indx = np.array([495,514,533,552,571]) #c("peer_swore_Childhood","Peer_hurtful_Childhood","Peer_Rumors_Childhood","Peer_Excluded_Childhood","Peer_afraid_Childhood")
    peer_phys_indx = np.array([590,609,628,647,666]) #c("Peer_threat_money_Childhood","Peer_forced_Childhood","Peer_pushed_Childhood","Peer_hit_Childhood","Peer_hit_med_Childhood")
    emot_negl_indx = np.array([723,742,799,818,989]) #c("M_unavail_poor_Childhood","F_unavail_poor_Childhood","R_P_loved_you_Childhood","R_P_special_Childhood","R_Fam_strength_Childhood")
    em_negl_rev = np.array([799,818,989])
    phys_negl_indx = np.array([837,856,875,894,970]) #c("R_P_protect_Childhood","R_P_ER_Childhood","No_food_Childhood","Dirty_clothes_Childhood","R_Looked_out_each_other_Childhood")
    phys_negl_rev = np.array([837,856,970])
    spank_indx = np.array([172,191,210])
    look_out = np.array(range(952,969))
    protect = np.array(range(819,836))
    er_md = np.array(range(838,855))
    ever_indx = np.concatenate([sex_ab_indx,pva_indx,nvea_indx,phys_ab_indx,wipv_indx,viol_sib_indx,peer_emot_indx,peer_phys_indx,emot_negl_indx,phys_negl_indx])
    ever_indx = np.sort(ever_indx)
    scoring_indices = {'sex_ab_indx':sex_ab_indx,'pva_indx':pva_indx,'nvea_indx':nvea_indx,'phys_ab_indx':phys_ab_indx,
                      'wipv_indx':wipv_indx, 'viol_sib_indx':viol_sib_indx, 'peer_emot_indx':peer_emot_indx, 'peer_phys_indx':peer_phys_indx,
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
    num_cols=len(dataframe.columns)+1 - start_col
    list_cols = [0,1]
    list_cols.extend(list(range(start_col,num_cols+1)))
    ever_hold = dataframe.iloc[: , list_cols].copy()
    # Calculate results
    # Sexual Abuse Scores
    sex_ab_hold = mace_age_fx(ever_hold, a['sex_ab_indx'], cutoffs[0], x['sex_ab_scoring'], min_acceptable[0])
    sex_ab_mace = (sex_ab_hold.iloc[:,18]).astype('int64')
    sex_ab_sum = sex_ab_hold.iloc[:,37]
    final_out = sex_ab_hold
    sex_ab_fam = ever_hold.iloc[:,(a['sex_ab_indx'][0:2])].sum(axis=1, skipna=True)
    sex_ab_fam.where(sex_ab_fam <1, other=1, inplace=True)
    sex_ab_fam.astype('int64')
    sex_ab_fam = sex_ab_mace & sex_ab_fam
    # Parental Verbal Abuse Scores
    pva_hold = mace_age_fx(ever_hold, a['pva_indx'], cutoffs[1], x['pva_scoring'], min_acceptable[1])
    pva_mace = pva_hold.iloc[:,18]
    pva_sum = pva_hold.iloc[:,37]
    final_out = pd.concat([final_out, pva_hold], axis=1)
    # Parental Non-Verbal Abuse Scores
    nvea_hold = mace_age_fx(ever_hold, a['nvea_indx'], cutoffs[2], x['nvea_scoring'], min_acceptable[2])
    nvea_mace = nvea_hold.iloc[:,18]
    nvea_sum = nvea_hold.iloc[:,37]
    final_out = pd.concat([final_out, nvea_hold], axis=1)
    # Parental Physical Maltreatment Scores
    phys_ab_hold = mace_age_fx(ever_hold, a['phys_ab_indx'], cutoffs[3], x['phys_ab_scoring'], min_acceptable[3])
    phys_ab_mace = phys_ab_hold.iloc[:,18]
    phys_ab_sum = phys_ab_hold.iloc[:,37]
    final_out = pd.concat([final_out, phys_ab_hold], axis=1)
    # Witnessing Interparental Violence Scores
    wipv_hold = mace_age_fx(ever_hold, a['wipv_indx'], cutoffs[4], x['wipv_scoring'], min_acceptable[4])
    wipv_mace = wipv_hold.iloc[:,18]
    wipv_sum = wipv_hold.iloc[:,37]
    final_out = pd.concat([final_out, wipv_hold], axis=1)
    # Witnessing Sibling Abuse Scores
    viol_sib_hold = mace_age_fx(ever_hold, a['viol_sib_indx'], cutoffs[5], x['viol_sib_scoring'], min_acceptable[5])
    viol_sib_mace = viol_sib_hold.iloc[:,18]
    viol_sib_sum = viol_sib_hold.iloc[:,37]
    final_out = pd.concat([final_out, viol_sib_hold], axis=1)
    # Peer Emotional Abuse Scores
    peer_emot_hold = mace_age_fx(ever_hold, a['peer_emot_indx'], cutoffs[6], x['peer_emot_scoring'], min_acceptable[6])
    peer_emot_mace = peer_emot_hold.iloc[:,18]
    peer_emot_sum = peer_emot_hold.iloc[:,37]
    final_out = pd.concat([final_out, peer_emot_hold], axis=1)
    # Peer Physical Abuse Scores
    peer_phys_hold = mace_age_fx(ever_hold, a['peer_phys_indx'], cutoffs[7], x['peer_phys_scoring'], min_acceptable[7])
    peer_phys_mace = peer_phys_hold.iloc[:,18]
    peer_phys_sum = peer_phys_hold.iloc[:,37]
    final_out = pd.concat([final_out, peer_phys_hold], axis=1)
    # Peer Emotional Neglect Scores
    ever_hold2 = reverse_item_fx(ever_hold, a['em_negl_rev'])
    emot_negl_hold = mace_age_fx(ever_hold2, a['emot_negl_indx'], cutoffs[8], x['emot_negl_scoring'], min_acceptable[8])
    emot_negl_mace = emot_negl_hold.iloc[:,18]
    emot_negl_sum = emot_negl_hold.iloc[:,37]
    final_out = pd.concat([final_out, emot_negl_hold], axis=1)
    # Peer Physical Neglect Scores
    ever_hold2 = reverse_item_fx(ever_hold2, a['phys_negl_rev']) #ever_hold2<-ever_hold2
    phys_negl_hold = mace_age_fx(ever_hold2,a['phys_negl_rev'],cutoffs[9],x['phys_negl_scoring'],min_acceptable[9]) 
    phys_negl_mace = phys_negl_hold.iloc[:,18]
    phys_negl_sum = phys_negl_hold.iloc[:,37]
    final_out = pd.concat([final_out, phys_negl_hold], axis=1)
    # Intrafamilial MACE score
    intrafam_frame = pd.concat([sex_ab_fam, pva_mace, nvea_mace, phys_ab_mace, wipv_mace, viol_sib_mace, emot_negl_mace, phys_negl_mace], axis=1)
    intrafam_mace = intrafam_frame.sum(axis=1, skipna=True).to_numpy(dtype='float32')
    intrafam_count = intrafam_frame.count(axis=1).to_numpy()
    for item in range(0,len(intrafam_mace)):
        if intrafam_count[item] < min_acc_intrafam:
            intrafam_mace[item] = np.nan
    selected = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,38,39,40,
                41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,76,77,78,
                79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,114,115,
                116,117,118,119,120,121,122,123,124,125,126,127,128,129,
                130,131,132,152,153,154,155,156,157,158,159,160,161,162,
                163,164,165,166,167,168,169,170,190,191,192,193,194,195,
                196,197,198,199,200,201,202,203,204,205,206,207,208,228,
                229,230,231,232,233,234,235,236,237,238,239,240,241,242,
                243,244,245,246,247,267,268,269,270,271,272,273,274,275,
                276,277,278,279,280,281,282,283,284,304,305,306,307,308,
                309,310,311,312,313,314,315,316,317,318,319,320,321,322,
                342,343,344,345,346,347,348,349,350,351,352,353,355,356,
                357,358,359,360]       
    final_mace = final_out.iloc[:,selected]
    final_sum = final_out.iloc[:, [i for i in list(final_out.columns) if i not in selected]]
    out1 = final_sum
    q = [1,20,39,58,77,96,115,134,153,172]
    mace_multi_age = None
    mace_sum_age = None
    # provide MACE_SUM and MACE_MULTI for all years - provided valid categories >= min_acc_total

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

