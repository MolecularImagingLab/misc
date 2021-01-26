#!/usr/bin/env python

""" Import libraries. """
import pandas as pd
import numpy as np
import itertools as itls

""" Reverse item scoring. """
def reverse_item_fx(dframe, indx):
    for i in range(-18,1):
        inx = indx+i
        #print('~~~rev inx:',inx)
        dframe.iloc[:,inx] = abs(1-dframe.iloc[:,inx])
    return dframe

""" Checks if number of missing values are acceptable based on cutoffs defined in mace_calculate(). """
def mace_cutoff_fx(mace, cutoff):
    for item in range(0,len(mace)):
        if mace[item] < cutoff:
            mace[item] = 0
        else:
            mace[item] = 1
    return mace

""" Mace item scoring. """
def mace_age_fx(dframe, indx, cutoff, scoring, min_acceptable):
    out_mat = np.zeros([dframe.shape[0],19*2])
    df = pd.DataFrame(data=out_mat)
    #print('~~~~~scoring~~~~~',scoring)
    for k,i in enumerate(range(-18,1), start=0): #for count, item
        j = k
        #print('j:',j)
        k = j +19
        #print('k:',k)
        inx = indx+i
        #print('inx:',inx)
        hold = dframe.iloc[:,inx]
        lnt = (hold.count(axis=1)).to_numpy()
        #print('lnt:',lnt)
        cnt = (hold.sum(axis=1, skipna=True)).to_numpy(dtype='float32')
        #print('cnt before:',cnt)
        for item in range(0,len(dframe.index)):
            if lnt[item] < min_acceptable:
                cnt[item] = np.nan
        final_cnt = cnt.copy()
        mace = mace_cutoff_fx(cnt, cutoff)
        qz = np.array2string(mace)
        #print('mace:',qz)
        #print('cnt after2:',cnt)
        mace_sum = []
        for c in final_cnt:
            c = int(c)
            mace_sum.append(scoring[c]) #c+1 in r-code
            #print('C:',c,'mace_sum:',mace_sum)
        mace_sum = np.array(mace_sum)
        #print('sum:',mace_sum)
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
    pva_scoring = np.array([0, 2.51, 5, 7.5, 10])
    nvea_scoring = np.array([0, 1.57203586201709, 3.07318170239233, 4.30017268552622, 5.66888216594181, 7.74694728216119, 10])
    phys_ab_scoring = np.array([0, 1.97350025645546, 3.80610533906172, 5.07761998873815, 6.34283867353684, 8.10910025721665, 10])
    wipv_scoring = np.array([0, 2.16265962002852, 4.24162614623762, 6.04358509621965, 7.99291014858454, 10])
    viol_sib_scoring = np.array([0, 2.51, 5, 7.5, 10])
    peer_emot_scoring = np.array([0, 2.01491889140193, 3.94389919669921, 5.65181683198404, 7.76456257878893, 10])
    peer_phys_scoring = np.array([0, 2.31610945680412, 4.49115107318588, 6.18181081484338, 8.05096066393912, 10])
    emot_negl_scoring = np.array([0, 2.06604852199299, 4.06369282630551, 5.86280636891976, 7.89386012994807, 10])
    phys_negl_scoring = np.array([0, 2.16341492872456, 4.2156227144729, 5.92172308914546, 7.91150096415818, 10])
    scoring_vals = {'sex_ab_scoring':sex_ab_scoring, 'pva_scoring':pva_scoring, 'nvea_scoring':nvea_scoring, 'phys_ab_scoring':phys_ab_scoring, 
                    'wipv_scoring':wipv_scoring, 'viol_sib_scoring':viol_sib_scoring, 'peer_emot_scoring':peer_emot_scoring, 
                    'peer_phys_scoring':peer_phys_scoring, 'emot_negl_scoring':emot_negl_scoring, 'phys_negl_scoring':phys_negl_scoring}
    for score in scoring_vals.keys():
        scoring_vals[score] = np.round(scoring_vals[score], decimals=0)
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
    phys_negl_hold = mace_age_fx(ever_hold2,a['phys_negl_indx'],cutoffs[9],x['phys_negl_scoring'],min_acceptable[9]) 
    phys_negl_mace = phys_negl_hold.iloc[:,18]
    phys_negl_sum = phys_negl_hold.iloc[:,37]
    final_out = pd.concat([final_out, phys_negl_hold], axis=1)
    # Intrafamilial MACE score
    intrafam_frame = pd.concat([sex_ab_fam, pva_mace, nvea_mace, phys_ab_mace, wipv_mace, viol_sib_mace, emot_negl_mace, phys_negl_mace], axis=1)
    intrafam_mace = intrafam_frame.sum(axis=1, skipna=True)
    intrafam_count = intrafam_frame.count(axis=1).to_numpy()
    for item in range(0,len(intrafam_mace)):
        if intrafam_count[item] < min_acc_intrafam:
            intrafam_mace[item] = np.nan
    selected = list(itls.chain(range(0,19), range(38,57), range(76,95), range(114,133),
                              range(152,171), range(190,209), range(228,247), range(266,285),
                              range(304,323), range(342,361)))
    final_mace = final_out.iloc[:,selected]
    final_sum = final_out.drop(final_out.iloc[:,selected], axis=1)
    out1 = final_sum
    q = np.array([0,19,38,57,76,95,114,133,152,171])
    mace_multi_age = None
    mace_sum_age = None
    # provide MACE_SUM and MACE_MULTI for all years - provided valid categories >= min_acc_total
    for i in range(0,18):
        k = q+i
        #print(k)
        hold = final_mace.iloc[:,k]
        lnt = hold.count(axis=1).to_numpy()
        mace_age = hold.sum(axis=1, skipna=True)
        for item in range(0,len(lnt)):
            if lnt[item] < min_acc_total:
                mace_age[item] = np.nan
        mace_multi_age = pd.concat([mace_multi_age, mace_age], axis=1)
        hold = final_sum.iloc[:,k]
        mace_age = hold.sum(axis=1, skipna=True)
        for item in range(0,len(lnt)):
            if lnt[item] < min_acc_total:
                mace_age[item] = np.nan
        mace_sum_age = pd.concat([mace_sum_age, mace_age], axis=1)
    k = q+18   
    #print(k)
    hold = final_mace.iloc[:,k]
    mace_multi_ever = hold.sum(axis=1, skipna=True)
    lnt = hold.count(axis=1)
    for item in range(0,len(lnt)):
        if lnt[item] < min_acc_total:
            mace_multi_ever[item] = np.nan
    hold = final_sum.iloc[:,k]
    mace_sum_ever = hold.sum(axis=1, skipna=True)
    for item in range(0,len(lnt)):
        if lnt[item] < min_acc_total:
            mace_sum_ever[item] = np.nan
    type_ever = pd.concat([sex_ab_sum, pva_sum, nvea_sum, phys_ab_sum, wipv_sum, viol_sib_sum, 
                          peer_emot_sum, peer_phys_sum, emot_negl_sum, phys_negl_sum], axis=1)
    final_frame = pd.concat([ever_hold.iloc[:,0], out1, mace_multi_age, mace_sum_age, mace_multi_ever,
                            mace_sum_ever, type_ever, intrafam_mace], axis=1)
    name_dict = ['Subject', 'MACE_SexAb_1', 'MACE_SexAb_2', 'MACE_SexAb_3', 'MACE_SexAb_4', 'MACE_SexAb_5', 'MACE_SexAb_6',
                 'MACE_SexAb_7', 'MACE_SexAb_8', 'MACE_SexAb_9', 'MACE_SexAb_10', 'MACE_SexAb_11', 'MACE_SexAb_12', 'MACE_SexAb_13', 
                 'MACE_SexAb_14', 'MACE_SexAb_15', 'MACE_SexAb_16', 'MACE_SexAb_17', 'MACE_SexAb_18', 'MACE_SexAb_EVER', 'MACE_PVA_1', 
                 'MACE_PVA_2', 'MACE_PVA_3', 'MACE_PVA_4', 'MACE_PVA_5', 'MACE_PVA_6', 'MACE_PVA_7', 'MACE_PVA_8', 'MACE_PVA_9', 
                 'MACE_PVA_10', 'MACE_PVA_11', 'MACE_PVA_12', 'MACE_PVA_13', 'MACE_PVA_14', 'MACE_PVA_15', 'MACE_PVA_16', 'MACE_PVA_17', 
                 'MACE_PVA_18', 'MACE_PVA_EVER', 'MACE__NVEA_1', 'MACE_NVEA_2', 'MACE_NVEA_3', 'MACE_NVEA_4', 'MACE_NVEA_5', 'MACE_NVEA_6', 
                 'MACE_NVEA_7', 'MACE_NVEA_8', 'MACE_NVEA_9', 'MACE_NVEA_10', 'MACE_NVEA_11', 'MACE_NVEA_12', 'MACE_NVEA_13', 'MACE_NVEA_14', 
                 'MACE_NVEA_15', 'MACE_NVEA_16', 'MACE_NVEA_17', 'MACE_NVEA_18', 'MACE_NVEA_EVER', 'MACE_PPhysM_1', 'MACE_PPhysM_2', 'MACE_PPhys_3', 
                 'MACE_PPhys_4', 'MACE_PPhys_5', 'MACE_PPhys_6', 'MACE_PPhys_7', 'MACE_PPhys_8', 'MACE_PPhys_9', 'MACE_PPhys_10', 'MACE_PPhys_11', 
                 'MACE_PPhys_12', 'MACE_PPhys_13', 'MACE_PPhys_14', 'MACE_PPhys_15', 'MACE_PPhys_16', 'MACE_PPhys_17', 'MACE_PPhys_18', 'MACE_PPhys_EVER',
                 'MACE_IPV_1', 'MACE_IPV_2', 'MACE_IPV_3', 'MACE_IPV_4', 'MACE_IPV_5', 'MACE_IPV_6', 'MACE_IPV_7', 'MACE_IPV_8', 'MACE_IPV_9', 
                 'MACE_IPV_10', 'MACE_IPV_11', 'MACE_IPV_12', 'MACE_IPV_13', 'MACE_IPV_14', 'MACE_IPV_15', 'MACE_IPV_16', 'MACE_IPV_17', 'MACE_IPV_18', 
                 'MACE_IPV_EVER', 'MACE_WSibA_1', 'MACE_WSibA_2', 'MACE_WSibA_3', 'MACE_WSibA_4', 'MACE_WSibA_5', 'MACE_WSibA_6', 'MACE_WSibA_7', 'MACE_WSibA_8',
                 'MACE_WSibA_9', 'MACE_WSibA_10', 'MACE_WSibA_11', 'MACE_WSibA_12', 'MACE_WSibA_13', 'MACE_WSibA_14', 'MACE_WSibA_15', 'MACE_WSibA_16', 'MACE_WSibA_17', 
                 'MACE_WSibA_18', 'MACE_WSibA_EVER', 'MACE_PeerVA_1', 'MACE_PeerVA_2', 'MACE_PeerVA_3', 'MACE_PeerVA_4', 'MACE_PeerVA_5', 'MACE_PeerVA_6', 'MACE_PeerVA_7', 
                 'MACE_PeerVA_8', 'MACE_PeerVA_9', 'MACE_PeerVA_10', 'MACE_PeerVA_11', 'MACE_PeerVA_12', 'MACE_PeerVA_13', 'MACE_PeerVA_14', 'MACE_PeerVA_15', 'MACE_PeerVA_16', 
                 'MACE_PeerVA_17', 'MACE_PeerVA_18', 'MACE_PeerVA_EVER', 'MACE_PeerPhys_1', 'MACE_PeerPhys_2', 'MACE_PeerPhys_3', 'MACE_PeerPhys_4', 'MACE_PeerPhys_5', 'MACE_PeerPhys_6', 
                 'MACE_PeerPhys_7', 'MACE_PeerPhys_8', 'MACE_PeerPhys_9', 'MACE_PeerPhys_10', 'MACE_PeerPhys_11', 'MACE_PeerPhys_12', 'MACE_PeerPhys_13', 'MACE_PeerPhys_14', 'MACE_PeerPhys_15', 
                 'MACE_PeerPhys_16', 'MACE_PeerPhys_17', 'MACE_PeerPhys_18', 'MACE_PeerPhys_EVER', 'MACE_EN_1', 'MACE_EN_2', 'MACE_EN_3', 'MACE_EN_4', 'MACE_EN_5', 'MACE_EN_6', 'MACE_EN_7', 
                 'MACE_EN_8', 'MACE_EN_9', 'MACE_EN_10', 'MACE_EN_11', 'MACE_EN_12', 'MACE_EN_13', 'MACE_EN_14', 'MACE_EN_15', 'MACE_EN_16', 'MACE_EN_17', 'MACE_EN_18', 'MACE_EN_EVER', 
                 'MACE_PN_1', 'MACE_PN_2', 'MACE_PN_3', 'MACE_PN_4', 'MACE_PN_5', 'MACE_PN_6', 'MACE_PN_7', 'MACE_PN_8', 'MACE_PN_9', 'MACE_PN_10', 'MACE_PN_11', 'MACE_PN_12', 'MACE_PN_13', 
                 'MACE_PN_14', 'MACE_PN_15', 'MACE_PN_16', 'MACE_PN_17', 'MACE_PN_18', 'MACE_PN_EVER', 'MACE_MULT_1', 'MACE_MULT_2', 'MACE_MULT_3', 'MACE_MULT_4', 'MACE_MULT_5', 'MACE_MULT_6', 'MACE_MULT_7', 
                 'MACE_MULT_8', 'MACE_MULT_9', 'MACE_MULT_10', 'MACE_MULT_11', 'MACE_MULT_12', 'MACE_MULT_13', 'MACE_MULT_14', 'MACE_MULT_15', 'MACE_MULT_16', 'MACE_MULT_17', 'MACE_MULT_18', 'MACE_SUM_1', 'MACE_SUM_2', 
                 'MACE_SUM_3', 'MACE_SUM_4', 'MACE_SUM_5', 'MACE_SUM_6', 'MACE_SUM_7', 'MACE_SUM_8', 'MACE_SUM_9', 'MACE_SUM_10', 'MACE_SUM_11', 'MACE_SUM_12', 'MACE_SUM_13', 'MACE_SUM_14', 'MACE_SUM_15', 'MACE_SUM_16', 
                 'MACE_SUM_17', 'MACE_SUM_18', 'MACE_MULT_EVER', 'MACE_SUM_EVER', 'MACE_SexAb_EVER', 'MACE_PVA_EVER', 'MACE_NVEA_EVER', 'MACE_PPhysM_EVER', 'MACE_IPV_EVER', 'MACE_WSibA_EVER', 'MACE_PeerVA_EVER', 'MACE_PeerPhysEVER', 
                 'MACE_EN_EVER', 'MACE_PN_EVER', 'Intrafamalial_MACE']
    final_frame.columns = name_dict
    return final_frame

""" Main function to process 52 question MACE, total of 988 items. """
def mace_execute(dataframe, start_col):
    dataframe=dataframe
    start_col=start_col
    
    num_cols=len(dataframe.columns) - start_col
    if num_cols != 988:
        print("Dataframe has",num_cols,"columns for scoring. Only 988 columns valid for scoring.")
    else:
        longform = mace_calculate(dataframe, start_col)
        summary= longform.loc[:,['Subject','MACE_MULT_EVER','MACE_SUM_EVER', 'MACE_SexAb_EVER', 'MACE_PVA_EVER', 'MACE_NVEA_EVER', 'MACE_PPhysM_EVER', 
                         'MACE_IPV_EVER', 'MACE_WSibA_EVER', 'MACE_PeerVA_EVER', 'MACE_PeerPhysEVER', 'MACE_EN_EVER', 'MACE_PN_EVER', 'Intrafamalial_MACE']]
        return longform,summary
        print("Dataframe has 988 columns for scoring. Calculating...")

""" This is an example of how to load the data and calculate the 52-item MACE scores. 
Sample MACE scores can be obtained from: https://europepmc.org/articles/PMC4340880/bin/pone.0117423.s016.gz
Many thanks to the authors for providing open source code to the science community: 
Teicher, Martin H., and Angelika Parigger. "The ‘Maltreatment and Abuse Chronology of Exposure’(MACE) scale 
for the retrospective assessment of abuse and neglect during development." PLoS one 10.2 (2015): e0117423."""
dataframe = pd.read_csv('/path/to/file/mace_sample.csv')
longform,summary = mace_execute(dataframe, start_col=2)

