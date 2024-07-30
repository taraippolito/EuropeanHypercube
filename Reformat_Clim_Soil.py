 
##########################################################################################################################################################################
# widens climate data so each row of the data is unique to a simulation unit x year with climate variables in format variable x month, year is kept as separate column
##########################################################################################################################################################################

import pandas as pd 
import numpy as np
import os 

#************* ANNUAL ENGINEERING ********************************
##########################################################################################################################################################################
##########################################################################################################################################################################

# WHEN PRESERVING YEAR COLUMN
##################################################################

# Import simulation unit data and all weather data 
path = "//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local"
simU_df = pd.read_csv(os.path.join(path, "_SimUData//SimUID_List.txt"), sep = ";")
pet_df = pd.read_csv(os.path.join(path, "_Weather//CORN_dyn_rf_BAU_R00_PET.txt"), sep = ",")
prcp_df = pd.read_csv(os.path.join(path, "_Weather//CORN_dyn_rf_BAU_R00_PRCP.txt"), sep = ",")
rad_df = pd.read_csv(os.path.join(path, "_Weather//CORN_dyn_rf_BAU_R00_RAD.txt"), sep = ",")
tmean_df = pd.read_csv(os.path.join(path, "_Weather//CORN_dyn_rf_BAU_R00_TMEAN.txt"), sep = ",")
vpd_df = pd.read_csv(os.path.join(path, "_Weather//CORN_dyn_rf_BAU_R00_VPD.txt"), sep = ",")

# change year to string values
for df in [pet_df, prcp_df, rad_df, tmean_df, vpd_df]: 
    df["YR"] = df["YR"].astype(str)
    print ("done.")

# get climate moisture deficit dataframe
cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# start with just simuid and year columns, then build
cmd_df = pet_df[["SimUID", "YR"]]

for i in range(len(cols)): 
    cmd_df[cols[i]] = prcp_df[cols[i]] - pet_df[cols[i]]

cmd_df["VAR"] = "CMD"
cmd_df["AGG"] = cmd_df[cols].sum(axis = 1)

# create list to store cleaned dataframes in addition to simUID dataframe
data_frames = []
# change columns to be variable specific 
# for all climate dataframes
for df in [pet_df, prcp_df, rad_df, tmean_df, vpd_df, cmd_df]:
    # create mean column 
    df["MEAN"] = (df.AGG / 12)    
    # pull column names 
    cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'AGG', "MEAN"]
    # pull variable name 
    var = df.VAR.unique()[0]
    # new column names 
    var_cols = [(col + "_" + var) for col in cols]
    # drop the variable column 
    try:
        df.drop(["VAR", "CROP", "SCEN"], axis = 1, inplace = True)
    except: 
        df.drop(["VAR"], axis = 1, inplace = True)
    df.columns = (['SimUID', 'YR'] + var_cols)
    # append to list
    data_frames.append(df)
    print ("done.")

data_frames.append(simU_df)

# merge climate data  
clim_merged = reduce(lambda left,right: pd.merge(left,right,on=['SimUID', 'YR'],
                                            how='left'), data_frames[:-1])
# merge simU data to climate data 
simU_merged = pd.merge(clim_merged, simU_df, on = "SimUID", how = "left")


# calculate soil attributes for the full soil profile depth 
simU_merged['full_depth'] = simU_merged.TOPL + simU_merged.SUBL 

# average these variables using weighted average
get_avgs = ["SAND", "SILT", "CLAY", "BD", "BS", "CEC", "SOB", "PH", "VS", "KS"]
# sum these variables
get_sums = ["FWC", "WP"]

# calculate profile averages of variables 
for var in get_avgs: 
    av_me = [col for col in simU_merged.columns if var in col]
    simU_merged[(var + "_PROFILE")] = simU_merged[av_me[0]] * (simU_merged.TOPL / simU_merged.full_depth) +  simU_merged[av_me[-1]] * (simU_merged.SUBL / simU_merged.full_depth)
 # calculate profile sum of variables   
for var in get_sums: 
    av_me = [col for col in simU_merged.columns if var in col]
    simU_merged[(var + "_PROFILE")] = simU_merged[av_me[0]] +  simU_merged[av_me[-1]]


# save files
clim_merged.to_csv(os.path.join(path, "_SimUData//SimUID_clim.csv"))
simU_merged.to_csv(os.path.join(path, "_SimUData//SimUID_static+clim.csv"))







































