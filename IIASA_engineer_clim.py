 
##########################################################################################################################################################################
# This python file is split into engineering of annual climate data and engineering of seasonal climate data
##########################################################################################################################################################################

import pandas as pd 
import numpy as np

#************* ANNUAL ENGINEERING ********************************
##########################################################################################################################################################################
##########################################################################################################################################################################

# WHEN PRESERVING YEAR COLUMN
##################################################################

# Import simulation unit data and all weather data 
simU_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_SimUData//SimUID_List.txt", sep = ";")
pet_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_PET.txt", sep = ",")
prcp_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_PRCP.txt", sep = ",")
rad_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_RAD.txt", sep = ",")
tmin_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_TMN.txt", sep = ",")
tmax_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_TMX.txt", sep = ",")
vpd_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_VPD.txt", sep = ",")

# change year to string values
for df in [pet_df, prcp_df, rad_df, tmin_df, tmax_df, vpd_df]: 
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
for df in [pet_df, prcp_df, rad_df, tmin_df, tmax_df, vpd_df, cmd_df]:
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

clim_merged.to_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_SimUData//SimUID_clim.csv")
simU_merged.to_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_SimUData//SimUID_static+clim.csv")


# WHEN WIDENING COLUMNS FOR SPECIFIC VARIABLE X MONTH X YEAR 
#################################################################

# Import simulation unit data and all weather data 
simU_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_SimUData//SimUID_List.txt", sep = ";")
pet_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_PET.txt", sep = ",")
prcp_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_PRCP.txt", sep = ",")
rad_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_RAD.txt", sep = ",")
tmin_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_TMN.txt", sep = ",")
tmax_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_TMX.txt", sep = ",")
vpd_df = pd.read_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_Weather//CORN_dyn_rf_BAU_R00_VPD.txt", sep = ",")

# change year to string values
for df in [pet_df, prcp_df, rad_df, tmin_df, tmax_df, vpd_df]: 
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
# reshape climate data before merging
# create list to store cleaned dataframes in addition to simUID dataframe
data_frames = [simU_df]

# for all climate dataframes
for df in [pet_df, prcp_df, rad_df, tmin_df, tmax_df, vpd_df, cmd_df]:
    # create mean column 
    df["MEAN"] = (df.AGG / 12)  
    # pivot on the year
    wide_df = df.pivot(index='SimUID', columns=['YR', 'VAR'], values=['JAN', 'FEB', 'MAR', 'APR',
       'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'AGG'])
    # fix column names
    wide_df.columns = wide_df.swaplevel(axis='columns').columns.to_flat_index().map('_'.join)
    # reset index
    wide_df.reset_index(inplace = True)
    # append to list
    data_frames.append(wide_df)
    print ("done.")
    
# merge climate data  
simUID_merged = reduce(lambda left,right: pd.merge(left,right,on=['SimUID'],
                                            how='left'), data_frames)

simUID_merged.to_csv("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//_SimUData//SimUID_static+clim_wide.csv")



#************* SEASONAL ENGINEERING ********************************
##########################################################################################################################################################################
##########################################################################################################################################################################






































