# get data 
def get_N_exp(crop, N, r, obs_df):
    import os 
    import pandas as pd 

    file_names = os.listdir("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//" + crop)
    # get annual files to pull yield 
    acy_files = [file for file in file_names if "ACY" in file]
    # get FNO3 files to pull AGG_FNO3
    N_files = [file for file in file_names if "FNO3" in file]
    
    # placeholder of dataframe so it updates iteratively
    merge_me = obs_df
    
    print ("starting ACY loop.")
    for acy in acy_files: 
        add_path = os.path.join(("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//" + crop), acy)
        # open the file 
        add = pd.read_csv(add_path)
        scen = add.SCEN.unique()[0][7:-3]
        amended = [(scen + col) for col in add.columns[4:]]
        new_cols = (list(add.columns[:4]) + amended)
        add.columns = new_cols
        print (acy, " open.")
        
        # then join this to the starting dataframe 
        merged = pd.merge(merge_me, add, how = "left", left_on = ["SimUID", "YR"], right_on = ["SimUID", "YR"])
        # drop index column 
#         merged.drop("index", axis = 1, inplace = True)
        print (merged.columns)
        # update dataframe to merge
        merge_me = merged
        
        print (acy, " merged.")
    
    print ("starting N loop")
    for N in N_files: 
        add_path = os.path.join(("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//" + crop), N)
        # open the file 
        add = pd.read_csv(add_path)
        scen = add.SCEN.unique()[0][7:-3]
        amended = [(scen + col) for col in add.columns[5:]]
        new_cols = (list(add.columns[:5]) + amended)
        add.columns = new_cols
        merge_cols = ["SimUID", "YR"] + [col for col in add.columns if "AGG" in col]
        ready = add[merge_cols]
        
        print (N, " open.")

        # then join this to the starting dataframe 
        merged = pd.merge(merge_me, ready, how = "left", left_on = ["SimUID", "YR"], right_on = ["SimUID", "YR"])
        # drop index column 
#         merged.drop("index", axis = 1, inplace = True)
        print (merged.columns)
        # update dataframe to merge
        merge_me = merged
        
        print (N, " merged.")

    # change index to simUID 
    return_df = merged.set_index('SimUID')
    
    # return the fully merged dataframe at the end
    return (return_df)

# DEFINE FUNCTION FOR PULLING ALL MONTHLY SIMULATION DATA TOGETHER FOR A GIVEN CROP X N X RESIDUE FOR EACH SIMU x YEAR
def all_run_data(arg, pull_vars): 
    crop = arg[0]
    nitr = arg[1]
    res = arg[2]
    # import statements for parallen processing
    import os 
    import pandas as pd
    
    # first pull all of the necessary data 
    # pull all variable names which will get linked to annual metrics 
    file_names = os.listdir("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//" + crop)
    # pull specific variable names - using N100 just to pull variable names 
    # variables = [f[21:-4] for f in [file for file in file_names if "N100" in file]]

    # OR CHOOSE VARIABLES YOURSELF
    # pull nitrogen and total organic carbon
    variables = pull_vars

    # pull files for specific nitrogen and residues 
    crop_nitr_res_files = [file for file in file_names if nitr in file and res in file]
    
    # starter dataframe = yearly measurement dataframe
    start_file = [file for file in crop_nitr_res_files if "ACY" in file][0]
    start_path = os.path.join(("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//" + crop), start_file)
    start = pd.read_csv(start_path)
    # placeholder of dataframe so it updates iteratively
    merge_me = start
    # take out the annual file as a variable now that we have that file used already 
    variables.remove("ACY")
    
    # make a large wide dataframe with all monthly variables and annual variables 
    # loop over all other variables and append to the annual output variables
    
    print ("starting variable loop.")
    for var in variables: 
        add_file = [file for file in crop_nitr_res_files if var in file][0]
        add_path = os.path.join(("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//" + crop), add_file)
        # open the file 
        add = pd.read_csv(add_path)
        print (var, " open.")
        
        # reformat so the variables are in the column names 
        wide_df = add.pivot(columns=['VAR'], values=['SimUID', 'YR', 'JAN', 'FEB', 'MAR', 'APR',
       'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'AGG'])
        # fix column names
        wide_df.columns = wide_df.swaplevel(axis='columns').columns.to_flat_index().map('_'.join)
        # reset index
        wide_df.reset_index(inplace = True)
        wide_df.rename(columns={(var + '_SimUID'):'SimUID', (var + '_YR'):'YR'}, inplace=True) 
        
        # then join this to the starting dataframe 
        merged = pd.merge(merge_me, wide_df, how = "left", left_on = ["SimUID", "YR"], right_on = ["SimUID", "YR"])
        # drop index column 
        merged.drop("index", axis = 1, inplace = True)
        # update dataframe to merge
        merge_me = merged
        
        print (var, " merged.")
        
    # change index to simUID 
    return_df = merged.set_index('SimUID')
    
    # return the fully merged dataframe at the end
    return (return_df)


def yield_run_data(arg): 
    crop = arg[0]
    nitr = arg[1]
    res = arg[2]
    # import statements for parallen processing
    import os 
    import pandas as pd
    
    # first pull all of the necessary data 
    # pull all variable names which will get linked to annual metrics 
    file_names = os.listdir("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//" + crop)
    # pull specific variable names - using N100 just to pull variable names 
    variables = [f[21:-4] for f in [file for file in file_names if "N100" in file]]
    # pull files for specific nitrogen and residues 
    crop_nitr_res_files = [file for file in file_names if nitr in file and res in file]
    
    # starter dataframe = yearly measurement dataframe
    start_file = [file for file in crop_nitr_res_files if "ACY" in file][0]
    start_path = os.path.join(("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//" + crop), start_file)
    start = pd.read_csv(start_path)
    
    # return the fully merged dataframe at the end
    return (start)


# get season length, start and end dates for a given treatment 
# pass in a dataframe of biomass data per simulation unit x year 
def get_season_info(arg):
    crop = arg[0]
    n = arg[1]
    res = arg[2]
    import os 
    import pandas as pd 
    import numpy as np 

    # pull all variable for a given crop
    file_names = os.listdir("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//" + crop)
    # pull files for specific nitrogen and residues 
    biom = [file for file in file_names if n in file and res in file and "BIOM" in file]
    
    # open biomass file
    biomass_path = os.path.join(("//Users//taraippolito//Desktop//Desktop_Tara’s_MacBook_Pro//EPIC_local//" + crop), biom[0])
    # open the file 
    in_file = pd.read_csv(biomass_path)
    
    # biomass columns only 
    biom_cols = [col for col in in_file.columns if col not in ["SimUID", "CROP", "SCEN", "YR", "VAR", "AGG"]]
    
    # calculate season length 
    in_file['season_length'] = in_file[biom_cols].ne(0).sum(axis=1)
    in_file.drop(in_file[in_file.season_length == 0].index, inplace = True)
    
    # calculate start index and end index 
    arr_biom = in_file[biom_cols].to_numpy()
    biom_st_ind = [np.min(np.nonzero(arr_biom[i])) for i in range(len(arr_biom))]
    biom_en_ind = [np.max(np.nonzero(arr_biom[i])) for i in range(len(arr_biom))]
    
    # add additional columns 
    in_file['start_index'] = biom_st_ind
    in_file['end_index'] = biom_en_ind
    
    # output 
    out = in_file[["SimUID", "YR", "season_length", "start_index", "end_index"]]
    
    return(out)

# pass in: 
# row from seasonal stats dataframe (calculated from biomass data)
# climate variable dataframe with year column in tact
# the metric desired: "skew", "mean", "sum" 
# the duration: "gs" for growing season or "gsy" for growing season year 

def get_gs_climate(arg):
    df = arg[0]
    clim_var_df = arg[1]

    import pandas as pd
    # pull the year specific chunk of the climate data
    yr_clim = clim_var_df[(clim_var_df.YR == df.YR.unique()[0])]
    c_var = yr_clim.VAR.unique()[0]
    # new columns 
    sum_col = c_var + "sumGS"
    mean_col = c_var + "avGS"
    skew_col = c_var + "skGS"
    
    # month columns to reference with indices
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    
    # get the months to calculate metric over  
    dur_months = months[df.start_index.unique()[0]:(df.end_index.unique()[0]+1)]
    
    GS_df = yr_clim[['SimUID']]
    # loop over metrics
    GS_df[sum_col] = yr_clim[dur_months].sum(axis = 1)
    GS_df[mean_col] = yr_clim[dur_months].mean(axis = 1)
    GS_df[skew_col] = yr_clim[dur_months].skew(axis = 1)
    
    out_df = pd.merge(df, GS_df, how= "left", on = "SimUID")

    return (out_df)
    # 
def row_gsy_climate(row, clim_var_df, metric):
    import pandas as pd
    import scipy 
    # pull the specific rows needed from climate variable dataframe
    row_clim = clim_var_df[(clim_var_df.SimUID == row.SimUID) & (clim_var_df.YR == row.YR)]
    row_clim_2 = clim_var_df[(clim_var_df.SimUID == row.SimUID) & (clim_var_df.YR == (row.YR + 1))]
    
    # month columns to reference with indices
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    
    # get the months to calculate metric over 
    dur_months = months[row.start_index:]
    dur_months_22 = months[:row.start_index]    
    
    # if the metric is "sum" 
    if metric == "sum":
        output = np.concatenate((row_clim[dur_months].values.flatten(), row_clim_2[dur_months_2].values.flatten())).sum()
    elif metric == "mean": 
        output = np.concatenate((row_clim[dur_months].values.flatten(), row_clim_2[dur_months_2].values.flatten())).mean()
    else: 
        # to calculate skew between two dataframes create array first
        arr = np.concatenate((row_clim[dur_months].values.flatten(), row_clim_2[dur_months_2].values.flatten()))
        output = scipy.stats.skew(arr)
        
    return (output)


def split_data(full_df):
    import pandas as pd
    split_dfs = []
    for val in full_df.start_index.unique(): 
        for length in full_df.season_length.unique():
            for yr in full_df.YR.unique(): 
                add_me = full_df[(full_df.start_index == val) & (full_df.season_length == length) & (full_df.YR == yr)]
                if len(add_me)>0:
                    split_dfs.append(add_me)
    
    return (split_dfs)

def tt_split_scale(df, target):
    # import statements for running in parallel 
    import sklearn.model_selection
    import sklearn.preprocessing
    import pandas as pd 
    
    # drop na 
    df.dropna(inplace= True)
    
    # columns to use in analysis - monthly data
    X_cols = [col for col in df.columns if col not in ['CROP', 'SCEN', 'YLDG', 'YLDF', 'YLC', 'BIOM', 'RW', 'mean_OCPD_change', 'YLDG_std', 'mean_OCPD_change_std']]
    
    # set target variable as YLDG 
    y = df[target].astype('float64')
    # set x columns as all other columns
    X_df = df[X_cols].astype('float64')
    
    # train test split - 75/25 split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_df, y, test_size=0.25, random_state=42)
    # print ("finished TT split.")

    train_ind = X_train.index.to_list()
    
    # instantiate scaler
    scaler_X = sklearn.preprocessing.StandardScaler()
    scaler_y = sklearn.preprocessing.StandardScaler()

    # fit scaler to training sets
    scaler_X.fit(X_train)
    scaler_y.fit(y_train.to_numpy().reshape(-1, 1))
    # print ("finished fitting scaler.")
    
    # transform the train and test sets accordingly
    scaled_X_train = scaler_X.transform(X_train)
    scaled_X_test = scaler_X.transform(X_test)
    # print ("finished X transformation.")
    
    scaled_y_train = scaler_y.transform(y_train.to_numpy().reshape(-1, 1))
    scaled_y_test = scaler_y.transform(y_test.to_numpy().reshape(-1, 1))   
    # print ("finished y transformation.")
    
    # return the split & scaled data
    return (scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test, train_ind)


def random_forest(X_train, X_test, y_train, y_test, n_est, depth): 
    import sklearn.ensemble 
    
    # instantiate the forest 
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=n_est, max_depth=depth, max_features=.3, bootstrap = True, max_samples = .8, n_jobs=-2)
    # fit the random forest 
    rf.fit(X_train, y_train)
    # predict 
    y_predicted = rf.predict(X_test)
    # get metrics 
    score = rf.score(X_test, y_test)
    feat_imp = rf.feature_importances_
    
    return (y_predicted, score, feat_imp)


    