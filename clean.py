#itay zaada #
#this function is auto clearing data, handling the nulls, ploting all of the numeric data compared to the last column, make OneHotEncoder to the categorical features 
#and remove highly correlated columns, and standardize the numric data #
def DataCleaner(name):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    from pandas_profiling import profile_report
    from sklearn.preprocessing import MinMaxScaler
    ## function that merge the dataframes and remove duplicates ## 
    pd.set_option("display.max_columns", None)
    ## function that merge the dataframes ##  
    
    def merge_fix_cols(df_company,df_product,uniqueID):## merge 2 dataframes and remove the duplicate ## 
        df_merged = pd.merge(df_company,df_product,how='left',on=uniqueID)    
        for col in df_merged:
            if col.endswith('_x'):
                df_merged.rename(columns = lambda col:col.rstrip('_x'),inplace=True)
            elif col.endswith('_y'):
                to_drop = [col for col in df_merged if col.endswith('_y')]
                df_merged.drop(to_drop,axis=1,inplace=True)
            else:
                pass
        return df_merged
    
    dataset = pd.read_csv(name)##read the data ##
    dataset=dataset.replace(r'^\s*$', np.nan, regex=True)
    fname=name.split(".")[0]## get only the file name for later ## 
    dataset["unique_id_zaada"] = dataset.index + 1## insert insex for rows so later can be merged ##
    first_column = dataset.pop('unique_id_zaada')## add my uniqe id ## 
    dataset.insert(0, 'unique_id_zaada', first_column)## add my uniqe id ##
    df=dataset


    ##stage 0 # 
        # Data exploration ##
    dfi=df
  ## clear all nulls in numric so plot can be made ## 
    for i in  dfi.columns:
        if dfi.dtypes[i] !='object'and i!='unique_id_zaada':
            dfi[i]= dfi[i].fillna(value=dfi[i].mean()) ## fil nulls with the mean ##                 
    
     #making a hist chart of all numric featurs,and all numric featuers vs the last feters -the "y"- the col we want to predict # 
    for i in (dfi.columns):
            plt.figure()
            if dfi.dtypes[i] !='object'  :
                if dfi.dtypes[-1] !='object':
                    df.plot.scatter(x=i,y=df.columns[-1],)
                df.hist(i)
                plt.title(i)
                plt.show()


    #show profile of the data##
    profile=dfi.profile_report()
    profile.to_file(output_file ="output.html")
    
    
    ##stage 1 #
       # Normalize (to split the data frame cells that have" , ; / \" to diffrent colums )##
    dataset=dataset.replace(';',',')
    dataset=dataset.replace('/' ,',')
    for i in df.columns:
         if dataset.dtypes[i] =='object':
            dataset[i]=dataset[i].str.replace(';',',')
            dataset[i]=dataset[i].str.replace('/' ,',')
    
    
    

    for i in dataset.columns:
        counters=0
        if dataset.dtypes[i] =='object':
            splited=dataset[i].str.split(",", expand=True)
            for j in splited:
                if (len(splited.columns))>1:
                    df[i+str(counters)]=splited[j]
                    counters=counters+1
                    if counters ==1 :
                        df=df.drop(i, axis=1)

# and make sure the are no typos of strings in nums ##
    strings=0
    nums=0
    for i in range(0,len(df.columns)):
        strings=0##strings counter ## 
        nums=0## nums counter ## 
        if df.dtypes[i] =='object': ## if the data types show as object -String ##
            for j in range(0,len(df.index)): ## run on all of the coulumn rows ## 
                type(df.loc[j,df.columns[i]])
                try:
                    df.loc[j,df.columns[i]]=float(df.loc[j,df.columns[i]])## try to make the rows as int if yes one more to the int team##
                    nums=nums+1  
                except:## if it failed one more to the string team ## 
                    df.loc[j,df.columns[i]]=str(df.loc[j,df.columns[i]])
                    strings=strings+1
                    
        indicator=(nums/len(df.index))## the % on nums in the all datafraeme ## 
        if indicator > 0.6 :## if the data frame have more then 60% ##
            df[df.columns[i]] = df[df.columns[i]].apply(pd.to_numeric, errors='coerce')## make all nummric ,and strings are nulls##
            df[df.columns[i]]= df[df.columns[i]].fillna(value=df[df.columns[i]].mean()) ## fil nulls with the mean ##
   
    # stage 2#
        ## Feature Preprocessing for Categorical Features and stadartaize data##
    
    counter=0 ## to make sure just the first one will make the first dataframe ## 
    df1=df
    for i in range (0,(len(df.columns))): 
            #clear nulls,make all lower case letters# 
            #make OneHotEncoder on the strings only #
    
        if df.dtypes[i] =='object':## make sure that the type is not numaric ## 
    
            ## first dataframe that the anothers df can  merged into ## 
    
            if df.dtypes[i] =='object' and counter==0 :## if the type of the data is string make One hoe encoder#
                counter=counter+1
                df[df.columns[i]]=df[df.columns[i]].fillna("empty_zaada") 
                df[df.columns[i]]=df[df.columns[i]].dropna()
                df[df.columns[i]]=df[df.columns[i]].str.strip().str.lower() ## all str is lower case and stripd  ## 
                enc =OneHotEncoder (sparse=False) ## make onehotencoder## 
                onehot= enc.fit_transform(df[df.columns[i]].values.reshape(-1,1))##reshape beacuse of eror#
                df1=pd.DataFrame(onehot, columns=enc.get_feature_names())## data frame of the onehotencoder##
                df1["unique_id_zaada"] = df.index + 1## make unique id so can merge the dataframes ##
                first_column = df1.pop("unique_id_zaada")#make uniqe id in the dataframe ## 
                df1.insert(0, 'unique_id_zaada', first_column) ## make the uniqe id first col ## 
    
                ## after the first dataframe making another dataframe that merge to the first each iteration  ## 
                #make OneHotEncoder on the strings only #
    
            elif df.dtypes[i] =='object' and counter!=0:## if the type of the data is string make One hoe encoder and not first iteration#
                #clear nulls,make all lower case letters ,make encoder #
                df[df.columns[i]]=df[df.columns[i]].fillna("empty_zaada")
                df[df.columns[i]]=df[df.columns[i]].dropna()
                df[df.columns[i]]=df[df.columns[i]].str.strip().str.lower() ## all str is lower case ## 
                enc =OneHotEncoder (sparse=False) ## make onehotencoder## 
                onehot= enc.fit_transform(df[df.columns[i]].values.reshape(-1,1))##reshape beacuse of eror#
                df2=pd.DataFrame(onehot, columns=enc.get_feature_names())## data frame of the onehotencoder##
                df2["unique_id_zaada"] = df.index + 1## make unique id so can merge the dataframes ##
                first_column = df2.pop("unique_id_zaada")#make uniqe id in the dataframe ## 
                df2.insert(0, 'unique_id_zaada', first_column)
                df1=merge_fix_cols(df1,df2,'unique_id_zaada')# merge the df of the encdoding to the first dataframe ## 
                  ## df1 is all the One hot encoding featuers df combined ## 
    


     # stage 3#             
         #Noramlize & Validation and headling eror  ## 

    df3=df

    for i in  df3.columns:
        if df3.dtypes[i] =='object' :
            df3=df3.drop(i, axis=1)
        elif df3.dtypes[i] !='object'and i!='unique_id_zaada':
            df3[i]=df3[i].fillna(value=df3[i].mean()) ## fil nulls with the mean ## 
            norm = MinMaxScaler().fit(df3[i].values.reshape(-1,1))
            df3[i] = norm.transform(df3[i].values.reshape(-1,1))
            df3[i].fillna(value=df3[i].values.mean()) ## fil nulls with the mean ##
        
        
    df=merge_fix_cols(df1,df3,'unique_id_zaada')# merge the df of the encdoding and the data frame of numric  ## 

    df.columns = [col.replace('x0_', '') for col in df.columns]
    
    try:## if there are any empty zaada nulls ## 
        df=df.drop("empty_zaada", axis=1) ## remove the empty_zaada col ##
    except:
        print(" ") 
    
    
    
    # stage 4#  
            ##Remove columns that are highly correlated ##
  
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df=df.drop(df[to_drop], axis=1)
    df.to_csv(fname+'_cleaned'+'.csv', index = False, header=True)


DataCleaner(file)## clear the data ##
