
def my_overlappingcount(pattern, string):
    import re
    
    left = 0
    count = 0
    while True:
        match = re.search(pattern, string[left:])
        if not match:
            break
        count += 1
        left += match.start() + 1
    return count



def df_unique_rows(df_input):
    
    import pandas as pd
    
    kmer_pool = df_input.index
    
    # first step: remove the dots from the front and back until an alphabetic character is encountered    
    def remove_dots(string):
        start_index = 0
        end_index = len(string) - 1

        while start_index < len(string) and not string[start_index].isalpha():
            start_index += 1

        while end_index >= 0 and not string[end_index].isalpha():
            end_index -= 1

        return string[start_index:end_index+1]
    
    sizes = []
    for string in kmer_pool:
        original_size = len(string)
        modified_string = remove_dots(string)
        modified_size = len(modified_string)
        #sizes.append((original_size, modified_string, modified_size))
        sizes.append((string, original_size, modified_string, modified_size))

    df_removedots = pd.DataFrame(sizes, columns=['miRNA','Original Size','Modified String', 'Modified Size'], index = [kmer_pool])
    df_removedots = df_removedots.rename_axis('motifs', axis='index')
    #df_removedots = pd.DataFrame(sizes, columns=['Original String', 'Original Size', 'Modified String', 'Modified Size'], index = [kmer_pool])

    # 2nd step: combine df_removedots and df_input
    df_combined = pd.merge(df_removedots, df_input, on='miRNA')
    df_combined.index = kmer_pool
    df_combined.drop(columns=['miRNA'],inplace=True)

    # 3rd step: Sort the index values based on ['Modified String', 'Original Size']
    df_sorted = df_combined.sort_values(['Modified String', 'Original Size'], ascending=[True, True])

    # 4th step: Remove duplicate rows based on a subset of columns, not counting 'Original Size'
    subset_columns = df_sorted.columns[1:]  # Specify the columns to consider
    df_no_duplicates = df_sorted.drop_duplicates(subset=subset_columns,)

    # 5th step: Drop certain columns
    columns_to_drop = ['Original Size','Modified String', 'Modified Size']  # Specify the columns to drop
    df_dropped = df_no_duplicates.drop(columns=columns_to_drop)
    
    return df_dropped

def my_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = round(tp/(tp+fn),2)
    spec =  round(tn/(tn+fp),2)
    FPR =  round(fp/(tn+fp),2)
    FNR =  round(fn/(tp+fn),2)
    accuracy = round((tp+tn)/len(y_true),2)
    f1score = round(2*tp/(2*tp+fp+fn),2)
    
    return (tp, tn, fp, fn, sens, spec, FPR, FNR, accuracy, f1score)

def my_LASSO(X_train_pos_motif,y_train,alphas, model):
    
    #import numpy as np
    import pandas as pd
    import sub_funcs as my_subfunc

    from sklearn.linear_model import LassoCV, LogisticRegressionCV

    coefs = []
    lass_motif_list = []
    n_nonzero_list = []
    nonzero_X_train_pos_list = []
    train_performance_metrics_list = []

    for i, a in enumerate(alphas):  # for each alpha
        print("Iteration: ", i)
        
        # current lasso model with alphas = a
        #model.set_params(alphas=[a])
        
        # fit the model
        #model.fit(X_train_pos_motif, y_train)
        
        # calculate nonzero coefficients indices
        if isinstance(model, LogisticRegressionCV):
            
            model.set_params(Cs=[a])
            model.fit(X_train_pos_motif, y_train)
            
            
            nonzero_coef_indices = model.coef_ != 0
            nonzero_coef_indices = nonzero_coef_indices[0]
        elif isinstance(model, LassoCV):
            
            # current lasso model with alphas = a
            model.set_params(alphas=[a])
            
            # fit the model
            model.fit(X_train_pos_motif, y_train)
            
            nonzero_coef_indices = model.coef_ != 0
        else:
            raise ValueError("Unsupported model type")
            
        # X_train with nonzero coefficient motifs
        nonzero_X_train_pos = X_train_pos_motif.iloc[:, nonzero_coef_indices]
        # total number of the nonzero coefficients
        n_nonzero = len(nonzero_X_train_pos.columns)#np.sum(nonzero_coef_indices)
        
        # motifs with nonzero coefficients
        selected_motif = nonzero_X_train_pos.columns
        #selected_motif = list(compress(motif_list_pos_only, nonzero_coef_indices))
        
        # 
        y_train_pred = [1 if item == True else 0 for item in nonzero_X_train_pos.any(axis=1)]
        train_performance_metrics = my_subfunc.my_confusion_matrix(y_train, y_train_pred)
        
        coefs.append(model.coef_)
        n_nonzero_list.append(n_nonzero)
        lass_motif_list.append(selected_motif)
        nonzero_X_train_pos_list.append(nonzero_X_train_pos.transpose())
        train_performance_metrics_list.append(train_performance_metrics)
        
    df_n_nonzero_list = pd.DataFrame(n_nonzero_list, columns=['N_nonzeros'])

    df_motif_alpha_list_train = pd.concat(nonzero_X_train_pos_list, keys=range(len(nonzero_X_train_pos_list)))
    df_motif_alpha_list_train = df_motif_alpha_list_train.reset_index()
    df_motif_alpha_list_train = df_motif_alpha_list_train.rename(columns={'level_0': 'Model'})
    df_motif_alpha_list_train = df_motif_alpha_list_train.rename(columns={'level_1': 'Motifs'})


    df_perfmetric_train = pd.DataFrame(train_performance_metrics_list, columns=['train_tp', 'train_tn', 'train_fp', 'train_fn', 'train_sens', 'train_spec', 'train_fpr', 'train_fnr', 'train_acc', 'train_f1'])
    df_perfmetric_train_nonzero = pd.merge(df_n_nonzero_list, df_perfmetric_train, left_index=True, right_index=True)
    
    df_final = pd.merge(df_perfmetric_train_nonzero, df_motif_alpha_list_train, left_index=True, right_on='Model')
    
    df_coefs = pd.DataFrame(coefs)
    
    return df_final, df_coefs, df_perfmetric_train_nonzero, df_motif_alpha_list_train

