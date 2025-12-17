
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sub_funcs as my_subfunc
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
import re

#%% read datasets
ta = time.time()

# files
seq_file = "training_testing_sequence.xlsx"
motif_file = "K-mer_candidates.xlsx"

is_save_to_file = 1
save_to_file = 'automatic_motif_finding_results.xlsx'

# train dataframe
df_seq_train = pd.read_excel(seq_file, sheet_name='train')
df_seq_train['sequence'] = df_seq_train['sequence'].str.lower()     # make all train seqs lower cases
# motif list dataframe
df_motif = pd.read_excel(motif_file, sheet_name='existence')
df_motif = df_motif.rename(columns={'Unnamed: 0': 'miRNA'})
# motif list 
motif_list = df_motif['miRNA']
df_motif.index = motif_list
df_motif = df_motif.rename_axis('motifs', axis='index')
#df_motif.drop(columns=['miRNA'],inplace=True)

# clean up the df_motif
df_motif_clean = my_subfunc.df_unique_rows(df_motif)
motif_list_clean = df_motif_clean.index

# # clean up the df_motif
# df_motif_clean = df_motif.drop(columns='miRNA')
# motif_list_clean = df_motif_clean.index

# prepare X, y from train for fitting
#X_train = df_motif.iloc[:,1:].transpose()
X_train = df_motif_clean.transpose()
# standardization of the train data before using lasso
scaler = MinMaxScaler().fit(X_train) 
# standardize the x train and x test data using the coefficients from the x train data
X_train_scaled = scaler.transform(X_train)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)


X_train.columns = motif_list_clean
y_train = df_seq_train['group']
y_train_reverse = 1-y_train                     # reverse y_train for Exp2

# seperate X_train to two groups
X_train_pos = X_train.loc[list(df_seq_train['group']==1),:]     # X_train_positive_inflammatory
X_train_neg = X_train.loc[list(df_seq_train['group']==0),:]     # X_train_negative_inflammatory

ind_pos_motif_only = X_train_pos.any(axis='rows')   # indices of motifs only in positive inflammatory
ind_neg_motif_only = X_train_neg.any(axis='rows')   # indices of motifs only in negative inflammatory

X_train_pos_motif = X_train.loc[:, ind_pos_motif_only]  # X_train_motifs only in positive inflammatory
X_train_neg_motif = X_train.loc[:, ind_neg_motif_only]  # X_train_motifs only in negative inflammatory

#motif_list_pos_only = motif_list[ind_pos_motif_only]    # list of motifs only in positive inflammatory
#motif_list_neg_only = motif_list[ind_neg_motif_only]    # list of motifs only in negative inflammatory

# # seqs for prediction
# test_seq = df_seq_test['sequence']

# plot fig size
fig_size_len = 8
fig_size_wid = 6
fig_dpi = 300

#%% Exp1: use motifs only in pos group
# testing different alpha values impact on the model
# define alpha list for Lasso model
nsample = 42
alphas = np.logspace(-2, 0, nsample)
lassocv = LassoCV(cv=5, max_iter=10000)

exp1_df_final, exp1_coefs, exp1_df_perfmetrics, exp1_df_motif_alpha_list = my_subfunc.my_LASSO(X_train_pos_motif, y_train, alphas,lassocv)

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'lines.linewidth':3})

df_Lambda = pd.DataFrame({'Lambda': alphas}, index=exp1_df_perfmetrics.index)
exp1_df_perfmetrics = pd.concat([df_Lambda, exp1_df_perfmetrics], axis=1)
exp1_df_perfmetrics.index.name = 'Model'

df_results = exp1_df_perfmetrics[['Lambda', 'N_nonzeros', 'train_f1']].copy()

df_results.sort_values(by=['train_f1','N_nonzeros','Lambda'], ascending=[False,True,False],inplace = True,ignore_index=False)

# get the final motif list
final_decision_idx = df_results.index[0]

s = exp1_df_motif_alpha_list['Model']
matching_idx = s.index[s.eq(final_decision_idx)]

final_decision_motifList = exp1_df_motif_alpha_list.loc[matching_idx, 'motifs']
final_decision_alpha = df_results['Lambda'][final_decision_idx]

print("The final motif list:", final_decision_motifList.tolist())

# plot the hyperparameter tuning results for the LASSO model
fig, ax1 = plt.subplots(figsize=(fig_size_len,fig_size_wid),dpi=fig_dpi)

color = 'tab:red'
ax1.set_xlabel(chr(955))
ax1.set_ylabel('F1 score', color=color)
plt.axvline(x = final_decision_alpha, color = 'k', label = 'axvline - full height', linewidth = 1,linestyle='--')
ax1.plot(alphas, exp1_df_perfmetrics['train_f1'], color=color,label='_nolegend_')
#plt.ylim((0,1))
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')
ax1.legend(['decision\npoint'], frameon=False)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Number of non-zero coeff motifs', color=color)  # we already handled the x-label with ax1
ax2.plot(alphas, exp1_df_perfmetrics['N_nonzeros'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
#plt.ylim((0,15))
ax2.set_xscale('log')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('LASSO_model_selection.png')
plt.show()

# write the final motifs to a txt file
with open("final_motifs.txt", "w") as text_file:
    text_file.write("Selected LASSO model lambda: %s \n" % final_decision_alpha)
    text_file.write("Final motifs: \n")
    text_file.write("\n".join(final_decision_motifList.tolist()) + "\n")

# save the dataframe
if is_save_to_file:
    writer = pd.ExcelWriter(save_to_file, engine='xlsxwriter')
    
    exp1_df_perfmetrics.to_excel(writer, sheet_name='perfmetric')
    exp1_df_motif_alpha_list.to_excel(writer, sheet_name='motif', index=False)
    
    writer.close()

print('Done')

tb = time.time()
# printing executive time
print('Executive time:', tb-ta)

#%% performance result for Testing data
# test dataframe
df_seq_test = pd.read_excel(seq_file, sheet_name='test')
df_seq_test['sequence'] = df_seq_test['sequence'].str.lower()       # make all test seqs lower cases

# prepare y from test for prediction
y_test = pd.DataFrame(list(df_seq_test['group']),columns=['group'],index=[df_seq_test.iloc[:,0]])

# seqs for prediction
test_seq = df_seq_test['sequence']

# the final motifs selected from the motif finding algorithm
selected_motif = final_decision_motifList.tolist()

# existence table on test seqs
occurrence_list = []
for selected_motif_ind in selected_motif:
    exist_test_group = [1 if re.search(selected_motif_ind, string) else 0 for string in test_seq]
    occurrence_list.append(exist_test_group)
    
# prepare prediction result to dataframe for csv file
df_motif_test = pd.DataFrame(occurrence_list,columns=df_seq_test.iloc[:,0].tolist(),index=[selected_motif])
df_motif_test = df_motif_test.transpose()
df_motif_test_list = df_motif_test.copy()
df_motif_test_list = df_motif_test_list.reset_index()
df_motif_test_list.columns = df_motif_test_list.columns.get_level_values(0)
df_motif_test_list = df_motif_test_list.rename(columns={df_motif_test_list.columns[0]: 'miRNA'})

# performance results for testing set
pred_test_bool = df_motif_test.any(axis=1)
y_test_pred = pd.DataFrame([1 if item == True else 0 for item in pred_test_bool],columns=['pred_group'],index=[df_seq_test.iloc[:,0]])

test_performance_metrics = my_subfunc.my_confusion_matrix(y_test, y_test_pred)
test_performance_metrics_list = []
test_performance_metrics_list.append(test_performance_metrics)
df_perfmetric_test = pd.DataFrame(test_performance_metrics_list, columns=['test_tp', 'test_tn', 'test_fp', 'test_fn', 'test_sens', 'test_spec', 'test_fpr', 'test_fnr', 'test_acc', 'test_f1'])

# save the testing result to excel
if is_save_to_file:
    save_to_file_test = save_to_file.replace('.xlsx', '_testingset.xlsx')
    writer = pd.ExcelWriter(save_to_file_test, engine='xlsxwriter')
    
    df_perfmetric_test.to_excel(writer, sheet_name='perfmetric', index=False)
    df_motif_test_list.to_excel(writer, sheet_name='motif', index=False)
    
    writer.close()

print('Done')
