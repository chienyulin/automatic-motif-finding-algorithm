
import time
import pandas as pd
import re
import multiprocessing
from itertools import compress
import sub_funcs as my_subfunc

ta = time.time()

#%% parameter setting
# input filename
seq_filename = "training_testing_sequence.xlsx"
# given min and max kmer
kmin = 2
kmax = 10
tol = 0.25 # the count in positive group need to be greater than N_pos*tol
# use parallel computing or not
is_parallel = 0
# save the ouput to a csv file or not and filename
is_save_to_file = 1
save_to_file = "K-mer_candidates.xlsx"

#%% subfunctions -- no parameter setting in this section
# read excel file into pandas DataFrame and create header
df = pd.read_excel(seq_filename, sheet_name='train')
df['sequence'] = df['sequence'].str.lower()

# seperate postive group and negative group
ind_pos = list(df.iloc[:,2]==1)
ind_neg = list(df.iloc[:,2]==0)
df_pos = df.loc[ind_pos,:]
df_neg = df.loc[ind_neg,:]

# display DataFrame
#print(df)

# get sequence data from the original dataframe, positive group and negative group
seqs = list(df.iloc[:,1])
seq_pos = list(df_pos.iloc[:,1])
seq_neg = list(df_neg.iloc[:,1])

# define a prefix to indicate the beginning of a new sequence when merging all sequence to one long string
prefix = "!"# or suffix

# add prefix to each sequence in the list
prefix_seq = [prefix + sub for sub in seqs]
# add suffix to each sequence in the list
#suffix_seq = [sub + prefix for sub in seqs]

# merge all sequences in the list to ONE sequence 
# motif with prefix sign will be removed after the count_kmers function
prefix_seq = ''.join(prefix_seq)

def optimize_kmer(K):   
    print("========================")
    # generator function to yield kmer substrings
    def k_length_substrings(s, k):
        for i in range(len(s)-k+1):
            yield s[i:i+k]
     
    # Extract K length substrings
    ta = time.time()
    res = list(k_length_substrings(prefix_seq, K))
    tb = time.time()
     
    # printing executive time
    print(K,'mer substring Time:', tb-ta)
    
    # remove string that includes prefix/suffix
    res_remove_prefix = [s for s in res if prefix not in s]
    # unique kmer substring
    res_unique = list(set(res_remove_prefix))
    
    # create pat with wildcards in all the positions, predefine wildcard positions
    wildcard_positions = list(range(K))  
    
    t0 = time.time()
    # Replace wildcards in each res_unique
    output_set = set()
    for key in res_unique:
        combinations = [[]]
        for pos in wildcard_positions:
            combinations += [c + [pos] for c in combinations]
        
        for comb in combinations:
            this_str = list(key)
            for index in comb:
                this_str[index-1] = '.'
            output_set.add(''.join(this_str))
    
    # Sort the output
    #output = sorted(output_set)
    t1 = time.time()
    print(K,'mer wildcards Time:', t1-t0, '; len(w/wildc, w/o wildC)', len(res_unique), len(output_set))

    return output_set

# Function to check if a pattern is present in a group
def is_pattern_exist(pattern, group):
    is_match = [1 if re.search(pattern, string) else 0 for string in group]
    return is_match, any(is_match), sum(is_match)

#%% Main -- no parameter setting in this section
cutoff_count_pos = len(seq_pos)*tol
cutoff_count_neg = len(seq_neg)*tol

#k_array = [k for k in range1(kmin, kmax)]
k_array = list(range(kmin, kmax+1))

kmer_pool = []
occurrence_list = []
count_list = []
Tstart = time.time()

if not is_parallel:
    # original     
    for num in k_array:
        kmer_list = optimize_kmer(num)
            
        for pattern in kmer_list:
           
            # consider (in pos and not neg) and (in neg and not pos)
            exist_array_group = [1 if re.search(pattern, string) else 0 for string in seqs]
            
            exist_count_in_neg = sum(list(compress(exist_array_group, ind_neg)))
            exist_count_in_pos = sum(list(compress(exist_array_group, ind_pos)))
                   
            if (exist_count_in_pos >= cutoff_count_pos and not exist_count_in_neg) or (exist_count_in_neg >= cutoff_count_neg and not exist_count_in_pos):
                
                count_array_group = [my_subfunc.my_overlappingcount(pattern, string) for string in seqs]
                count_list.append(count_array_group)
                
                kmer_pool.append(pattern)
                occurrence_list.append(exist_array_group)
    
    Tend = time.time()
    print("========================")
    print('Total executive time:', Tend-Tstart)

    # prepare occurrence_list to dataframe for csv file
    df_kmer = pd.DataFrame(occurrence_list,columns=list(df['miRNA']),index=[kmer_pool])
    df_count = pd.DataFrame(count_list,columns=list(df['miRNA']),index=[kmer_pool])

    # sort dataframe before export to a file    
    # save the dataframe to a file
    if is_save_to_file:
        writer = pd.ExcelWriter(save_to_file, engine='xlsxwriter')
        
        df_kmer.to_excel(writer, sheet_name='existence')
        df_count.to_excel(writer, sheet_name='count')
        
        writer.close()

    print('Done')
    
    
else:
    # parallel computing
    def process_pattern(pattern):
        
        # consider (in pos and not neg) and (in neg and not pos)
        exist_array_group = [1 if re.search(pattern, string) else 0 for string in seqs]
        
        exist_count_in_neg = sum(list(compress(exist_array_group, ind_neg)))
        exist_count_in_pos = sum(list(compress(exist_array_group, ind_pos)))
               
        if (exist_count_in_pos >= cutoff_count_pos and not exist_count_in_neg) or (exist_count_in_neg >= cutoff_count_neg and not exist_count_in_pos):
            count_array_group = [my_subfunc.my_overlappingcount(pattern, string) for string in seqs]            
            return (pattern, exist_array_group, count_array_group)
        return None
    
    if __name__ == '__main__':
        pool = multiprocessing.Pool()
    
        results = []
        for num in k_array:
            kmer_list = optimize_kmer(num)
            results.extend(pool.map(process_pattern, kmer_list))
        
        pool.close()
        pool.join()
    
        kmer_pool = [result[0] for result in results if result is not None]
        occurrence_list = [result[1] for result in results if result is not None]
        count_list = [result[2] for result in results if result is not None]
    
    
        Tend = time.time()
        print("========================")
        print('Total executive time:', Tend-Tstart)
    
        # prepare occurrence_list to dataframe
        #df_kmer = pd.DataFrame(occurrence_list,columns=[df.iloc[:,0]],index=[kmer_pool])
        df_kmer = pd.DataFrame(occurrence_list,columns=list(df['miRNA']),index=[kmer_pool])
        df_count = pd.DataFrame(count_list,columns=list(df['miRNA']),index=[kmer_pool])
        
        # sort dataframe before export
        # save the dataframe
        if is_save_to_file:
            writer = pd.ExcelWriter(save_to_file, engine='xlsxwriter')
            
            df_kmer.to_excel(writer, sheet_name='existence')
            df_count.to_excel(writer, sheet_name='count')
            
            writer.close()

        print('Done')

tb = time.time()
# printing executive time
print('Executive time:', tb-ta)
