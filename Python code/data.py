import sys, os
import csv
from time import sleep
from collections import defaultdict
import pandas as pd


def import_data(save=True,try_from_csv=False,saved_csvs=['MaleRankCount.csv','FemaleRankCount.csv',
                                                                       'MaleRankIDs.csv','FemaleRankIDs.csv',
                                                                       'male_name_ids.csv','female_name_ids.csv']):
    cwd = os.getcwd()
    subdir = 'data'
    path = os.path.join(cwd,subdir)
    dir_info = os.walk(path)
    (_,_,files) = dir_info.__next__()
    csv_files = set([file for file in files if '.csv' in file])
    txt_files = [file for file in files if '.txt' in file]
    name_id = {'M':{},'F':{}}
    ranks_count = {'M':None,'F':None}
    ranks_ids = {'M':None,'F':None}
    if try_from_csv and len(saved_csvs) == 6 and csv_files.issuperset(saved_csvs):
        try:
            sys.stdout.write('\nImporting from CSV files...\n')
            sys.stdout.flush()
            show_status(0./6)
            with open(os.path.join(path,saved_csvs[0])) as f:
                ranks_count['M'] = pd.read_csv(f)
            show_status(1./6)
            with open(os.path.join(path,saved_csvs[1])) as f:
                ranks_count['F'] = pd.read_csv(f)
            show_status(2./6)
            with open(os.path.join(path,saved_csvs[2])) as f:
                ranks_ids['M'] = pd.read_csv(f)
            show_status(3./6)
            with open(os.path.join(path,saved_csvs[3])) as f:
                ranks_ids['F'] = pd.read_csv(f)
            show_status(4./6)
            with open(os.path.join(path,saved_csvs[4])) as f:
                curr_reader = csv.reader(f)
                for id,name in curr_reader:
                    name_id['M'][int(id)] = name
            show_status(5./6)
            with open(os.path.join(path,saved_csvs[5])) as f:
                curr_reader = csv.reader(f)
                for id,name in curr_reader:
                    name_id['F'][int(id)] = name
            show_status(1)
            return name_id, ranks_count, ranks_ids
        except:
            sys.stdout.write("\nError: Could not import from CSV. Attempting import from txt files\n")
            sys.flush()
    rankings = {i:defaultdict(list) for i in range(len(txt_files))}
    mmax_len = 0; fmax_len = 0; high_prime = 104729 # 449563
    sys.stdout.write('Extracting Data From txt files...\n')
    sys.stdout.flush()
    for i in range(len(txt_files)):
        show_status(float(i+1)/len(txt_files))
        file = txt_files[i]
        with open(os.path.join(path,file)) as f:
            data = list(csv.reader(f))
            for name,gender,births in data:
                h = str2hash(name,maxval=high_prime)
                hval = name_id.get(h,'-1')
                count = 1
                while name != hval and hval != '-1':
                    h = (h**count + ord(name[0])**count + ord(name[-1])**count) % high_prime
                    count+=1
                    hval = name_id.get(h,'-1')
                if name != hval:
                    name_id[gender][h] = name
                rankings[i][gender].append((h,births))
            mmax_len = max(mmax_len,len(rankings[i]['M']))
            fmax_len = max(fmax_len,len(rankings[i]['F']))

    ranks_count = {'M':pd.DataFrame(columns=list(range(len(rankings.keys()))), index=list(range(mmax_len)))
             , 'F':pd.DataFrame(columns=list(range(len(rankings.keys()))), index=list(range(fmax_len)))}
    ranks_ids = {'M':pd.DataFrame(columns=list(range(len(rankings.keys()))), index=list(range(mmax_len)))
             , 'F':pd.DataFrame(columns=list(range(len(rankings.keys()))), index=list(range(fmax_len)))}
    sys.stdout.write('\nConstructing DataFrame...\n')
    sys.stdout.flush()
    for year in rankings.keys():
        #show_status(float(year + 1)/len(rankings.keys()))
        for gender in rankings[year].keys():
            size = len(rankings[year][gender])
            ranks_ids[gender].loc[:size-1,year] = [rankings[year][gender][i][0] for i in range(size)]
            ranks_count[gender].loc[:size-1,year] = [int(rankings[year][gender][i][1]) for i in range(size)]
    if save:
        sys.stdout.write('\nSaving Data...\n')
        sys.stdout.flush()
        with open(os.path.join('data','male_name_ids.csv'),'w',newline='') as fout:
            w = csv.writer(fout)
            w.writerows(name_id['M'].items())
        with open(os.path.join('data','female_name_ids.csv'),'w',newline='') as fout:
            w = csv.writer(fout)
            w.writerows(name_id['F'].items())
        export_data(df=ranks_count['M'],filename='MaleRankCount.csv',path='data')
        export_data(df=ranks_count['F'],filename='FemaleRankCount.csv',path='data')
        export_data(df=ranks_ids['M'],filename='MaleRankIDs.csv',path='data')
        export_data(df=ranks_ids['F'],filename='FemaleRankIDs.csv',path='data')
    return name_id,ranks_count,ranks_ids

def export_data(df=None,filename='temp.csv',path=None,):
    if 'sys' not in vars().keys():
        import sys
    if 'os' not in vars().keys():
        import os
    if not path: # No path specified
        path = os.getcwd()
    else: # Checking if path exists
        if not os.path.exists(path):
            if not os.path.exists(os.path.join(os.getcwd(),path)):
                path = os.getcwd()
            else:
                path = os.path.join(os.getcwd(),path)
    # Checking if file exists, deletes if found!
    try:
        if os.path.isfile(os.path.join(path,filename)):
            os.remove(os.path.join(path,filename))
        fullpath = os.path.join(path,filename)
        df.to_csv(fullpath,index=False)
    except:
        sys.stdout.write('\nerror: Writing to date file')
        import time
        df.to_csv('temp'+time.strftime('%y_%m_%d__%H%M%S')+'.csv')


def str2hash(string,type='djb2',maxval=2**32-1):
    char_arr = [(ord(c)) for c in string]
    if type == 'djb2':
        hashval= 5381
        for i in range(len(char_arr)):
            hashval = (hashval * 33 + char_arr[i]) % maxval
    elif type == 'sdbm':
        hashval = 0
        for i in range(len(char_arr)):
            hash = (hash * 65599 + char_arr[i])% maxval
    else:
        print('string_hash:inputs','unknown type')
        return 0
    return hashval


def show_status(perc):
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int(perc*100/5), int(perc*100)))
    sys.stdout.flush()
    sleep(0.01)

if __name__ == "__main__":
    #a,b,c = import_data()
    a,b,c = import_data(try_from_csv=True)


