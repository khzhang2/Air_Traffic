from operator import index
import numpy as np
import pandas as pd

import multiprocessing
from multiprocessing import Pool

def construct_OD(process_name, from_ind, to_ind, data, airport_lst, OD):
    print('Start process' + process_name)
    
    for i in range(from_ind, to_ind):
        # [from_ind, to_ind)
        for j in range(len(airport_lst)):
            print(process_name, i-from_ind, '/ %i'%(to_ind-from_ind), j, '/ %i'%len(airport_lst), )
            trip_count = data.loc[(data['Origin']==airport_lst[i]) \
                        & (data['Dest']==airport_lst[j])]['Passengers'].sum()

            OD.loc[airport_lst[i], airport_lst[j]] = trip_count
        
    print('End process' + process_name)
    return OD


if __name__ == '__main__':
    year = 2019
    quarter = 2
    path = './data/Origin_and_Destination_Survey_DB1BMarket_%i_%i.csv'%(year, quarter)
    trip_data = pd.read_csv(path)

    airport_lst = list(pd.read_csv('./data/airport_lst_201904_org.csv', index_col=0).values.flatten())
    for i in list(trip_data['Origin'].drop_duplicates().values.flatten()):
        if i not in airport_lst:
            airport_lst.append(i)
    for i in list(trip_data['Dest'].drop_duplicates().values.flatten()):
        if i not in airport_lst:
            airport_lst.append(i)

    num_airports = len(airport_lst)  # =437 in 201904
    # dims: (org, dest)
    OD = pd.DataFrame(0, index=airport_lst, columns=airport_lst)

    num_interval = int(multiprocessing.cpu_count()*0.9)
    interval = len(airport_lst)//num_interval * np.arange(num_interval)
    interval = np.append(interval, len(airport_lst))

    n_cpu = num_interval

    pool = Pool(processes=n_cpu)
    params = []
    for i in range(len(interval)-1):
        from_ = interval[i]
        to_ = interval[i+1]
        process_name = 'P' + str(i)
        params.append((process_name, from_, to_, trip_data, airport_lst, OD))

    OD_set = pool.starmap(func=construct_OD, iterable=params)

    for i in range(n_cpu):
        # OD_set[i].to_csv('./outputs/%i0%i_p_%i_res.csv'%(year, quarter, i))
        OD += OD_set[i].fillna(0)

    OD.to_csv('./outputs/%i0%i_OD.csv'%(year, quarter))

    # please set a breakpoint here, check the stored data
    print('end')