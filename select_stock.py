import tushare as ts
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

# pick stock by industry randomly
def pick_stock():
    # get all stocks: code, name, c_name
    df = ts.get_industry_classified()
    # get stocks from 5 industries
    industry_list = ['汽车制造', '生物制药', '电子信息', '传媒娱乐', '家电行业']
    df = df.loc[df.c_name.isin(industry_list)]
    # filter stocks, 600* means Stocks of SH
    df_sh = df[(df.code.str.startswith('600'))]
    # shuffle
    df_sh_shuffle = shuffle(df_sh)
    # stock_table_all_c = df_sh_shuffle.drop_duplicates(subset=['c_name'], keep='first')
    stock_table_all_c = df_sh_shuffle.groupby(['c_name'], as_index=False)
    # pick 4 stocks from every industry
    stock_table_all_c_group = stock_table_all_c.apply(lambda _stock_table_all_c: _stock_table_all_c.sample(n=4))
    stock_table_all_c_df = pd.DataFrame(stock_table_all_c_group)
    # pick 19 stocks randomly, sort by c_name
    stock_table = stock_table_all_c_df.sample(n=19).sort_values(['c_name'])
    print(stock_table)
    # save stock table
    stock_table.to_csv('./stock_table.csv', index=False)


def concat_all_file(path):
    flag = True
    for file in os.listdir(path):
        print(file)
        reader = pd.read_csv(os.path.join(path, file, 'Day.csv'), skiprows=1, names=\
            ['SecurityID', 'DateTime', 'PreClosePx', 'OpenPx', 'HighPx', 'LowPx', 'LastPx',\
             'Volume', 'Amount', 'IOPV', 'fp_Volume', 'fp_Amount'
             ])
        if flag:
            big_file = reader
            flag = False
        else:
            big_file = pd.concat([big_file, reader])
    print("Head of file:\n", big_file.head())
    print("Row number of file:\n", len(big_file))
    return big_file


def generate_all_table():
    data_path = './data/quot/'
    print("########Strat: concat all data file..." + '\n')
    all_file = concat_all_file(data_path)
    all_file.to_csv('./all_data.csv', index=False)


def generate_sub_table(in_path, out_path):
    df = pd.read_csv(in_path)
    stock_code_list = pd.read_csv('./stock_table.csv')['code']
    stock_file = df[df['SecurityID'].isin(stock_code_list)]
    stock_file.to_csv(out_path, index=False)


def generate_sub_all_table():
    all_data = "./all_data.csv"
    sub_all_data = "./sub_all_data.csv"
    generate_sub_table(all_data, sub_all_data)


def transfer_table(in_path, out_path, key):
    old_df = pd.read_csv(in_path)
    stock_code_lsit = old_df['SecurityID'].unique()
    date = old_df.drop_duplicates(subset=['DateTime'], keep='first').loc[:, ['DateTime']]
    for i in range(len(stock_code_lsit)):
        new = old_df[old_df['SecurityID'].isin([stock_code_lsit[i]])][['DateTime', key]]
        new.columns = ['DateTime', str(stock_code_lsit[i])]
        date = pd.merge(date, new, how='left', on=['DateTime'])
    date = date.sort_values(['DateTime'])
    # fill nan with pre-LastPx
    date.ffill(axis=0, inplace=True)
    date.bfill(axis=0, inplace=True)
    date.to_csv(out_path, index=False)


def transfer_sub_all_table():
    sub_all_data = "./sub_all_data.csv"
    all_data_set = "./all_set.csv"
    key = 'LastPx'
    transfer_table(sub_all_data, all_data_set, key)


def generate_test_train_set():
    sub_all_set = pd.read_csv('./all_set.csv')

    train_set = sub_all_set.loc[(sub_all_set.DateTime < 20180000)]
    train_set.to_csv('./train_set.csv', index=False)

    test_set = sub_all_set.loc[(sub_all_set.DateTime > 20180000)]
    test_set.to_csv('./test_set.csv', index=False)


def check_pearson_corr(path):
    df = pd.read_csv(path)
    df = df.drop(['DateTime'], axis=1)
    # plot LastPx
    df.plot(figsize=(14, 9))
    plt.savefig('./19_stocks_raw.png')
    plt.close()
    # plot percentage
    df_pc = df.pct_change()
    df_pc.to_csv('./19_stocks_pct_change_on_train_set.csv')
    df_pc.plot(figsize=(14, 9))
    plt.savefig('./19_stock_pct.png')
    plt.close()
    # pair plot
    plt.figure(figsize=(15, 10))
    sns.pairplot(df.dropna())
    plt.savefig('./19_stocks_pairplot.png')
    plt.close()
    # calculate corr
    corr = df.corr(method='pearson', min_periods=1)  # pearson方法计算相关性
    print(corr)
    corr.to_csv('./train_set_corr.csv', index=False)
    # corr heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr)
    plt.savefig('./19_stocks_heatmap.png')
    plt.close()


def preprocess_sub_all_data():
    h5_file_path = './stock_history.h5'
    stock_code_list = pd.read_csv('./stock_table.csv')['code']
    l = []
    for key in stock_code_list:
        path = str(key)+'.csv'
        reader = pd.read_csv(path)
        l.append(reader.values)
        print(reader.shape)
    history = np.array(l)

    with h5py.File(h5_file_path, 'w') as f:
        f.create_dataset('history', data=history)


def transfer_sub_all_table_key():
    sub_all_data = "./sub_all_data.csv"
    col = ['DateTime','OpenPx', 'HighPx', 'LowPx', 'LastPx', 'Volumne']
    newcol = ['Open','High','low','Close','Volume']
    old_df = pd.read_csv(sub_all_data)
    stock_code_lsit = old_df['SecurityID'].unique()
    for i in range(len(stock_code_lsit)):
        date = old_df.drop_duplicates(subset=['DateTime'], keep='first').loc[:, ['DateTime']]
        out_path = str(stock_code_lsit[i]) + '.csv'
        new = old_df[old_df['SecurityID'].isin([stock_code_lsit[i]])][col]
        date = pd.merge(date, new, how='left', on=['DateTime'])
        date = date.sort_values(['DateTime'])
        # fill nan with pre-LastPx
        date.ffill(axis=0, inplace=True)
        date.bfill(axis=0, inplace=True)
        date = date.drop(columns=['DateTime'])
        date.columns = newcol
        date.to_csv(out_path, index=False)


def generate_all_set_key():
    col = ['OpenPx', 'HighPx', 'LowPx', 'LastPx', 'Volumne']
    for i in col:
        transfer_sub_all_table_key(i)


# pick_stock()
# generate_all_table()
# generate_sub_all_table()
# transfer_sub_all_table()
# generate_test_train_set()
# check_pearson_corr('./train_set.csv')

# generate_all_set_key()
# transfer_sub_all_table_key()
# preprocess_sub_all_data()
with h5py.File('stock_history.h5', 'r') as f:
    history = f['history'][:]
print(history)

