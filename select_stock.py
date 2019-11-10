import tushare as ts
import pandas as pd
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns


def concat_all_file(path):
    flag = True
    for file in os.listdir(path):
        print(file)
        reader = pd.read_csv(os.path.join(path, file, 'Day.csv'), skiprows=1, names=\
            ['SecurityID', 'DateTime', 'PreClosePx', 'OpenPx', 'HighPx', 'LowPx', 'LastPx',\
             'Volumne', 'Amount', 'IOPV', 'fp_Volume', 'fp_Amount'
             ])
        if flag:
            big_file = reader
            flag = False
        else:
            big_file = pd.concat([big_file, reader])
    print("Head of file:\n", big_file.head())
    print("Row number of file:\n", len(big_file))
    return big_file

def generate_train_test_all_table():
    test_path = './data/quot/test'
    train_path = './data/quot/train'

    print("########Strat: concat test data file..." + '\n')
    test_file = concat_all_file(test_path)
    test_file.to_csv('./test_data.csv', index=False)
    print("########Finish: concat test data file. Length of test file:", len(test_file))

    print("########Strat: concat train data file..." + '\n')
    train_file = concat_all_file(train_path)
    train_file.to_csv('./train_data.csv', index=False)
    print("########Finish: concat trian data file. Length of train file:", len(train_file))

    print("########Strat: concat train data file and test data file..." + '\n')
    all_file = pd.concat([test_file, train_file])
    all_file.to_csv('./all_data.csv', index=False)

# pick stock by industry randomly
def pick_stock():
    # get all stocks: code, name, c_name
    df = ts.get_industry_classified()
    # filter stocks, 600* means Stocks of SH
    df_sh = df[(df.code.str.startswith('600'))]
    # shuffle
    df_sh_shuffle = shuffle(df_sh)
    # pick 1 stock from every industry
    stock_table_all_c = df_sh_shuffle.drop_duplicates(subset=['c_name'], keep='first')
    # pick 19 stocks randomly
    stock_table = stock_table_all_c.sample(n=19)
    print(stock_table)
    # save stock table
    stock_table.to_csv('./stock_table.csv', index=False)


def stock_data_preprocess():
    pro = ts.pro_api()
    s_qjd = '600597'
    s_gm = '600033' #福建高速
    start_date = '2016-01-01'#起止日期
    end_date = '2016-12-31'
    df_qjd = pro.daily(ts_code=s_qjd, start_date=start_date, end_date=end_date)
    df_gm = pro.daily(ts_code=s_gm, start_date=start_date, end_date=end_date)
    df = pd.concat([df_qjd.close,df_gm.close], axis = 1, keys=['qjd_close', 'gm_close'])#合并
    df.ffill(axis=0, inplace=True) #填充缺失数据
    df.to_csv('qjd_gm.csv')

def generate_sub_table(in_path, out_path):
    df = pd.read_csv(in_path)
    stock_code_list = pd.read_csv('./stock_table.csv')['code']
    stock_file = df[df['SecurityID'].isin(stock_code_list)]
    stock_file.to_csv(out_path, index=False)

def generate_sub_test_train_all_table():
    # def generate_sub_test_table():
    test_data = "./test_data.csv"
    sub_test_data = "./sub_test_data.csv"
    generate_sub_table(test_data, sub_test_data)

    # def generate_sub_train_table():
    train_data = "./train_data.csv"
    sub_train_data = "./sub_train_data.csv"
    generate_sub_table(train_data, sub_train_data)

    # def generate_sub_all_table():
    all_data = "./all_data.csv"
    sub_all_data = "./sub_all_data.csv"
    generate_sub_table(all_data, sub_all_data)

def transfer_table(in_path, out_path):
    old_df = pd.read_csv(in_path)
    stock_code_lsit = old_df['SecurityID'].unique()
    date = old_df.drop_duplicates(subset=['DateTime'], keep='first').loc[:, ['DateTime']]
    for i in range(len(stock_code_lsit)):
        new = old_df[old_df['SecurityID'].isin([stock_code_lsit[i]])][['DateTime', 'LastPx']]
        new.columns = ['DateTime', str(stock_code_lsit[i])]
        date = pd.merge(date, new, how='left', on=['DateTime'])
    # fill nan with pre-LastPx
    date.ffill(axis=0, inplace=True)
    date.to_csv(out_path, index=False)

def transfer_sub_table():
    # def transfer_sub_test_table():
    sub_test_data = "./sub_test_data.csv"
    test_set = "./test_set.csv"
    transfer_table(sub_test_data, test_set)

    # def transfer_sub_train_table():
    sub_train_data = "./sub_train_data.csv"
    train_set = "./train_set.csv"
    transfer_table(sub_train_data, train_set)

    # def transfer_sub_all_table():
    sub_all_data = "./sub_all_data.csv"
    all_data_set = "./all_set.csv"
    transfer_table(sub_all_data, all_data_set)


def check_pearson_corr(path):
    df = pd.read_csv(path)
    df = df.drop(['DateTime'], axis=1)
    # plot LastPx
    df.plot(figsize=(14, 9))
    plt.savefig('./19_stocks_raw.png')
    plt.close()
    # plot percentage
    df_pc = df.pct_change()
    df_pc.plot(figsize=(14, 9))
    plt.savefig('./19_stock_pct.png')
    plt.close()
    # pair plot
    sns.pairplot(df.dropna())
    plt.savefig('./19_stocks_pairplot.png', figsize=(9, 9))
    plt.close()
    # calculate corr
    corr = df.corr(method='pearson', min_periods=1)  # pearson方法计算相关性
    print(corr)
    corr.to_csv('./train_set_corr.csv', index=False)
    # corr heatmap
    sns.heatmap(corr)
    plt.savefig('./19_stocks_heatmap.png', figsize=(9,9))
    plt.close()


# pick_stock()
# stock_data_preprocess()
# generate_train_test_all_table()
# generate_sub_test_train_all_table()
# transfer_sub_table()
check_pearson_corr('./train_set.csv')




