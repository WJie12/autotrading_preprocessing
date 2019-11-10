import tushare as ts
import pandas as pd
import os
from sklearn.utils import shuffle


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
        big_file = pd.concat([big_file, reader])
    print("Head of file:\n", big_file.head())
    print("Row number of file:\n", len(big_file))
    return big_file


def write_file():
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

def generate_sub_table():
    df = pd.read_csv('./all_stocks_all_day.csv')
    stock_code_list = pd.read_csv('./stock_table.csv')['code']
    stock_file = df[df['SecurityID'].isin(stock_code_list)]
    stock_file.to_csv('./19_stock_all_day.csv', index=False)

def check_pearson_corr():
    pass


# pick_stock()
# stock_data_preprocess()
write_file()



