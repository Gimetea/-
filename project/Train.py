import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('expand_frame_repr', False)  # 当列太多时显示不清楚

# ===选股参数设定
select_stock_num = 5
c_rate = 1.2 / 10000  # 手续费，手续费>万1.2
t_rate = 1 / 1000  # 印花税
risk_free_rate = 0.02 / 252 # 无风险收益率，假设年化2%，换算为日收益率
# ===导入数据
df = pd.read_csv('price_train.csv', encoding='gbk', parse_dates=['TRADE_DT'], low_memory=False)  # 从csv文件中读取整理好的所有股票数据
#数据合并处理

for i in range(1, 6):
    # 读取CSV文件
    df_temp = pd.read_csv(f'factor{i}_train.csv', encoding='gbk', parse_dates=[0], low_memory=False)

    # 重命名日期列
    df_temp = df_temp.rename(columns={'Unnamed: 0': 'TRADE_DT'})

    # 使用melt函数进行格式转换
    df_temp = pd.melt(df_temp,
                      id_vars=['TRADE_DT'],  # 指定不需要转换的列
                      var_name='S_INFO_WINDCODE',  # 指定新的列名，用于存储原来的列名
                      value_name=f'y{i}')  # 指定新的列名，用于存储值

    # 合并因子
    df = pd.merge(df, df_temp, on=['TRADE_DT', 'S_INFO_WINDCODE'])
# 计算每日涨幅
df0 = df.sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
df0['CHANGE_RATE'] = df0.groupby('S_INFO_WINDCODE')['S_DQ_ADJCLOSE'].pct_change()
df0.loc[df0['TRADE_DT'] == pd.to_datetime('2019-01-02'), 'CHANGE_RATE'] = 0
df = df0.sort_values(['TRADE_DT', 'S_INFO_WINDCODE'])
# ====================
df
# ===构建选股因子
df['因子'] = df['y3']
# ===对股票数据进行筛选

# ===选股
df['排名'] = df.groupby('TRADE_DT')['因子'].rank()  # 根据选股因子对股票进行排名
df = df[df['排名'] <= select_stock_num]  # 选取排名靠前的股票
df = df.rename(columns={'TRADE_DT': '交易日期', 'CHANGE_RATE': '下周期每天涨跌幅', 'S_INFO_WINDCODE': '股票代码'})

# ===整理选中股票数据，计算涨跌幅
# 挑选出选中股票
df['股票代码'] += ' '
#df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].astype(float)
df
group = df.groupby('交易日期')
group
select_stock = pd.DataFrame()
select_stock['买入股票代码'] = group['股票代码'].sum()
# 计算下周期每天的资金曲线
select_stock['选股下周期每天资金曲线'] = group['下周期每天涨跌幅'].apply(lambda x: (x + 1).mean())
# 扣除买入手续费
select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'] * (1 - c_rate)  # 计算有不精准的地方
# 扣除卖出手续费、印花税
select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'] * (1 - c_rate - t_rate)
select_stock['资金曲线'] = select_stock['选股下周期每天资金曲线'].cumprod()
select_stock


# ===计算收益率
total_return = select_stock['资金曲线'].iloc[-1] - 1  # 最终资金曲线值减去初始值
print(f"最终收益率: {total_return * 100:.2f}%")

# ===计算夏普比率
select_stock['每日收益率'] = select_stock['资金曲线'].pct_change()  # 计算每日收益率
mean_daily_return = select_stock['每日收益率'].mean()  # 平均每日收益率
std_daily_return = select_stock['每日收益率'].std()  # 收益率的标准差（波动率）
sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return * np.sqrt(252)  # 夏普比率，年化
print(f"夏普比率: {sharpe_ratio:.2f}")

