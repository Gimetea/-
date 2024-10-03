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
df = pd.read_csv('供选股数据.csv', encoding='gbk', parse_dates=['交易日期'], low_memory=False)  # 从csv文件中读取整理好的所有股票数据
# exit()
# ===筛选回测时间范围至2024年1月
end_date = '2024-01-01'  # 设置回测结束日期
df = df[df['交易日期'] <= pd.to_datetime(end_date)]  # 筛选数据时间范围
# ====================

# ===构建选股因子
df['因子'] = df['总市值']
# ===对股票数据进行筛选
df = df[df['上市至今交易天数'] > 250]  # 删除上市不满一年的股票
df = df[~df['股票代码'].str.contains('bj')] #去除北交所，~代表反选
#df = df[df['新版申万一级行业名称'].isin(['计算机'])]  # 对所在行业进行筛选
# # '钢铁', '交通运输', '房地产', '公用事业', '化工', '休闲服务', '医药生物', '商业贸易', '食品饮料', '家用电器', '轻工制造', '纺织服装', '综合', '农林牧渔', '有色金属', '采掘', '电子', '银行', '汽车', '非银金融', '机械设备', '传媒', '国防军工', '建筑装饰', '通信', '电气设备', '计算机', '建筑材料'
df = df[df['归母净利润_单季同比'] > 0 ]  # 对财务数据进行筛选
df = df[df['归母净利润'] > 0 ]
df = df[df['归母净利润_单季环比'] > 0 ]
# ===选股
df['排名'] = df.groupby('交易日期')['因子'].rank()  # 根据选股因子对股票进行排名
df = df[df['排名'] <= select_stock_num]  # 选取排名靠前的股票

# ===整理选中股票数据，计算涨跌幅
# 挑选出选中股票
df['股票代码'] += ' '
df['股票名称'] += ' '
df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: ast.literal_eval(x))
df
group = df.groupby('交易日期')
group
select_stock = pd.DataFrame()
select_stock['买入股票代码'] = group['股票代码'].sum()
select_stock['买入股票名称'] = group['股票名称'].sum()
select_stock
# 计算下周期每天的资金曲线
select_stock['选股下周期每天资金曲线'] = group['下周期每天涨跌幅'].apply(lambda x: np.cumprod(np.array(list(x))+1, axis=1).mean(axis=0))
# 扣除买入手续费
select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'] * (1 - c_rate)  # 计算有不精准的地方
# 扣除卖出手续费、印花税。最后一天的资金曲线值，扣除印花税、手续费
select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'].apply(lambda x: list(x[:-1]) + [x[-1] * (1 - c_rate - t_rate)])
# 计算下周期整体涨跌幅
select_stock['选股下周期涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(lambda x: x[-1] - 1)
del select_stock['选股下周期每天资金曲线']
# 计算整体资金曲线
select_stock.reset_index(inplace=True)
select_stock['资金曲线'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()
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

# ===画图
select_stock.set_index('交易日期', inplace=True)
plt.plot(select_stock['资金曲线'])
plt.show()



