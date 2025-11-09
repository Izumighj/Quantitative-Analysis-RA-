import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os
from pandas.tseries.offsets import BDay  # <-- 【新增】用于计算交易日

# --- 1. Matplotlib 中文显示配置 ---
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'STHeiti', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 2. 策略参数配置中心 (CONFIG) ---
# (保持不变)
CONFIG = {
    # --- 文件路径 ---
    "DATA_PATH_FINANCIALS": "data/ind_financial_indicators_2015q1_2025q2.csv",
    "DATA_PATH_PRICES": "data/stk_ind_data_20150101_20251017.csv",
    
    # --- 行业筛选 ---
    "INDUSTRY_LIST": [
        "制冷空调设备", "其他专用设备", "光学元件", "通信网络设备及器件",
        "半导体材料", "锂", "锂电池", "电池化学品", "膜材料",
        "风电整机", "风电零部件", "线缆部件及其他", "硅料硅片",
        "光伏电池组件", "玻璃制造", "光伏辅材", "逆变器", "蓄电池及其他电池"
    ],
    
    # --- 数据清理参数 ---
    "CLEANING_COLS": ['q_sales_yoy', 'q_profit_yoy'],
    "WINDSORIZE_LOW": 0.01,
    "WINDSORIZE_HIGH": 0.99,
    
    # --- 核心策略参数 (可调) ---
    "PARAM_Y_GROWTH_METRICS": ['q_profit_yoy', 'q_sales_yoy'], 
    "PARAM_Y_GROWTH_LOGIC": 'AND', 
    "PARAM_Y_GROWTH_THRESHOLD": 20.0,
    "PARAM_Z_CONSECUTIVE_QUARTERS": 2,
    "PARAM_X_INDUSTRY_THRESHOLD": 0.5,
    
    # --- 回测参数 (可调) ---
    "PARAM_M_HOLDING_MONTHS": 3,
}

# --- 3. 功能函数 ---

def load_data(financial_path: str, price_path: str) -> (pd.DataFrame, pd.DataFrame):
    """加载财务和价格数据"""
    print(f"Loading financial data from {financial_path}")
    fs_df = pd.read_csv(financial_path)
    
    print(f"Loading price data from {price_path}")
    price_df = pd.read_csv(price_path)
    
    return fs_df, price_df

def clean_financial_data(df_raw: pd.DataFrame, 
                         ind_list: List[str], 
                         cols_to_clean: List[str], 
                         p_low: float, 
                         p_high: float) -> pd.DataFrame:
    """
    对原始财务数据进行筛选、填充缺失值和缩尾处理。
    【修改】: 确保 ann_date 和 end_date 都是 datetime
    """
    print("Step 2.1: Filtering industries...")
    df = df_raw[df_raw['l3_name'].isin(ind_list)].copy()
    
    # 【修改】确保日期被正确解析
    df['ann_date'] = pd.to_datetime(df['ann_date'], format='%Y%m%d', errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], format='%Y%m%d', errors='coerce')
    df.dropna(subset=['ann_date', 'end_date'], inplace=True) # 删除无法解析的日期

    print("Step 2.2: Filling missing values (NaN)...")
    # 填充缺失值：按“报告期”和“行业”分组，然后用该组的“中位数”来填充
    for col in cols_to_clean:
        df[col] = df.groupby(['end_date', 'l3_name'])[col].transform(
            lambda x: x.fillna(x.median())
        )
    # 使用“市场中位数”进行截面填充
    for col in cols_to_clean:
        df[col] = df.groupby('end_date')[col].transform(
            lambda x: x.fillna(x.median())
        )
    # 删除剩余的 NaT
    df.dropna(subset=cols_to_clean, inplace=True)

    print(f"Step 2.3: Winsorizing data at {p_low*100}% and {p_high*100}%...")
    for col in cols_to_clean:
        low_val = df[col].quantile(p_low)
        high_val = df[col].quantile(p_high)
        df[col] = df[col].clip(lower=low_val, upper=high_val)
        
    print("Financial data cleaning complete.")
    return df

def generate_stock_signals(df: pd.DataFrame, 
                         y_threshold: float, 
                         z_quarters: int,
                         growth_metrics: List[str],
                         logic: str
                         ) -> pd.DataFrame:
    """
    根据 Y (增长阈值), Z (连续季度) 和指定的逻辑(AND/OR) 生成个股信号。
    【修改】: 排序使用 end_date 来计算 streak，但函数返回的 DataFrame 包含
             所有需要的信息 (ann_date, ts_code, stock_signal)。
    """
    print(f"Step 3: Generating stock signals (Y={y_threshold}%, Z={z_quarters}q, Metrics={growth_metrics}, Logic={logic})...")
    
    # 关键: 必须按 end_date 排序来计算“连续季度”
    df_sorted = df.sort_values(by=['ts_code', 'end_date'])
    
    # 1. 标记每个指标是否达标
    metric_bool_cols = []
    for metric in growth_metrics:
        col_name = f'is_growth_{metric}'
        df_sorted[col_name] = df_sorted[metric] > y_threshold
        metric_bool_cols.append(col_name)

    # 2. 根据 'AND' 或 'OR' 逻辑，创建最终的 'is_growth' 列
    if logic.upper() == 'AND':
        df_sorted['is_growth'] = np.logical_and.reduce(
            [df_sorted[col] for col in metric_bool_cols]
        )
    elif logic.upper() == 'OR':
        df_sorted['is_growth'] = np.logical_or.reduce(
            [df_sorted[col] for col in metric_bool_cols]
        )
    else:
        print(f"Warning: Unknown logic '{logic}'. Defaulting to first metric only.")
        df_sorted['is_growth'] = df_sorted[metric_bool_cols[0]]

    # 3. 计算连续达标季度数
    rolling_sum = df_sorted.groupby('ts_code')['is_growth'].rolling(window=z_quarters).sum()
    df_sorted['consecutive_growth_count'] = rolling_sum.reset_index(level=0, drop=True)
    
    # 4. 创建“个股信号”
    df_sorted['stock_signal'] = (df_sorted['consecutive_growth_count'] == z_quarters)
    
    # 返回包含所有关键列的 DataFrame
    return df_sorted[['ts_code', 'ann_date', 'l3_name', 'stock_signal']]

def generate_industry_signals(df_stock_signals: pd.DataFrame, x_threshold: float) -> pd.DataFrame:
    """
    根据 X (行业阈值) 聚合个股信号，生成行业（策略）信号。
    【修改】: 按 ann_date 聚合，而不是 end_date。
    """
    print(f"Step 4: Generating industry signals (X={x_threshold*100}%)...")
    
    # 【修改】按“公告日”和“行业”分组
    industry_pct = df_stock_signals.groupby(['ann_date', 'l3_name'])['stock_signal'].mean()
    industry_signal_df = industry_pct.reset_index(name='industry_pct')
    
    # 产生最终的策略信号
    industry_signal_df['strategy_signal'] = (industry_signal_df['industry_pct'] > x_threshold)
    
    return industry_signal_df

def prepare_price_data(price_df_raw: pd.DataFrame, m_months: int) -> pd.DataFrame:
    """
    处理价格数据，并计算未来 M 个月的收益率。
    (逻辑保持不变)
    """
    print(f"Step 5: Preparing price data and calculating future {m_months}M returns...")
    df = price_df_raw.copy()
    
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values(by=['ts_code', 'trade_date'])
    
    m_days = m_months * 21  # 估算交易日
    
    future_price = df.groupby('ts_code')['close'].shift(-m_days)
    df[f'future_return_{m_months}M'] = (future_price / df['close']) - 1
    
    return df

# --- 【函数已删除】 ---
# def map_report_to_trade_date(end_date: pd.Timestamp) -> pd.Timestamp:
#     (此函数不再需要，已删除)

def run_backtest(df_stock_signals: pd.DataFrame, # <-- 【新增】
                 industry_signal_df: pd.DataFrame, 
                 price_df_processed: pd.DataFrame, 
                 m_months: int) -> pd.Series:
    """
    执行回测：将信号与未来收益相匹配，并计算等权组合收益。
    【修改】: 
    1. 使用 ann_date + 1 BDay 作为交易日。
    2. 只买入 'strategy_signal' == True 且 'stock_signal' == True 的股票。
    """
    print("Step 6: Running backtest by matching signals to future returns...")
    
    # --- 1. 修复精准度 (Problem #2) ---
    # 筛选出“好行业”信号
    good_industries = industry_signal_df[
        industry_signal_df['strategy_signal'] == True
    ][['ann_date', 'l3_name']].drop_duplicates()
    
    # 筛选出“好个股”信号
    good_stocks = df_stock_signals[
        df_stock_signals['stock_signal'] == True
    ]
    
    # 【关键】合并：只保留“好行业”中的“好个股”
    trade_list = pd.merge(
        good_stocks,
        good_industries,
        on=['ann_date', 'l3_name'],
        how='inner' # 'inner' merge 确保两者都为 True
    )
    
    trade_list = trade_list[['ts_code', 'l3_name', 'ann_date']].drop_duplicates()

    # --- 2. 修复时效性 (Problem #1) ---
    # 计算真实的交易入场日：公告日 + 1 个交易日
    trade_list['trade_entry_date'] = trade_list['ann_date'] + BDay(1)
    
    # 3. 关键合并：
    # 将我们的“买入清单”与“价格数据”合并
    # 匹配条件：(trade_date == trade_entry_date) AND (ts_code == ts_code)
    results_df = pd.merge(
        price_df_processed, 
        trade_list, 
        left_on=['trade_date', 'ts_code'], 
        right_on=['trade_entry_date', 'ts_code']
    )
    
    # 【重命名】 l3_name_x 是来自 price_df 的行业, l3_name_y 是来自 trade_list 的
    # 它们应该是一样的，我们保留一个即可
    results_df.rename(columns={'l3_name_x': 'l3_name'}, inplace=True)

    # 4. 计算组合收益
    # 现在的 results_df 只包含我们精准筛选的股票
    # .mean() 现在是“好股票”的等权平均收益
    return_col = f'future_return_{m_months}M'
    portfolio_returns = results_df.groupby(
        ['trade_entry_date', 'l3_name']
    )[return_col].mean()
    
    print("Backtest complete.")
    return portfolio_returns

def analyze_results(portfolio_returns: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析并展示回测结果。
    【修改】: 增加了 std_dev 和 sharpe_ratio
    """
    
    # --- 1. 从 config 中提取参数 ---
    try:
        m_months = config["PARAM_M_HOLDING_MONTHS"]
        y = config["PARAM_Y_GROWTH_THRESHOLD"]
        z = config["PARAM_Z_CONSECUTIVE_QUARTERS"]
        x = config["PARAM_X_INDUSTRY_THRESHOLD"]
        logic = config.get("PARAM_Y_GROWTH_LOGIC", "NA")
        RESULTS_DIR = config.get("RESULTS_DIR", "results")
    except KeyError as e:
        print(f"!! 错误: analyze_results 无法在 config 中找到参数: {e}")
        return {
            "m_months": np.nan, "avg_return": np.nan, "median_return": np.nan, 
            "win_rate": np.nan, "num_signals": 0, "std_dev": np.nan, "sharpe_ratio": np.nan
        }

    if portfolio_returns.empty:
        print("--- 策略未产生任何交易信号，无法分析。 ---")
        return {
            "m_months": m_months, "avg_return": np.nan, "median_return": np.nan, 
            "win_rate": np.nan, "num_signals": 0, "std_dev": np.nan, "sharpe_ratio": np.nan
        }

    print("\n--- 策略表现分析 ---")
    print(f"策略在 {m_months} 个月持有期内的表现（等权组合）：")

    avg_return = portfolio_returns.mean()
    print(f"\n平均收益率: {avg_return:.2%}")

    median_return = portfolio_returns.median()
    print(f"收益中位数: {median_return:.2%}")

    win_rate = (portfolio_returns > 0).mean()
    print(f"胜率 (收益 > 0%): {win_rate:.2%}")
    
    num_signals = len(portfolio_returns)
    print(f"信号总数 (行业-季度): {num_signals}")
    
    # --- 【新增】计算夏普比率 ---
    std_dev = portfolio_returns.std()
    annualization_factor = 12 / m_months
    sharpe_ratio = (avg_return / std_dev) * np.sqrt(annualization_factor) if std_dev > 0 else np.nan
    
    print(f"收益标准差: {std_dev:.2%}")
    print(f"年化夏普比率: {sharpe_ratio:.2f}")
    # --- 结束新增 ---

    # --- 2. 绘制并保存图表 ---
    plot_title = f"Strategy (Y={y}, Z={z}, X={x}, M={m_months}, Logic={logic}) - {num_signals} Signals"
    plot_filename = os.path.join(
        RESULTS_DIR, 
        f"plot_Y{y}_Z{z}_X{x}_M{m_months}_Logic{logic}_v2.png"
    )
    
    plt.figure(figsize=(10, 6))
    portfolio_returns.hist(bins=30, alpha=0.75, edgecolor='black')
    plt.axvline(avg_return, color='red', linestyle='--', linewidth=2, label=f'平均收益 ({avg_return:.2%})')
    plt.axvline(median_return, color='orange', linestyle=':', linewidth=2, label=f'中位收益 ({median_return:.2%})')
    
    plt.title(plot_title)
    plt.xlabel("收益率")
    plt.ylabel("次数")
    plt.legend()
    plt.grid(False)
    
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True) 
        plt.savefig(plot_filename)
        print(f"图表已保存至: {plot_filename}")
    except Exception as e:
        print(f"!! 错误: 保存图表失败: {e}")
        
    plt.close()

    # --- 3. 返回结果字典 (【修改】增加夏普) ---
    return {
        "m_months": m_months,
        "avg_return": avg_return,
        "median_return": median_return,
        "win_rate": win_rate,
        "num_signals": num_signals,
        "std_dev": std_dev,          # <-- 新增
        "sharpe_ratio": sharpe_ratio # <-- 新增
    }

# --- 4. 主函数 (流程编排) ---

def main(config: Dict[str, Any], fs_df_raw: pd.DataFrame, price_df_raw: pd.DataFrame) -> Dict[str, Any]:
    """
    主函数，用于编排整个回测流程。
    【修改】: 调整了 run_backtest 的入参。
    """
    # 步骤 1: 【已删除】
    
    # 步骤 2: 清理财务数据
    df_cleaned = clean_financial_data(
        fs_df_raw, 
        config["INDUSTRY_LIST"], 
        config["CLEANING_COLS"], 
        config["WINDSORIZE_LOW"], 
        config["WINDSORIZE_HIGH"]
    )
    
    # 步骤 3: 生成个股信号
    df_stock_signals = generate_stock_signals(
        df_cleaned, 
        config["PARAM_Y_GROWTH_THRESHOLD"], 
        config["PARAM_Z_CONSECUTIVE_QUARTERS"],
        config["PARAM_Y_GROWTH_METRICS"],
        config["PARAM_Y_GROWTH_LOGIC"]
    )
    
    # 步骤 4: 生成行业信号
    industry_signal_df = generate_industry_signals(
        df_stock_signals, # 使用个股信号来聚合
        config["PARAM_X_INDUSTRY_THRESHOLD"]
    )
    
    # 步骤 5: 准备价格数据
    price_df_processed = prepare_price_data(
        price_df_raw, 
        config["PARAM_M_HOLDING_MONTHS"]
    )
    
    # 步骤 6: 执行回测
    # 【修改】传入 df_stock_signals 和 industry_signal_df
    portfolio_returns = run_backtest(
        df_stock_signals, 
        industry_signal_df,
        price_df_processed, 
        config["PARAM_M_HOLDING_MONTHS"]
    )
    
    # 步骤 7: 分析结果
    results = analyze_results(
        portfolio_returns, 
        config
    )
    
    return results

# --- 5. 脚本执行入口 ---
if __name__ == "__main__":
    
    print("--- 开始自动化参数扫描 ---")
    
    # --- 1. 定义你的参数网格 ---
    param_grid = {
        "Y_GROWTH": [20.0, 30.0, 40.0, 50.0, 60.0],                
        "Z_QUARTERS": [2],                   
        "X_INDUSTRY_PCT": [0.4, 0.5, 0.6, 0.7],      
        "M_MONTHS": [1,3,6],
        "Y_GROWTH_LOGIC": ['AND', 'OR']
    }
    
    # --- 2. 定义结果文件夹 ---
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True) 
    print(f"所有结果将保存到: {RESULTS_DIR}/")

    # --- 3. 在循环外加载一次数据 ---
    print("--- 正在加载数据 (仅一次)... ---")
    fs_df_raw, price_df_raw = load_data(
        CONFIG["DATA_PATH_FINANCIALS"], 
        CONFIG["DATA_PATH_PRICES"]
    )
    print("--- 数据加载完毕 ---")

    # --- 4. 循环执行 ---
    all_results = []  
    
    for y in param_grid["Y_GROWTH"]:
      for z in param_grid["Z_QUARTERS"]:
        for x in param_grid["X_INDUSTRY_PCT"]:
          for m in param_grid["M_MONTHS"]:
            for logic in param_grid["Y_GROWTH_LOGIC"]:
                    
                print(f"\n--- 正在测试: Y={y}, Z={z}, X={x}, M={m}, Logic={logic} ---")
                
                current_config = CONFIG.copy()
                
                current_config["PARAM_Y_GROWTH_THRESHOLD"] = y
                current_config["PARAM_Z_CONSECUTIVE_QUARTERS"] = z
                current_config["PARAM_X_INDUSTRY_THRESHOLD"] = x
                current_config["PARAM_M_HOLDING_MONTHS"] = m
                current_config["PARAM_Y_GROWTH_LOGIC"] = logic
                current_config["RESULTS_DIR"] = RESULTS_DIR 
                
                try:
                    results = main(current_config, fs_df_raw.copy(), price_df_raw.copy()) # 使用 .copy() 确保原始数据不被修改
                    
                    run_log = {
                        "Y": y, "Z": z, "X": x, "M": m, "Logic": logic, 
                        **results
                    }
                    all_results.append(run_log)
                    
                except Exception as e:
                    print(f"!!! 运行失败: Y={y}, Z={z}, X={x}, M={m}, Logic={logic}. 错误: {e} !!!")
                    # 可以在这里添加 raise e 来停止执行并调试
                    # raise e 

    # --- 5. 汇总并保存结果 ---
    print("\n--- 自动化参数扫描完成 ---")
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="sharpe_ratio", ascending=False) # 【修改】按夏普比率排序
    results_df = results_df.round(3) # 保留3位小数以便查看
    
    print("参数扫描结果汇总:")
    print(results_df)

    # 保存为 CSV
    output_path = os.path.join(RESULTS_DIR, "strategy_param_sweep_results_v2.csv")

    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n汇总结果已保存至: {output_path}")