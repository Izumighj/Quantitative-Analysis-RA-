import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os

# --- 1. Matplotlib 中文显示配置 ---
# (这部分保持不变)
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'STHeiti', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 2. 策略参数配置中心 (CONFIG) ---
# 所有可调参数都集中在这里
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
    "WINDSORIZE_LOW": 0.01,  # 1% 缩尾
    "WINDSORIZE_HIGH": 0.99, # 99% 缩尾
    
    # --- 核心策略参数 (可调) ---
    # 【修改】定义我们关心的增长指标
    "PARAM_Y_GROWTH_METRICS": ['q_profit_yoy', 'q_sales_yoy'], 
    # 【新增】定义组合逻辑: 'AND' (双重增长) 或 'OR' (任意增长)
    "PARAM_Y_GROWTH_LOGIC": 'AND', 
    
    "PARAM_Y_GROWTH_THRESHOLD": 20.0,  # (Y) 增长阈值 (Y%): 比如20%
    "PARAM_Z_CONSECUTIVE_QUARTERS": 2, # (Z) 连续季度 (Z): 比如2个季度
    "PARAM_X_INDUSTRY_THRESHOLD": 0.5,  # (X) 行业阈值 (X%): 比如 50% 的公司
    
    # --- 回测参数 (可调) ---
    "PARAM_M_HOLDING_MONTHS": 3,       # (M) 持有月数 (M)
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
    """
    print("Step 2.1: Filtering industries...")
    df = df_raw[df_raw['l3_name'].isin(ind_list)].copy()

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
                         growth_metrics: List[str], # <-- 新增
                         logic: str                 # <-- 新增
                         ) -> pd.DataFrame:
    """
    根据 Y (增长阈值), Z (连续季度) 和指定的逻辑(AND/OR) 生成个股信号。
    """
    print(f"Step 3: Generating stock signals (Y={y_threshold}%, Z={z_quarters}q, Metrics={growth_metrics}, Logic={logic})...")
    df_sorted = df.sort_values(by=['ts_code', 'end_date'])
    
    # 1. 标记每个指标是否达标
    metric_bool_cols = []
    for metric in growth_metrics:
        col_name = f'is_growth_{metric}'
        df_sorted[col_name] = df_sorted[metric] > y_threshold
        metric_bool_cols.append(col_name)

    # 2. 根据 'AND' 或 'OR' 逻辑，创建最终的 'is_growth' 列
    if logic.upper() == 'AND':
        # np.logical_and.reduce 会对列表中的所有列执行 'AND'
        df_sorted['is_growth'] = np.logical_and.reduce(
            [df_sorted[col] for col in metric_bool_cols]
        )
    elif logic.upper() == 'OR':
        # np.logical_or.reduce 会对列表中的所有列执行 'OR'
        df_sorted['is_growth'] = np.logical_or.reduce(
            [df_sorted[col] for col in metric_bool_cols]
        )
    else:
        # 如果逻辑不是 AND 或 OR, 默认只使用第一个指标
        print(f"Warning: Unknown logic '{logic}'. Defaulting to first metric only.")
        df_sorted['is_growth'] = df_sorted[metric_bool_cols[0]]

    # 3. 计算连续达标季度数 (这部分逻辑不变)
    rolling_sum = df_sorted.groupby('ts_code')['is_growth'].rolling(window=z_quarters).sum()
    df_sorted['consecutive_growth_count'] = rolling_sum.reset_index(level=0, drop=True)
    
    # 4. 创建“个股信号” (这部分逻辑不变)
    df_sorted['stock_signal'] = (df_sorted['consecutive_growth_count'] == z_quarters)
    
    return df_sorted

def generate_industry_signals(df_stock_signals: pd.DataFrame, x_threshold: float) -> pd.DataFrame:
    """
    根据 X (行业阈值) 聚合个股信号，生成行业（策略）信号。
    """
    print(f"Step 4: Generating industry signals (X={x_threshold*100}%)...")
    
    # 计算行业内触发信号的公司比例
    industry_pct = df_stock_signals.groupby(['end_date', 'l3_name'])['stock_signal'].mean()
    industry_signal_df = industry_pct.reset_index(name='industry_pct')
    
    # 产生最终的策略信号
    industry_signal_df['strategy_signal'] = (industry_signal_df['industry_pct'] > x_threshold)
    
    return industry_signal_df

def prepare_price_data(price_df_raw: pd.DataFrame, m_months: int) -> pd.DataFrame:
    """
    处理价格数据，并计算未来 M 个月的收益率。
    """
    print(f"Step 5: Preparing price data and calculating future {m_months}M returns...")
    df = price_df_raw.copy()
    
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values(by=['ts_code', 'trade_date'])
    
    m_days = m_months * 21  # 估算交易日
    
    future_price = df.groupby('ts_code')['close'].shift(-m_days)
    df[f'future_return_{m_months}M'] = (future_price / df['close']) - 1
    
    return df

def map_report_to_trade_date(end_date: pd.Timestamp) -> pd.Timestamp:
    """
    将财报报告期(end_date)映射到信号生效的交易日。
    规则：Q1->5/1, Q2->9/1, Q3->11/1, Q4->次年5/1
    """
    month = end_date.month
    year = end_date.year
    
    if month == 3:   # Q1
        return pd.Timestamp(year=year, month=5, day=1)
    elif month == 6: # Q2
        return pd.Timestamp(year=year, month=9, day=1)
    elif month == 9: # Q3
        return pd.Timestamp(year=year, month=11, day=1)
    elif month == 12: # Q4
        return pd.Timestamp(year=year + 1, month=5, day=1)
    return pd.NaT

def run_backtest(industry_signal_df: pd.DataFrame, 
                 price_df_processed: pd.DataFrame, 
                 m_months: int) -> pd.Series:
    """
    执行回测：将信号与未来收益相匹配，并计算等权组合收益。
    """
    print("Step 6: Running backtest by matching signals to future returns...")
    
    # 1. 确保日期类型
    industry_signal_df['end_date'] = pd.to_datetime(industry_signal_df['end_date'])
    
    # 2. 映射交易入口日期
    industry_signal_df['trade_entry_date'] = industry_signal_df['end_date'].apply(map_report_to_trade_date)
    
    # 3. 筛选出所有触发的信号
    trade_signals = industry_signal_df[
        industry_signal_df['strategy_signal'] == True
    ][['trade_entry_date', 'l3_name']].dropna()

    # 4. 关键合并：将信号（行业, 日期）与价格（行业, 日期, 股票, 未来收益）匹配
    results_df = pd.merge(
        price_df_processed, 
        trade_signals, 
        left_on=['trade_date', 'l3_name'], 
        right_on=['trade_entry_date', 'l3_name']
    )

    # 5. 计算组合收益（按日期和行业分组，对组内所有股票的未来收益取均值）
    return_col = f'future_return_{m_months}M'
    portfolio_returns = results_df.groupby(
        ['trade_entry_date', 'l3_name']
    )[return_col].mean()
    
    print("Backtest complete.")
    return portfolio_returns

def analyze_results(portfolio_returns: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析并展示回测结果。
    【修改】：
    1. 接收完整的 config 字典。
    2. 保存图表到 data/ 目录，并使用参数命名。
    3. 返回包含关键指标的字典。
    """
    
    # --- 1. 从 config 中提取参数 ---
    try:
        m_months = config["PARAM_M_HOLDING_MONTHS"]
        y = config["PARAM_Y_GROWTH_THRESHOLD"]
        z = config["PARAM_Z_CONSECUTIVE_QUARTERS"]
        x = config["PARAM_X_INDUSTRY_THRESHOLD"]
        RESULTS_DIR = config.get("RESULTS_DIR", "results")
    except KeyError as e:
        print(f"!! 错误: analyze_results 无法在 config 中找到参数: {e}")
        # 即使出错，也返回一个空结果
        return {
            "m_months": np.nan, "avg_return": np.nan, "median_return": np.nan, 
            "win_rate": np.nan, "num_signals": 0
        }

    if portfolio_returns.empty:
        print("--- 策略未产生任何交易信号，无法分析。 ---")
        return {
            "m_months": m_months, "avg_return": np.nan, "median_return": np.nan, 
            "win_rate": np.nan, "num_signals": 0
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

    # --- 2. 绘制并保存图表 ---
    
    # ...
    logic = config.get("PARAM_Y_GROWTH_LOGIC", "profit") 
    # ...
    plot_title = f"Strategy (Y={y}, Z={z}, X={x}, M={m_months}, Logic={logic}) - {num_signals} Signals"
    plot_filename = os.path.join(
        RESULTS_DIR, 
        f"plot_Y{y}_Z{z}_X{x}_M{m_months}_Logic{logic}.png"
    )
    # ...
    
    plt.figure(figsize=(10, 6))
    portfolio_returns.hist(bins=30, alpha=0.75, edgecolor='black')
    plt.axvline(avg_return, color='red', linestyle='--', linewidth=2, label=f'平均收益 ({avg_return:.2%})')
    plt.axvline(median_return, color='orange', linestyle=':', linewidth=2, label=f'中位收益 ({median_return:.2%})')
    
    plt.title(plot_title) # 使用包含参数的标题
    plt.xlabel("收益率")
    plt.ylabel("次数")
    plt.legend()
    plt.grid(False)
    
    try:
        # 确保目录存在 (虽然主流程也会创建，但这里多一步保险)
        os.makedirs(RESULTS_DIR, exist_ok=True) 
        plt.savefig(plot_filename) # <--- 现在 plot_filename 是正确的路径了
        print(f"图表已保存至: {plot_filename}")
    except Exception as e:
        print(f"!! 错误: 保存图表失败: {e}")
        
    plt.close() # 关闭图形，防止在循环中占用过多内存

    # --- 3. 返回结果字典 ---
    return {
        "m_months": m_months,
        "avg_return": avg_return,
        "median_return": median_return,
        "win_rate": win_rate,
        "num_signals": num_signals
    }

# --- 4. 主函数 (流程编排) ---

def main(config: Dict[str, Any], fs_df_raw: pd.DataFrame, price_df_raw: pd.DataFrame) -> Dict[str, Any]:
    """
    主函数，用于编排整个回测流程。
    【修改】: 接收 fs_df_raw 和 price_df_raw 作为参数，不再自己加载。
    """
    # 步骤 1: 【已删除】数据加载在主流程中完成。
    
    # 步骤 2: 清理财务数据
    # 【修改】使用传入的 fs_df_raw
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
        df_stock_signals, 
        config["PARAM_X_INDUSTRY_THRESHOLD"]
    )
    
    # 步骤 5: 准备价格数据
    # 【修改】使用传入的 price_df_raw
    price_df_processed = prepare_price_data(
        price_df_raw, 
        config["PARAM_M_HOLDING_MONTHS"]
    )
    
    # 步骤 6: 执行回测
    portfolio_returns = run_backtest(
        industry_signal_df, 
        price_df_processed, 
        config["PARAM_M_HOLDING_MONTHS"]
    )
    
    # 步骤 7: 分析结果
    results = analyze_results(
        portfolio_returns, 
        config
    )
    
    # 【关键】将结果字典返回给调用者
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
        "Y_GROWTH_LOGIC": ['AND', 'OR']  # <-- 【新增】测试两种逻辑
    }
    
    # --- 2. 定义结果文件夹 ---
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True) 
    print(f"所有结果将保存到: {RESULTS_DIR}/")

    # --- 3. 【优化】在循环外加载一次数据 ---
    print("--- 正在加载数据 (仅一次)... ---")
    fs_df_raw, price_df_raw = load_data(
        CONFIG["DATA_PATH_FINANCIALS"], 
        CONFIG["DATA_PATH_PRICES"]
    )
    print("--- 数据加载完毕 ---")

    # --- 3. 循环执行 ---
    all_results = []  
    
    # 【新增】嵌套循环
    for y in param_grid["Y_GROWTH"]:
      for z in param_grid["Z_QUARTERS"]:
        for x in param_grid["X_INDUSTRY_PCT"]:
          for m in param_grid["M_MONTHS"]:
            for logic in param_grid["Y_GROWTH_LOGIC"]: # <-- 新增循环
                    
                print(f"\n--- 正在测试: Y={y}, Z={z}, X={x}, M={m}, Logic={logic} ---")
                
                current_config = CONFIG.copy()
                
                # 更新所有循环中的参数
                current_config["PARAM_Y_GROWTH_THRESHOLD"] = y
                current_config["PARAM_Z_CONSECUTIVE_QUARTERS"] = z
                current_config["PARAM_X_INDUSTRY_THRESHOLD"] = x
                current_config["PARAM_M_HOLDING_MONTHS"] = m
                current_config["PARAM_Y_GROWTH_LOGIC"] = logic # <-- 新增
                current_config["RESULTS_DIR"] = RESULTS_DIR 
                
                try:
                    results = main(current_config,fs_df_raw, price_df_raw)
                    # 【修改】将 Logic 也记录到 log 中
                    run_log = {
                        "Y": y, "Z": z, "X": x, "M": m, "Logic": logic, 
                        **results
                    }
                    all_results.append(run_log)
                    
                except Exception as e:
                    print(f"!!! 运行失败: Y={y}, Z={z}, X={x}, M={m}, Logic={logic}. 错误: {e} !!!")

    # --- 4. 汇总并保存结果 ---
    print("\n--- 自动化参数扫描完成 ---")
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="avg_return", ascending=False)
    results_df = results_df.round(2)
    
    print("参数扫描结果汇总:")
    print(results_df)
    
    output_path = os.path.join(RESULTS_DIR, "strategy_param_sweep_results.csv")
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n汇总结果已保存至: {output_path}")