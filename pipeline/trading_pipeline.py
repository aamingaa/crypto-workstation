"""
交易分析管道主控制器
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from core.base import ConfigManager
from data.trades_processor import TradesProcessor
from data.dollar_bars import DollarBarBuilder
from data.time_bars import TimeBarBuilder
# from features import MicrostructureFeatureExtractor
from features.microstructure_extractor import MicrostructureFeatureExtractor
from ml.models import ModelFactory
from ml.validators import PurgedBarValidator
# from utils.visualization import TradingVisualizer


class TradingPipeline:
    """交易分析管道主控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config_manager = ConfigManager()
        if config:
            for module, module_config in config.items():
                self.config_manager.set_config(module, module_config)
        
        # 设置默认标签配置
        default_label_config = {
            'horizon_bars': [1, 3, 5, 10]  # 默认预测持有期（以bar为单位）
        }
        if 'labels' not in self.config_manager._configs:
            self.config_manager.set_config('labels', default_label_config)
        
        # 设置默认数据配置
        default_data_config = {
            'load_mode': 'auto',  # 'daily', 'monthly', 'auto'
            'prefer_feather': True
        }
        if 'data' not in self.config_manager._configs:
            self.config_manager.set_config('data', default_data_config)
        
        # 初始化组件
        self.trades_processor = TradesProcessor()
        # 设置默认特征配置（开启 bucketed_flow）
        default_features_config = {
            'bucketed_flow': {
                'enabled': True,
                'low_q': 0.2,
                'high_q': 0.8,
                'lag': 1,
                'vpin_bins': 10,
                'min_trades_alpha': 50
            },
            # 其余保留原有默认
        }
        if 'features' not in self.config_manager._configs:
            self.config_manager.set_config('features', default_features_config)
        else:
            # 将默认项与外部传入合并（仅补充缺失键）
            feat_cfg = self.config_manager.get_config('features') or {}
            for k, v in default_features_config.items():
                if k not in feat_cfg:
                    feat_cfg[k] = v
            self.config_manager.set_config('features', feat_cfg)

        self.feature_extractor = MicrostructureFeatureExtractor(
            self.config_manager.get_config('features')
        )
        # self.visualizer = TradingVisualizer()
        
        # 运行时数据
        self.trades_context = None
        self.bars = None
        self.features = None
        self.labels = None
        self.model = None
        self.evaluation_results = None
    
    def load_data(self, trades_data: Optional[pd.DataFrame] = None,
                  date_range: Optional[Tuple[str, str]] = None,
                  daily_data_template: Optional[str] = None,
                  monthly_data_template: Optional[str] = None) -> pd.DataFrame:
        """加载交易数据（支持按日/按月/自动模式）
        
        参数:
            trades_data: 直接提供的DataFrame
            date_range: 日期范围 (start_date, end_date)
            daily_data_template: 日数据路径模板，支持 {date} 和 {ext}
            monthly_data_template: 月数据路径模板，支持 {month} 和 {ext}
        
        加载逻辑:
            1. 如果提供 trades_data，直接返回
            2. 根据配置的 load_mode 决定加载方式：
               - 'monthly': 优先使用月度文件
               - 'daily': 使用日数据文件
               - 'auto': 自动选择最优方案（默认）
        """
        if trades_data is not None:
            return trades_data
        
        if not (date_range and (daily_data_template or monthly_data_template)):
            raise ValueError("必须提供 date_range 和至少一个数据模板")
        
        # 获取数据加载配置
        data_config = self.config_manager.get_config('data')
        load_mode = data_config.get('load_mode', 'auto')
        prefer_feather = data_config.get('prefer_feather', True)
        
        start_date, end_date = date_range
        
        # 根据 load_mode 决定加载策略
        if load_mode == 'monthly' and monthly_data_template:
            return self._load_monthly_data(start_date, end_date, monthly_data_template, prefer_feather)
        elif load_mode == 'daily' and daily_data_template:
            return self._load_daily_data(start_date, end_date, daily_data_template, prefer_feather)
        elif load_mode == 'auto':
            # 自动选择：优先尝试月度数据，如果不存在则使用日数据
            return self._load_auto_data(start_date, end_date, daily_data_template, monthly_data_template, prefer_feather)
        else:
            raise ValueError(f"不支持的 load_mode: {load_mode} 或缺少对应的数据模板")
    
    def _load_monthly_data(self, start_date: str, end_date: str, 
                          monthly_template: str, prefer_feather: bool = True) -> pd.DataFrame:
        """按月加载数据"""
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 生成月份列表
        months = []
        current = start.replace(day=1)
        while current <= end:
            months.append(current.strftime('%Y-%m'))
            current += relativedelta(months=1)
        
        print(f"📂 按月加载数据: {start_date} 到 {end_date} ({len(months)} 个月)")
        raw_df = []
        
        exts = ['feather', 'zip'] if prefer_feather else ['zip', 'feather']
        
        for month in months:
            loaded = False
            for ext in exts:
                file_path = monthly_template.format(month=month, ext=ext)
                if os.path.exists(file_path):
                    if ext == 'feather':
                        df = pd.read_feather(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    raw_df.append(df)
                    print(f"   ✓ 加载 {month} ({ext})")
                    loaded = True
                    break
            
            if not loaded:
                print(f"   ⚠️  警告: 月度文件不存在 {month}")
        
        if not raw_df:
            raise ValueError("未找到任何月度数据文件")
        
        print(f"✅ 加载了 {len(raw_df)} 个月度文件，开始合并...")
        trades_df = pd.concat(raw_df, ignore_index=True, copy=False)
        
        # 过滤到指定日期范围
        # trades_df = self._filter_date_range(trades_df, start_date, end_date)
        # print(f"✅ 合并完成，过滤后共 {len(trades_df):,} 条记录")
        
        return trades_df
    
    def _load_daily_data(self, start_date: str, end_date: str,
                        daily_template: str, prefer_feather: bool = True) -> pd.DataFrame:
        """按日加载数据"""
        date_list = self._generate_date_range(start_date, end_date)
        
        print(f"📂 按日加载数据: {start_date} 到 {end_date} ({len(date_list)} 天)")
        raw_df = []
        
        exts = ['feather', 'zip'] if prefer_feather else ['zip', 'feather']
        
        for i, date in enumerate(date_list, 1):
            loaded = False
            for ext in exts:
                file_path = daily_template.format(date=date, ext=ext)
                if os.path.exists(file_path):
                    if ext == 'feather':
                        df = pd.read_feather(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    raw_df.append(df)
                    loaded = True
                    break
            
            if not loaded:
                print(f"   ⚠️  警告: 日数据文件不存在 {date}")
            elif i % 10 == 0:
                print(f"   已加载 {i}/{len(date_list)} 个文件...")
        
        if not raw_df:
            raise ValueError("未找到任何日数据文件")
        
        print(f"✅ 加载了 {len(raw_df)} 个日文件，开始合并...")
        trades_df = pd.concat(raw_df, ignore_index=True, copy=False)
        print(f"✅ 合并完成，共 {len(trades_df):,} 条记录")
        
        return trades_df
    
    def _load_auto_data(self, start_date: str, end_date: str,
                       daily_template: Optional[str], monthly_template: Optional[str],
                       prefer_feather: bool = True) -> pd.DataFrame:
        """自动选择最优加载方案"""
        from datetime import datetime
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days_span = (end - start).days + 1
        
        # 如果跨度超过 10 天且有月度文件可用，优先尝试月度
        if days_span > 10 and monthly_template:
            print(f"🔍 自动模式：数据跨度 {days_span} 天，尝试使用月度数据...")
            try:
                return self._load_monthly_data(start_date, end_date, monthly_template, prefer_feather)
            except (ValueError, FileNotFoundError) as e:
                print(f"   月度数据加载失败，回退到日数据: {e}")
        
        # 回退到日数据
        if daily_template:
            print(f"🔍 自动模式：使用日数据加载...")
            return self._load_daily_data(start_date, end_date, daily_template, prefer_feather)
        else:
            raise ValueError("无可用的数据模板")
    
    def _filter_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """过滤数据到指定日期范围"""
        # 确保 time 列是 datetime 类型
        if 'time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
            
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            
            mask = (df['time'] >= start) & (df['time'] <= end)
            return df[mask].reset_index(drop=True)
        
        return df
    
    def build_bars(self,
                   trades_df: pd.DataFrame,
                   dollar_threshold: float,
                   bar_cache_template: Optional[str] = None,
                   bar_zip_path: Optional[str] = None,  # 向后兼容
                   bar_type: str = 'dollar',
                   time_freq: Optional[str] = None) -> pd.DataFrame:
        """构建 Bars（支持 Dollar 和 Time）

        参数:
            trades_df: 交易数据
            dollar_threshold: dollar bar 阈值
            bar_cache_template: 缓存路径模板，支持 {ext}，如 'path/bars.{ext}'
            bar_zip_path: (已废弃) 直接指定zip路径，仅为向后兼容保留
            bar_type: 'dollar' 或 'time'
            time_freq: 当 bar_type='time' 时的频率字符串（如 '1H', '15min'）
        
        加载优先级:
            1. feather 缓存（最快）
            2. zip 缓存
            3. 构建新的bars并缓存
        """
        # 向后兼容：如果只提供了 bar_zip_path，转换为 template
        if bar_zip_path and not bar_cache_template:
            bar_cache_template = bar_zip_path.replace('.zip', '.{ext}')
        
        # 优先从 feather 缓存加载
        if bar_cache_template:
            feather_path = bar_cache_template.format(ext='feather')
            if os.path.exists(feather_path):
                print(f"✓ 从 Feather 缓存加载 bars: {feather_path}")
                bars = pd.read_feather(feather_path)
                self.bars = bars
                return bars
            
            # 其次从 zip 缓存加载
            zip_path = bar_cache_template.format(ext='zip')
            if os.path.exists(zip_path):
                print(f"✓ 从 ZIP 缓存加载 bars: {zip_path}")
                bars = pd.read_csv(zip_path)
                
                # 转存为 feather 格式
                print(f"→ 转存为 Feather 格式: {feather_path}")
                os.makedirs(os.path.dirname(feather_path), exist_ok=True)
                bars.to_feather(feather_path)
                
                self.bars = bars
                return bars
        
        # 构建新的 bars
        if bar_type == 'time':
            freq = time_freq or '1H'
            print(f"构建 Time Bars（freq={freq}）...")
            bar_builder = TimeBarBuilder(freq=freq)
        else:
            print(f"构建 Dollar Bars（threshold={dollar_threshold:,.0f}）...")
            bar_builder = DollarBarBuilder(dollar_threshold)

        bars = bar_builder.process(trades_df)
        print(f"✓ 构建了 {len(bars)} 个 bars")
            
        # 缓存已处理的 trades 上下文（避免在 extract_features 中重复处理）
        self.trades_context = bar_builder.trades_processor.context
        

        # 保存缓存（同时保存 feather 和 zip）
        if bar_cache_template:
            feather_path = bar_cache_template.format(ext='feather')
            zip_path = bar_cache_template.format(ext='zip')
            
            print(f"→ 保存 Feather 缓存: {feather_path}")
            os.makedirs(os.path.dirname(feather_path), exist_ok=True)
            bars.to_feather(feather_path)
            
            print(f"→ 保存 ZIP 缓存: {zip_path}")
            os.makedirs(os.path.dirname(zip_path), exist_ok=True)
            bars.to_csv(
                zip_path, 
                index=False,
                compression={'method': 'zip', 'archive_name': 'bars.csv'}
            )
        
        self.bars = bars
        return bars
    
    def extract_features(self, trades_df: pd.DataFrame, bars: pd.DataFrame,
                        feature_window_bars: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """提取特征和标签"""
        # 检查是否已经有缓存的 trades_context（来自 build_bars）
        if not hasattr(self, 'trades_context') or self.trades_context is None:
            print("构建交易上下文...")
            self.trades_context = self.trades_processor.build_context(trades_df)
        else:
            print("✓ 使用已缓存的交易上下文（避免重复处理）")
        
        print("提取特征...")
        features_list = []
        labels_list = []
        
        bars = bars.reset_index(drop=True)
        bars['bar_id'] = bars.index
        bars['start_time'] = pd.to_datetime(bars['start_time'])
        bars['end_time'] = pd.to_datetime(bars['end_time'])
        
        close_prices = bars.set_index('bar_id')['close']
        end_times = bars.set_index('bar_id')['end_time']
        
        # 计算标签（多个持有期）
        label_config = self.config_manager.get_config('labels')
        horizon_bars = label_config.get('horizon_bars', [1, 3, 5, 10])
        labels_df = pd.DataFrame(index=close_prices.index)
        
        for horizon in horizon_bars:
            log_return = np.log(close_prices.shift(-horizon) / close_prices)
            labels_df[f'log_return_{horizon}'] = log_return
            labels_df[f't0_time_{horizon}'] = end_times
            labels_df[f'tH_time_{horizon}'] = end_times.shift(-horizon)
        
        # 提取特征
        idx = 1
        for bar_id in close_prices.index:
            bar_window_start_idx = bar_id - feature_window_bars
            if bar_window_start_idx < 0:
                continue
            
            bar_window_end_idx = bar_id - 1
            
            feature_start_ts = bars.loc[bar_window_start_idx, 'start_time']
            feature_end_ts = bars.loc[bar_window_end_idx, 'end_time']
            
            # 提取微观结构特征
            features = self.feature_extractor.extract_from_context(
                ctx=self.trades_context,
                start_ts=feature_start_ts,
                end_ts=feature_end_ts,
                bars=bars,
                bar_window_start_idx=bar_window_start_idx,
                bar_window_end_idx=bar_window_end_idx
            )
            
            if idx % 100 == 0:
                print(f"处理进度: {idx}/{len(close_prices.index)}")
            idx += 1
            
            features['bar_id'] = bar_id
            features['feature_start'] = feature_start_ts
            features['feature_end'] = feature_end_ts
            features['prediction_time'] = bars.loc[bar_id, 'end_time']
            
            features_list.append(features)
        
        # 构建特征DataFrame
        X = pd.DataFrame(features_list).set_index('bar_id')
        y = labels_df.loc[X.index]
        
        # 过滤无效数据（分步处理，确保 X 和 y 严格对齐）
        # 步骤 1: 过滤 y 中所有 log_return 列的无效值
        log_return_cols = [col for col in y.columns if col.startswith('log_return_')]
        y_mask = pd.Series(True, index=y.index)
        for col in log_return_cols:
            y_mask &= y[col].notna() & np.isfinite(y[col].values)
        
        X = X.loc[y_mask]
        y = y.loc[y_mask]
        
        # 步骤 2: 替换 X 中的 inf 为 nan
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # 步骤 3: 删除 X 中包含 NaN 的行，并同步删除 y 中对应的行
        x_valid_mask = ~X.isna().any(axis=1)
        X = X.loc[x_valid_mask]
        y = y.loc[x_valid_mask]
        
        self.features = X
        self.labels = y
        
        return X, y
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.DataFrame,
                          model_type: str = 'ridge',
                          target_horizon: int = 5,
                          n_splits: int = 5,
                          embargo_bars: int = 3) -> Dict:
        """训练模型并评估"""
        print(f"开始训练{model_type}模型...")
        
        # 选择目标标签
        target_col = f'log_return_{target_horizon}'
        if target_col not in y.columns:
            raise ValueError(f"标签列{target_col}不存在")
        
        y_target = y[target_col]
        
        # 过滤特征列（排除时间相关列）
        feature_cols = [col for col in X.columns 
                       if not any(time_word in col.lower() 
                                for time_word in ['time', 'start', 'end', 'settle'])]
        X_features = X[feature_cols]
        
        # 创建模型
        model_config = self.config_manager.get_config('model')
        self.model = ModelFactory.create_model(model_type, **model_config)
        
        # 交叉验证
        validator = PurgedBarValidator(n_splits=n_splits, embargo_bars=embargo_bars)
        results = validator.evaluate(self.model, X_features, y_target)
        
        self.evaluation_results = results
        
        print("模型评估完成!")
        print(f"平均Pearson IC: {results['summary']['pearson_ic_mean']:.4f}")
        print(f"平均Spearman IC: {results['summary']['spearman_ic_mean']:.4f}")
        print(f"平均RMSE: {results['summary']['rmse_mean']:.4f}")
        print(f"平均方向准确率: {results['summary']['dir_acc_mean']:.4f}")
        
        return results
    
    def visualize_results(self, save_dir: Optional[str] = None):
        """可视化结果"""
        if self.evaluation_results is None:
            print("尚未进行模型评估")
            return
        
        predictions = self.evaluation_results['predictions']
        target_col = f'log_return_5'  # 假设使用5期作为目标
        y_true = self.labels[target_col].loc[predictions.index]
        
        # self.visualizer.plot_predictions_vs_truth(
        #     predictions, y_true, 
        #     title="预测值 vs 真实值",
        #     save_path=os.path.join(save_dir, "predictions_vs_truth.png") if save_dir else None
        # )
        
        # 绘制特征重要性
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            # self.visualizer.plot_feature_importance(
            #     importance,
            #     title="特征重要性",
            #     save_path=os.path.join(save_dir, "feature_importance.png") if save_dir else None
            # )
    
    def run_full_pipeline(self, **kwargs) -> Dict:
        """运行完整的分析管道"""
        # 允许通过 kwargs.features_config 覆盖特征配置
        features_override = kwargs.get('features_config')
        if features_override is not None:
            feat_cfg = self.config_manager.get_config('features') or {}
            # 浅合并：以外部为准
            for k, v in features_override.items():
                feat_cfg[k] = v
            self.config_manager.set_config('features', feat_cfg)
            # 重建特征提取器
            self.feature_extractor = MicrostructureFeatureExtractor(feat_cfg)
        # 加载数据
        trades_df = self.load_data(**kwargs.get('data_config', {}))
        print(f"加载了{len(trades_df)}条交易记录")
        
        # 构建bars
        dollar_threshold = kwargs.get('dollar_threshold', 60000000)
        bar_type = kwargs.get('bar_type', 'dollar')
        time_freq = kwargs.get('time_freq')
        bars = self.build_bars(
            trades_df,
            dollar_threshold,
            bar_cache_template=kwargs.get('bar_cache_template'),
            bar_zip_path=kwargs.get('bar_zip_path'),  # 向后兼容
            bar_type=bar_type,
            time_freq=time_freq,
        )
        
        # 提取特征
        feature_window_bars = kwargs.get('feature_window_bars', 10)
        X, y = self.extract_features(trades_df, bars, feature_window_bars)
        print(f"提取了{len(X)}个样本，{len(X.columns)}个特征")
        
        # 训练评估
        results = self.train_and_evaluate(
            X, y,
            model_type=kwargs.get('model_type', 'ridge'),
            target_horizon=kwargs.get('target_horizon', 5),
            n_splits=kwargs.get('n_splits', 5),
            embargo_bars=kwargs.get('embargo_bars', 3)
        )
        
        # 可视化
        if kwargs.get('save_plots'):
            self.visualize_results(kwargs.get('plot_save_dir'))
        
        return {
            'evaluation': results,
            'features': X,
            'labels': y,
            'bars': bars,
            'model': self.model
        }
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """生成日期范围"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_list = []
        current = start
        while current <= end:
            date_list.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        return date_list
