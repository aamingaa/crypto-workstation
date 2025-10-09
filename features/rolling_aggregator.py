"""
Bar 级特征滚动统计提取器
对已聚合的 bar 级特征进行时间序列统计，捕获动态变化模式
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from scipy.stats import linregress


class RollingAggregator:
    """Bar 级特征的滚动统计提取器"""
    
    def __init__(self, feature_extractor=None, windows: Optional[List[int]] = None):
        """
        参数:
            feature_extractor: MicrostructureFeatureExtractor 实例（复用现有特征提取器）
            windows: 滚动窗口列表（以 bar 为单位）
                    None 则使用单一窗口（由 extract 方法指定）
        """
        self.feature_extractor = feature_extractor
        self.windows = windows
    
    def extract_bar_level_features(self, bars: pd.DataFrame, 
                                   ctx, 
                                   bar_idx: int) -> Dict[str, float]:
        """提取单个 bar 的微观结构特征（复用现有特征提取器）
        
        参数:
            bars: 所有 bars 的 DataFrame
            ctx: TradesContext
            bar_idx: 当前 bar 的索引
        
        返回:
            该 bar 的特征字典
        """
        if bar_idx >= len(bars):
            return {}
        
        bar = bars.iloc[bar_idx]
        
        # 获取该 bar 的时间范围
        start_ts = pd.to_datetime(bar['start_time'])
        end_ts = pd.to_datetime(bar['end_time'])
        
        # 使用现有的特征提取器提取该 bar 的特征
        if self.feature_extractor is not None:
            features = self.feature_extractor.extract_from_context(
                ctx=ctx,
                start_ts=start_ts,
                end_ts=end_ts,
                bars=bars,
                bar_window_start_idx=bar_idx,
                bar_window_end_idx=bar_idx
            )
            
            # 给特征加上 'bar_' 前缀以区分单 bar 特征和窗口特征
            bar_features = {f'bar_{k}': v for k, v in features.items()}
            return bar_features
        else:
            # 如果没有提供特征提取器，使用简化版本（向后兼容）
            return self._extract_bar_level_features_simple(bars, ctx, bar_idx)
    
    def _extract_bar_level_features_simple(self, bars: pd.DataFrame, 
                                          ctx, 
                                          bar_idx: int) -> Dict[str, float]:
        """简化版 bar 级特征提取（向后兼容）"""
        bar = bars.iloc[bar_idx]
        s = int(bar['start_trade_idx'])
        e = int(bar['end_trade_idx']) + 1
        
        if e - s <= 0:
            return {}
        
        features = {}
        
        # 1. 波动率特征
        rv = self._sum_range(ctx.csum_ret2, s, e)
        bpv = self._sum_range(ctx.csum_bpv, s, e)
        features['bar_rv'] = rv
        features['bar_bpv'] = bpv
        features['bar_jump'] = max(rv - bpv, 0.0) if (np.isfinite(rv) and np.isfinite(bpv)) else 0.0
        
        # 2. 订单流特征（分桶）
        dv = ctx.quote[s:e]
        sign = ctx.sign[s:e]
        
        if len(dv) >= 5:
            low_q = np.quantile(dv, 0.2)
            high_q = np.quantile(dv, 0.8)
            
            small_mask = dv <= low_q
            large_mask = dv >= high_q
            
            features['bar_small_signed_dollar'] = float(np.sum(sign[small_mask] * dv[small_mask]))
            features['bar_large_signed_dollar'] = float(np.sum(sign[large_mask] * dv[large_mask]))
            features['bar_small_count'] = int(np.sum(small_mask))
            features['bar_large_count'] = int(np.sum(large_mask))
            
            # VPIN 近似
            features['bar_vpin'] = self._calculate_vpin(dv, sign, bins=10)
        else:
            features['bar_small_signed_dollar'] = 0.0
            features['bar_large_signed_dollar'] = 0.0
            features['bar_small_count'] = 0
            features['bar_large_count'] = 0
            features['bar_vpin'] = 0.0
        
        # 3. 价格路径特征
        if e - s >= 2:
            prices = ctx.price[s:e]
            features['bar_hl_ratio'] = float((np.max(prices) - np.min(prices)) / (np.mean(prices) + 1e-12))
        else:
            features['bar_hl_ratio'] = 0.0
        
        # 4. 成交量特征
        features['bar_signed_volume'] = self._sum_range(ctx.csum_signed_qty, s, e)
        features['bar_volume'] = self._sum_range(ctx.csum_qty, s, e)
        
        return features
    
    def _get_statistics_for_feature(self, feature_name: str) -> List[str]:
        """根据特征类型返回需要计算的统计量列表
        
        参数:
            feature_name: 特征名称
        
        返回:
            统计量名称列表
        """
        feat_lower = feature_name.lower()
        
        # ⭐⭐⭐⭐⭐ Tier 1: 核心波动率特征（最重要，全套统计）
        if any(kw in feat_lower for kw in ['rv', 'bpv', 'jump', 'volatility']):
            return ['mean', 'std', 'min', 'max', 'trend', 'slope', 
                   'momentum', 'zscore', 'quantile', 'acceleration', 'autocorr']
        
        # ⭐⭐⭐⭐⭐ Tier 1: 核心订单流特征（VPIN，关键指标）
        elif 'vpin' in feat_lower:
            return ['mean', 'std', 'trend', 'slope', 'zscore', 
                   'momentum', 'quantile', 'acceleration']
        
        # ⭐⭐⭐⭐ Tier 2: 订单流金额特征（趋势+位置）
        elif any(kw in feat_lower for kw in ['signed_dollar', 'signed_quote', 
                                               'buy_dollar', 'sell_dollar']):
            return ['mean', 'std', 'trend', 'zscore', 'momentum', 'quantile']
        
        # ⭐⭐⭐⭐ Tier 2: 冲击/流动性特征
        elif any(kw in feat_lower for kw in ['impact', 'lambda', 'amihud', 
                                               'kyle', 'hasbrouck']):
            return ['mean', 'std', 'trend', 'zscore', 'quantile']
        
        # ⭐⭐⭐ Tier 3: 动量/反转特征
        elif any(kw in feat_lower for kw in ['momentum', 'reversion', 
                                               'dp_short', 'dp_zscore']):
            return ['mean', 'trend', 'zscore', 'momentum']
        
        # ⭐⭐⭐ Tier 3: 价格路径特征
        elif any(kw in feat_lower for kw in ['hl_ratio', 'amplitude', 
                                               'comovement', 'vwap_deviation']):
            return ['mean', 'std', 'trend', 'zscore']
        
        # ⭐⭐ Tier 4: 计数/强度特征（基础统计）
        elif any(kw in feat_lower for kw in ['count', 'intensity', 'arrival', 
                                               'trades', 'duration']):
            return ['mean', 'std', 'trend']
        
        # ⭐⭐ Tier 4: 成交量特征
        elif any(kw in feat_lower for kw in ['volume', 'qty']):
            return ['mean', 'std', 'trend']
        
        # ⭐ Tier 5: 其他特征（最小统计集）
        else:
            return ['mean', 'trend', 'zscore']
    
    def extract_rolling_statistics(self, bar_features: pd.DataFrame, 
                                   window: int,
                                   current_idx: int) -> Dict[str, float]:
        """对 bar 级特征序列进行滚动统计（智能选择统计量）
        
        参数:
            bar_features: 包含 bar 级特征的 DataFrame
            window: 滚动窗口大小（bar 数量）
            current_idx: 当前预测的 bar 索引
        
        返回:
            滚动统计特征字典
        """
        if current_idx < window:
            return {}
        
        window_start = current_idx - window
        window_end = current_idx
        
        # 提取窗口数据
        window_data = bar_features.iloc[window_start:window_end]
        
        if len(window_data) < window:
            return {}
        
        features = {}
        
        # 🔥 自动识别所有 bar_ 开头的数值特征（而不是硬编码列表）
        stat_features = [col for col in window_data.columns 
                        if col.startswith('bar_') and col != 'bar_id'
                        and pd.api.types.is_numeric_dtype(window_data[col])]
        
        if not stat_features:
            # 如果没有 bar_ 前缀的特征，尝试使用所有数值列
            stat_features = [col for col in window_data.columns 
                           if pd.api.types.is_numeric_dtype(window_data[col])]
        
        # 对每个特征进行滚动统计（根据类型智能选择统计量）
        for feat in stat_features:
            if feat not in window_data.columns:
                continue
            
            series = window_data[feat].values
            
            if len(series) < 2:
                continue
            
            # 跳过全为零或无效的序列
            if np.all(series == 0) or not np.any(np.isfinite(series)):
                continue
            
            prefix = f'{feat}_w{window}'
            
            # 🔥 根据特征类型智能选择统计量
            selected_stats = self._get_statistics_for_feature(feat)
            
            # 计算选中的统计量
            # 1. 基础统计量
            if 'mean' in selected_stats:
                features[f'{prefix}_mean'] = float(np.mean(series))
            if 'std' in selected_stats:
                features[f'{prefix}_std'] = float(np.std(series))
            if 'min' in selected_stats:
                features[f'{prefix}_min'] = float(np.min(series))
            if 'max' in selected_stats:
                features[f'{prefix}_max'] = float(np.max(series))
            
            # 2. 趋势特征
            if 'trend' in selected_stats:
                if series[0] != 0 and np.isfinite(series[0]):
                    features[f'{prefix}_trend'] = float((series[-1] - series[0]) / (np.abs(series[0]) + 1e-12))
                else:
                    features[f'{prefix}_trend'] = 0.0
            
            if 'momentum' in selected_stats:
                # 动量（后1/4 vs 前1/4）
                q = window // 4
                if q >= 1:
                    features[f'{prefix}_momentum'] = float(np.mean(series[-q:]) - np.mean(series[:q]))
            
            if 'slope' in selected_stats:
                # 线性回归斜率
                if len(series) >= 3:
                    try:
                        x = np.arange(len(series))
                        slope, _, _, _, _ = linregress(x, series)
                        features[f'{prefix}_slope'] = float(slope)
                    except:
                        pass  # 跳过计算失败的情况
            
            # 3. 相对位置特征
            if 'zscore' in selected_stats or 'quantile' in selected_stats:
                current_val = series[-1]
                mean_val = np.mean(series)
                std_val = np.std(series)
                
                if 'zscore' in selected_stats:
                    if std_val > 0 and np.isfinite(std_val):
                        features[f'{prefix}_zscore'] = float((current_val - mean_val) / std_val)
                    else:
                        features[f'{prefix}_zscore'] = 0.0
                
                if 'quantile' in selected_stats and len(series) >= 5:
                    # 分位数位置
                    rank = np.sum(series < current_val)
                    features[f'{prefix}_quantile'] = float(rank / len(series))
            
            # 4. 波动特征
            if 'range_norm' in selected_stats:
                range_val = np.max(series) - np.min(series)
                mean_val = np.mean(series)
                if mean_val != 0:
                    features[f'{prefix}_range_norm'] = float(range_val / (np.abs(mean_val) + 1e-12))
                else:
                    features[f'{prefix}_range_norm'] = 0.0
            
            # 5. 加速度（二阶趋势）
            if 'acceleration' in selected_stats and len(series) >= 3:
                recent_trend = series[-1] - series[-2] if len(series) >= 2 else 0
                early_trend = series[1] - series[0] if len(series) >= 2 else 0
                features[f'{prefix}_acceleration'] = float(recent_trend - early_trend)
            
            # 6. 自相关（lag=1）
            if 'autocorr' in selected_stats and len(series) >= 3:
                try:
                    corr = np.corrcoef(series[:-1], series[1:])[0, 1]
                    if np.isfinite(corr):
                        features[f'{prefix}_autocorr'] = float(corr)
                except:
                    pass  # 跳过计算失败的情况
        
        # 7. 交叉特征相关性（自动查找匹配的特征）
        # RV vs VPIN
        rv_cols = [col for col in window_data.columns if 'rv' in col.lower() and not 'corr' in col.lower()]
        vpin_cols = [col for col in window_data.columns if 'vpin' in col.lower()]
        
        if rv_cols and vpin_cols:
            try:
                rv_col = rv_cols[0]  # 使用第一个匹配的
                vpin_col = vpin_cols[0]
                rv_series = window_data[rv_col].values
                vpin_series = window_data[vpin_col].values
                if len(rv_series) >= 3 and np.std(rv_series) > 0 and np.std(vpin_series) > 0:
                    corr = np.corrcoef(rv_series, vpin_series)[0, 1]
                    if np.isfinite(corr):
                        features[f'rv_vpin_corr_w{window}'] = float(corr)
            except:
                pass
        
        # Large vs Small
        large_cols = [col for col in window_data.columns if 'large' in col.lower() and 'signed' in col.lower()]
        small_cols = [col for col in window_data.columns if 'small' in col.lower() and 'signed' in col.lower()]
        
        if large_cols and small_cols:
            try:
                large_col = large_cols[0]
                small_col = small_cols[0]
                large_series = window_data[large_col].values
                small_series = window_data[small_col].values
                if len(large_series) >= 3 and np.std(large_series) > 0 and np.std(small_series) > 0:
                    corr = np.corrcoef(large_series, small_series)[0, 1]
                    if np.isfinite(corr):
                        features[f'large_small_corr_w{window}'] = float(corr)
            except:
                pass
        
        return features
    
    def _sum_range(self, cumsum_array: np.ndarray, start: int, end: int) -> float:
        """利用前缀和计算区间和（O(1)）"""
        if start >= len(cumsum_array) or end > len(cumsum_array):
            return 0.0
        if start == 0:
            return float(cumsum_array[end - 1])
        return float(cumsum_array[end - 1] - cumsum_array[start - 1])
    
    def _calculate_vpin(self, dv: np.ndarray, sign: np.ndarray, bins: int = 10) -> float:
        """计算 VPIN (Volume-synchronized Probability of Informed Trading)"""
        total = dv.sum()
        if total <= 0 or len(dv) < bins:
            return 0.0
        
        target = total / bins
        acc = 0.0
        buy = 0.0
        sell = 0.0
        vals = []
        
        for i in range(len(dv)):
            q = dv[i]
            acc += q
            if sign[i] > 0:
                buy += q
            elif sign[i] < 0:
                sell += q
            
            if acc >= target:
                denom = buy + sell
                v = abs(buy - sell) / denom if denom > 0 else 0.0
                vals.append(v)
                acc = 0.0
                buy = 0.0
                sell = 0.0
        
        if not vals:
            return 0.0
        
        return float(np.mean(vals))
    
    def get_feature_names(self, window: int, bar_features_df: Optional[pd.DataFrame] = None) -> List[str]:
        """获取生成的特征名称列表
        
        参数:
            window: 滚动窗口大小
            bar_features_df: bar 级特征 DataFrame（用于自动识别特征）
        
        返回:
            特征名称列表
        """
        # 如果提供了 bar_features_df，自动识别特征
        if bar_features_df is not None:
            base_features = [col for col in bar_features_df.columns 
                           if col.startswith('bar_') and 
                           pd.api.types.is_numeric_dtype(bar_features_df[col])]
        else:
            # 使用已知的特征提取器获取特征名称
            if self.feature_extractor is not None:
                # 获取特征提取器的所有特征名称
                extractor_features = self.feature_extractor.get_feature_names()
                base_features = [f'bar_{feat}' for feat in extractor_features]
            else:
                # 回退到默认列表
                base_features = [
                    'bar_rv', 'bar_bpv', 'bar_jump',
                    'bar_vpin', 'bar_small_signed_dollar', 'bar_large_signed_dollar',
                    'bar_hl_ratio', 'bar_signed_volume', 'bar_volume'
                ]
        
        suffixes = ['mean', 'std', 'min', 'max', 'trend', 'momentum', 'slope', 
                   'zscore', 'quantile', 'range_norm', 'acceleration', 'autocorr']
        
        names = []
        for feat in base_features:
            for suffix in suffixes:
                names.append(f'{feat}_w{window}_{suffix}')
        
        # 交叉特征（如果存在相关特征）
        if any('rv' in f for f in base_features) and any('vpin' in f for f in base_features):
            names.append(f'rv_vpin_corr_w{window}')
        if any('large' in f for f in base_features) and any('small' in f for f in base_features):
            names.append(f'large_small_corr_w{window}')
        
        return names

