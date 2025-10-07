"""
大单特征提取器
包含大单方向性特征、尾部分布特征等
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .base_extractor import MicrostructureBaseExtractor
from data.trades_processor import TradesContext


class TailFeatureExtractor(MicrostructureBaseExtractor):
    """大单特征提取器"""
    
    def _get_default_config(self) -> Dict[str, bool]:
        """获取默认特征配置"""
        return {
            'enabled': True,
            'quantiles': [0.9, 0.95],  # 大单定义的分位数阈值
            # 扫盘（簇）识别参数
            'sweep_window_sec': 1.0,   # 同向大单之间时间差阈值（秒），相邻大单间隔<=该阈值且同向视为同一簇
            'sweep_min_trades': 2,     # 形成扫盘簇的最小笔数（小于该数则忽略该簇）
        }
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        quantiles = self.feature_config.get('quantiles', [0.9, 0.95])
        names = []
        
        for q in quantiles:
            tag = f"q{int(round(q * 100))}"
            names.extend([
                # 规模类
                f'large_{tag}_count',
                f'large_{tag}_dollar_sum',
                f'large_{tag}_avg_dollar',
                f'large_{tag}_max_dollar',
                f'large_{tag}_std_dollar',
                f'large_{tag}_participation_dollar',
                # 方向/不平衡类（Taker 方向）
                f'large_{tag}_buy_dollar_sum',
                f'large_{tag}_sell_dollar_sum',
                f'large_{tag}_taker_buy_count',
                f'large_{tag}_taker_sell_count',
                f'large_{tag}_imbalance_by_count',  # (buy_count - sell_count)/total_count
                f'large_{tag}_imbalance_by_dollar', # (buy_dollar - sell_dollar)/total_dollar
                f'large_{tag}_signed_dollar_sum',
                f'large_{tag}_lti',  # 保留兼容：与 imbalance_by_dollar 等价
                # 集中度
                f'large_{tag}_herfindahl_dollar',   # ∑(份额^2)，越大表示越集中
                f'large_{tag}_top1_share_dollar',   # 最大一笔大单金额占比
                f'large_{tag}_top3_share_dollar',   # 最大三笔大单金额占比
                # 到达/簇性（时序）
                f'large_{tag}_arrival_intensity',   # 大单到达强度 = count/秒
                f'large_{tag}_interarrival_mean',   # 大单相邻到达时间间隔（秒）的均值
                f'large_{tag}_interarrival_std',    # 大单相邻到达时间间隔（秒）的标准差
                f'large_{tag}_runlen_buy_max',      # 连续同向（买）的最长 run 长度
                f'large_{tag}_runlen_sell_max',     # 连续同向（卖）的最长 run 长度
                f'large_{tag}_burstiness',          # (std-mean)/(std+mean)，衡量簇性
                f'large_{tag}_sweep_count',         # 扫盘簇个数（同向、间隔小、笔数>=阈值）
                f'large_{tag}_sweep_avg_size',      # 扫盘簇金额均值（按簇均值）
                f'large_{tag}_sweep_max_size',      # 扫盘簇金额最大值
            ])
        
        return names
    
    def _extract_features(self, ctx: TradesContext, s: int, e: int,
                         start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
        """提取大单特征"""
        if not self.feature_config.get('enabled', True):
            return {}
        
        if e - s <= 0:
            return {}
        
        # 获取交易金额和方向
        dv = ctx.quote[s:e]  # 逐笔成交额（USDT）
        if dv.size == 0:
            return {}
        
        sign = ctx.sign[s:e]  # 方向：+1= Taker 主动买，-1= Taker 主动卖
        t_ns = ctx.t_ns[s:e]  # 逐笔纳秒级时间戳（已按时间排序）
        eps = 1e-12  # 防止除零
        
        quantiles = self.feature_config.get('quantiles', [0.9, 0.95])
        features = {}
        window_total_dollar = float(dv.sum()) if dv.size > 0 else 0.0  # 窗口内总金额
        duration_sec = max(1e-9, (end_ts - start_ts).total_seconds())   # 窗口时长（秒）

        # 辅助函数：最长连续 True 长度
        def _max_runlen(mask_arr: np.ndarray) -> int:
            if mask_arr.size == 0:
                return 0
            # 将 False 作为分隔，统计连续 True 的最大长度
            # 利用差分定位边界
            x = mask_arr.astype(np.int32)
            if x.sum() == 0:
                return 0
            # 计算 run-length
            # 找到 0 和 1 的边界位置
            diff = np.diff(np.concatenate(([0], x, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            return int((ends - starts).max()) if starts.size and ends.size else 0
        
        # 辅助函数：扫盘（簇）统计（同向、相邻间隔<=阈值，长度>=min_trades）
        def _sweep_stats(ts_ns: np.ndarray, amount: np.ndarray, direction: np.ndarray,
                         gap_sec: float, min_trades: int) -> Tuple[int, float, float]:
            if ts_ns.size == 0:
                return 0, 0.0, 0.0
            # 排序保证时间单调（通常已按时间排序）
            order = np.arange(ts_ns.size)
            ts = ts_ns.astype(np.float64) / 1e9
            # 遍历，按同向+间隔约束聚类
            count = 0
            sizes: List[float] = []
            i = 0
            while i < ts.size:
                j = i + 1
                # 当前簇方向
                dir0 = direction[i]
                cluster_sum = float(amount[i])
                cluster_len = 1
                while j < ts.size and (ts[j] - ts[j - 1] <= gap_sec) and (direction[j] == dir0):
                    cluster_sum += float(amount[j])
                    cluster_len += 1
                    j += 1
                if cluster_len >= min_trades:
                    count += 1
                    sizes.append(cluster_sum)
                i = j
            if count == 0:
                return 0, 0.0, 0.0
            sizes_arr = np.asarray(sizes, dtype=float)
            return count, float(sizes_arr.mean()), float(sizes_arr.max())
        
        for q in quantiles:
            # 计算分位数阈值
            thr = float(np.quantile(dv, q)) if np.isfinite(dv).all() else np.nan
            if not np.isfinite(thr) or thr <= 0:
                continue
            
            # 识别大单
            mask = dv >= thr
            if not mask.any():
                continue
            
            # 大单子集
            dv_large = dv[mask]
            sign_large = sign[mask]
            t_large = t_ns[mask]

            # 规模类
            large_count = int(mask.sum())
            large_dollar_sum = float(dv_large.sum())
            large_avg = float(dv_large.mean()) if large_count > 0 else np.nan
            large_max = float(dv_large.max()) if large_count > 0 else np.nan
            large_std = float(dv_large.std()) if large_count > 1 else np.nan
            participation = (large_dollar_sum / (window_total_dollar + eps)) if window_total_dollar > 0 else np.nan  # 大单金额相对窗口总额占比

            # 方向（Taker）与不平衡
            taker_buy_mask = sign_large > 0
            taker_sell_mask = sign_large < 0
            taker_buy_count = int(taker_buy_mask.sum())
            taker_sell_count = int(taker_sell_mask.sum())
            taker_buy_dollar = float(dv_large[taker_buy_mask].sum())
            taker_sell_dollar = float(dv_large[taker_sell_mask].sum())
            imbalance_count = ((taker_buy_count - taker_sell_count) / (taker_buy_count + taker_sell_count)) if (taker_buy_count + taker_sell_count) > 0 else np.nan
            imbalance_dollar = ((taker_buy_dollar - taker_sell_dollar) / (taker_buy_dollar + taker_sell_dollar + eps))  # 金额不平衡
            signed_dollar_sum = taker_buy_dollar - taker_sell_dollar
            lti = imbalance_dollar  # 兼容命名（Liquidity Taker Imbalance）

            # 集中度
            if large_dollar_sum > 0:
                share = (dv_large / large_dollar_sum)
                herfindahl = float(np.sum(share * share))  # ∑(份额^2)
                # top-k 占比
                # 使用 partial 排序避免全量排序开销（但这里数量一般不大，直接排序即可）
                dv_sorted = np.sort(dv_large)[::-1]
                top1_share = float(dv_sorted[0] / large_dollar_sum) if dv_sorted.size >= 1 else 0.0
                top3_share = float(dv_sorted[:3].sum() / large_dollar_sum) if dv_sorted.size >= 3 else float(dv_sorted.sum() / large_dollar_sum)
            else:
                herfindahl = np.nan
                top1_share = np.nan
                top3_share = np.nan

            # 到达与簇性
            if large_count >= 1:
                arrival_intensity = large_count / duration_sec
            else:
                arrival_intensity = 0.0

            if t_large.size >= 2:
                gaps_sec = np.diff(t_large.astype(np.float64)) / 1e9
                interarrival_mean = float(gaps_sec.mean())
                interarrival_std = float(gaps_sec.std())
                if interarrival_mean + interarrival_std > 0:
                    burstiness = float((interarrival_std - interarrival_mean) / (interarrival_std + interarrival_mean))  # [-1,1]，越大越簇状
                else:
                    burstiness = np.nan
            else:
                interarrival_mean = np.nan
                interarrival_std = np.nan
                burstiness = np.nan

            runlen_buy_max = _max_runlen(taker_buy_mask)
            runlen_sell_max = _max_runlen(taker_sell_mask)

            # 扫盘（簇）统计：同向且相邻间隔<=阈值、长度>=min_trades
            sweep_window_sec = float(self.feature_config.get('sweep_window_sec', 1.0))
            sweep_min_trades = int(self.feature_config.get('sweep_min_trades', 2))
            sweep_count_b, sweep_avg_b, sweep_max_b = _sweep_stats(
                ts_ns=t_large[taker_buy_mask],
                amount=dv_large[taker_buy_mask],
                direction=np.ones(taker_buy_mask.sum(), dtype=np.int8),
                gap_sec=sweep_window_sec,
                min_trades=sweep_min_trades,
            )
            sweep_count_s, sweep_avg_s, sweep_max_s = _sweep_stats(
                ts_ns=t_large[taker_sell_mask],
                amount=dv_large[taker_sell_mask],
                direction=-np.ones(taker_sell_mask.sum(), dtype=np.int8),
                gap_sec=sweep_window_sec,
                min_trades=sweep_min_trades,
            )
            sweep_count = sweep_count_b + sweep_count_s
            # 聚合平均/最大（对无簇时返回0.0）
            if sweep_count == 0:
                sweep_avg_size = 0.0
                sweep_max_size = 0.0
            else:
                # 均值用总均值（按簇数权重相等）
                sizes = [v for v in [sweep_avg_b, sweep_avg_s] if v > 0]
                # 更合理是拼接实际 sizes 列表，这里近似取两边非零均值的平均
                sweep_avg_size = float(np.mean(sizes)) if sizes else 0.0
                sweep_max_size = float(max(sweep_max_b, sweep_max_s))

            tag = f"q{int(round(q * 100))}"
            features.update({
                # 规模
                f'large_{tag}_count': large_count,
                f'large_{tag}_dollar_sum': large_dollar_sum,
                f'large_{tag}_avg_dollar': large_avg,
                f'large_{tag}_max_dollar': large_max,
                f'large_{tag}_std_dollar': large_std,
                f'large_{tag}_participation_dollar': participation,
                # 方向/不平衡
                f'large_{tag}_buy_dollar_sum': taker_buy_dollar,
                f'large_{tag}_sell_dollar_sum': taker_sell_dollar,
                f'large_{tag}_taker_buy_count': taker_buy_count,
                f'large_{tag}_taker_sell_count': taker_sell_count,
                f'large_{tag}_imbalance_by_count': float(imbalance_count) if np.isfinite(imbalance_count) else np.nan,
                f'large_{tag}_imbalance_by_dollar': imbalance_dollar,
                f'large_{tag}_signed_dollar_sum': signed_dollar_sum,
                f'large_{tag}_lti': lti,
                # 集中度
                f'large_{tag}_herfindahl_dollar': herfindahl,
                f'large_{tag}_top1_share_dollar': top1_share,
                f'large_{tag}_top3_share_dollar': top3_share,
                # 到达/簇性
                f'large_{tag}_arrival_intensity': arrival_intensity,
                f'large_{tag}_interarrival_mean': interarrival_mean,
                f'large_{tag}_interarrival_std': interarrival_std,
                f'large_{tag}_runlen_buy_max': runlen_buy_max,
                f'large_{tag}_runlen_sell_max': runlen_sell_max,
                f'large_{tag}_burstiness': burstiness,
                f'large_{tag}_sweep_count': sweep_count,
                f'large_{tag}_sweep_avg_size': sweep_avg_size,
                f'large_{tag}_sweep_max_size': sweep_max_size,
            })
        
        return features
