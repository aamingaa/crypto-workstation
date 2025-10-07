"""
分桶与跨层订单流/冲击特征提取器

功能：
- 按成交额（quote）分位切分 small/mid/large（默认 q20, q80）
- 直接分桶统计：各桶买/卖金额、签名金额、到达强度
- 跨层指标：
  - 大单对小单（signed 差/比）
  - 跨层方向错配度（滞后相关，lag=1 默认）
  - 跨层冲击传导比（回归斜率对比，大/小）
  - 冲击凹性 α（|r| = k * size^α 的局部拟合）
  - 分桶 OFI/VPIN（近似法）

说明：
- 仅使用逐笔交易级别的 `TradesContext`，不依赖盘口快照。
- VPIN 采用滚动桶近似：在当前窗口内按成交额切分固定桶数，计算买卖失衡占总额。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from .base_extractor import MicrostructureBaseExtractor
from data.trades_processor import TradesContext


class BucketedFlowFeatureExtractor(MicrostructureBaseExtractor):
    """按分位切分 small/mid/large，并计算跨层特征"""

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'enabled': True,
            'low_q': 0.2,          # 小单上限分位
            'high_q': 0.8,         # 大单下限分位
            'lag': 1,              # 错配度/传导比的滞后步数（逐笔级）
            'vpin_bins': 10,       # VPIN 桶数（窗口内）
            'min_trades_alpha': 50 # 拟合凹性所需最少交易数
        }

    def get_feature_names(self) -> List[str]:
        names = [
            # 分桶基础
            'small_buy_dollar', 'small_sell_dollar', 'small_signed_dollar', 'small_count', 'small_arrival_intensity',
            'mid_buy_dollar', 'mid_sell_dollar', 'mid_signed_dollar', 'mid_count', 'mid_arrival_intensity',
            'large_buy_dollar', 'large_sell_dollar', 'large_signed_dollar', 'large_count', 'large_arrival_intensity',
            # 跨层关系
            'large_vs_small_signed_diff', 'large_vs_small_signed_ratio',
            'mismatch_corr_lag1',       # 大单流与小单流的滞后相关
            'impact_pass_ratio',        # 传导比：beta_large / beta_small
            'impact_beta_large', 'impact_beta_small',
            # 冲击凹性
            'impact_concavity_alpha',
            # 分桶 OFI/VPIN
            'small_ofi_signed_dollar', 'mid_ofi_signed_dollar', 'large_ofi_signed_dollar',
            'small_vpin', 'mid_vpin', 'large_vpin', 'vpin_all'
        ]
        return names

    def _extract_features(self, ctx: TradesContext, s: int, e: int,
                           start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
        if not self.feature_config.get('enabled', True):
            return {}

        if e - s <= 0:
            return {}

        # 取窗口内逐笔
        dv = ctx.quote[s:e]
        if dv.size == 0:
            return {}
        sign = ctx.sign[s:e]
        t_ns = ctx.t_ns[s:e]

        # 分位阈值
        low_q = float(self.feature_config.get('low_q', 0.2))
        high_q = float(self.feature_config.get('high_q', 0.8))
        dv_q_low = float(np.quantile(dv, low_q)) if np.isfinite(dv).all() else np.nan
        dv_q_high = float(np.quantile(dv, high_q)) if np.isfinite(dv).all() else np.nan
        if not (np.isfinite(dv_q_low) and np.isfinite(dv_q_high) and dv_q_low < dv_q_high):
            return {}

        small_mask = dv <= dv_q_low
        large_mask = dv >= dv_q_high
        mid_mask = ~(small_mask | large_mask)

        # 基础统计
        def _bucket_stats(mask: np.ndarray) -> Tuple[float, float, float, int, float]:
            if mask.sum() == 0:
                return 0.0, 0.0, 0.0, 0, 0.0
            buy_dollar = float(dv[mask][sign[mask] > 0].sum())
            sell_dollar = float(dv[mask][sign[mask] < 0].sum())
            signed_dollar = float((sign[mask] * dv[mask]).sum())
            count = int(mask.sum())
            duration_sec = max(1e-9, (end_ts - start_ts).total_seconds())
            intensity = count / duration_sec
            return buy_dollar, sell_dollar, signed_dollar, count, intensity

        s_buy, s_sell, s_signed, s_cnt, s_int = _bucket_stats(small_mask)
        m_buy, m_sell, m_signed, m_cnt, m_int = _bucket_stats(mid_mask)
        l_buy, l_sell, l_signed, l_cnt, l_int = _bucket_stats(large_mask)

        # 大对小：signed 差与比
        eps = 1e-12
        large_vs_small_signed_diff = l_signed - s_signed
        large_vs_small_signed_ratio = (l_signed / (abs(s_signed) + eps)) if abs(s_signed) > 0 else np.nan

        # 错配度：滞后相关（lag 按逐笔步长）
        lag = int(self.feature_config.get('lag', 1))
        def _lagged_corr(a: np.ndarray, b: np.ndarray, lag_steps: int) -> float:
            if a.size < lag_steps + 3 or b.size < lag_steps + 3:
                return np.nan
            a2 = a[lag_steps:]
            b2 = b[:-lag_steps]
            sa = np.std(a2)
            sb = np.std(b2)
            if sa == 0 or sb == 0:
                return np.nan
            c = np.corrcoef(a2, b2)[0, 1]
            return float(c) if np.isfinite(c) else np.nan

        large_flow = sign[large_mask] * dv[large_mask]
        small_flow = sign[small_mask] * dv[small_mask]
        mismatch_corr_lag1 = _lagged_corr(large_flow, small_flow, lag) if (large_flow.size>0 and small_flow.size>0) else np.nan

        # 传导比：未来短窗收益 r 对 当前 signed flow（大/小）的回归斜率比
        # 这里使用窗口内相邻对数收益（ctx.ret 与 price 对齐方式见 TradesContext），
        # 将逐笔收益截取为 s:e 区间的差分（与 dv 对齐需偏移一位），简化近似。
        # 使用 OLS beta = cov(flow, r) / var(flow)
        # 对大/小分别估计，再取比值。
        def _compute_betas(flow_mask: np.ndarray) -> float:
            idx = np.where(flow_mask)[0]
            if idx.size < 5:
                return np.nan
            # 构造同长度的收益序列：使用相邻 logp 差分与该成交对齐（去首尾）
            # 对齐：以成交 i 对应 r_i = logp[i] - logp[i-1]
            r = ctx.ret[s:e]
            # pad 使长度与 dv 一致（ret 比价序列少1个，这里截取与 dv 同长-1，再对齐中间部分）
            if r.size != dv.size - 1:
                # 保险兜底
                r = r[:max(0, dv.size - 1)]
            # 将 flow 与对应 r 对齐（忽略最后一笔的 r 缺失）
            eff_len = min(idx.size, r.size)
            if eff_len < 5:
                return np.nan
            x = (sign[s:e][:-1] * dv[s:e][:-1])[idx[:eff_len]]
            y = r[:eff_len]
            vx = np.var(x)
            if vx <= 0:
                return np.nan
            beta = float(np.cov(x, y, bias=True)[0, 1] / (vx + eps))
            return beta

        impact_beta_large = _compute_betas(large_mask)
        impact_beta_small = _compute_betas(small_mask)
        impact_pass_ratio = (impact_beta_large / impact_beta_small) if (impact_beta_small not in [0.0, np.nan] and np.isfinite(impact_beta_small)) else np.nan

        # 冲击凹性 α：|r| = k * size^α，log|r| = log k + α log size
        def _concavity_alpha() -> float:
            # 使用所有逐笔，过滤零与异常
            r = ctx.ret[s:e]
            if r.size < 20 or dv.size - 1 < 20:
                return np.nan
            # 对齐长度
            size = dv[:-1]
            abs_r = np.abs(r)
            mask = (size > 0) & np.isfinite(size) & (abs_r > 0) & np.isfinite(abs_r)
            if mask.sum() < int(self.feature_config.get('min_trades_alpha', 50)):
                return np.nan
            xs = np.log(size[mask])
            ys = np.log(abs_r[mask])
            vx = np.var(xs)
            if vx <= 0:
                return np.nan
            alpha = float(np.cov(xs, ys, bias=True)[0, 1] / (vx + eps))
            return alpha

        impact_concavity_alpha = _concavity_alpha()

        # 分桶 OFI：这里以签名金额和作为近似（逐笔级，没有盘口队列变化）
        small_ofi = s_signed
        mid_ofi = m_signed
        large_ofi = l_signed

        # VPIN 近似：窗口内将成交额按等额桶切片，计算每桶的 |Buy - Sell|/Total，取均值
        def _vpin_for_mask() -> float:
            # 全窗口 VPIN
            total = dv.sum()
            if total <= 0:
                return np.nan
            bins = int(self.feature_config.get('vpin_bins', 10))
            # 目标每桶金额
            target = total / bins
            acc = 0.0
            buy = 0.0
            sell = 0.0
            vals: List[float] = []
            for i in range(dv.size):
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
                return np.nan
            return float(np.mean(vals))

        vpin_all = _vpin_for_mask()

        # 分桶 VPIN：在各桶子序列上重复流程（按该子序列总额等额切片）
        def _vpin_on_subseq(mask: np.ndarray) -> float:
            idx = np.where(mask)[0]
            if idx.size < 5:
                return np.nan
            sub_dv = dv[idx]
            sub_sign = sign[idx]
            total = sub_dv.sum()
            if total <= 0:
                return np.nan
            bins = int(self.feature_config.get('vpin_bins', 10))
            target = total / bins
            acc = 0.0
            buy = 0.0
            sell = 0.0
            vals: List[float] = []
            for i in range(sub_dv.size):
                q = sub_dv[i]
                acc += q
                if sub_sign[i] > 0:
                    buy += q
                elif sub_sign[i] < 0:
                    sell += q
                if acc >= target:
                    denom = buy + sell
                    v = abs(buy - sell) / denom if denom > 0 else 0.0
                    vals.append(v)
                    acc = 0.0
                    buy = 0.0
                    sell = 0.0
            if not vals:
                return np.nan
            return float(np.mean(vals))

        small_vpin = _vpin_on_subseq(small_mask)
        mid_vpin = _vpin_on_subseq(mid_mask)
        large_vpin = _vpin_on_subseq(large_mask)

        duration_sec = max(1e-9, (end_ts - start_ts).total_seconds())

        return {
            # 基础分桶
            'small_buy_dollar': s_buy,
            'small_sell_dollar': s_sell,
            'small_signed_dollar': s_signed,
            'small_count': s_cnt,
            'small_arrival_intensity': s_int,
            'mid_buy_dollar': m_buy,
            'mid_sell_dollar': m_sell,
            'mid_signed_dollar': m_signed,
            'mid_count': m_cnt,
            'mid_arrival_intensity': m_int,
            'large_buy_dollar': l_buy,
            'large_sell_dollar': l_sell,
            'large_signed_dollar': l_signed,
            'large_count': l_cnt,
            'large_arrival_intensity': l_int,
            # 跨层
            'large_vs_small_signed_diff': large_vs_small_signed_diff,
            'large_vs_small_signed_ratio': large_vs_small_signed_ratio,
            'mismatch_corr_lag1': mismatch_corr_lag1,
            'impact_pass_ratio': impact_pass_ratio,
            'impact_beta_large': impact_beta_large,
            'impact_beta_small': impact_beta_small,
            # 凹性
            'impact_concavity_alpha': impact_concavity_alpha,
            # OFI/VPIN
            'small_ofi_signed_dollar': small_ofi,
            'mid_ofi_signed_dollar': mid_ofi,
            'large_ofi_signed_dollar': large_ofi,
            'small_vpin': small_vpin,
            'mid_vpin': mid_vpin,
            'large_vpin': large_vpin,
            'vpin_all': vpin_all,
        }


