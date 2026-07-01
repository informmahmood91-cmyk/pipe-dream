# =================================================================
# Mike Trend.py – MIKA Compressor v5.6.1
# Divergence & Exhaustion Fixes + PATCHED divergence detector
# (pure pandas/numpy – no external dependencies)
#
# CHANGELOG vs v5.6:
#   FIX 1: _detect_divergence rewritten to use TRAILING (right-aligned)
#          swing detection instead of centered rolling(3, center=True).
#          The old centered method could NEVER classify the most recent
#          (current/live) bar as a swing high/low, because a centered
#          3-bar window requires a bar AFTER it to confirm the peak.
#          On live data the current bar is always the last bar, so
#          divergence on the live bar could never be detected.
#   FIX 2: Divergence lookback window widened from 20 -> 40 bars
#          (configurable via DIVERGENCE_LOOKBACK) so peaks/troughs
#          spaced further apart are still compared.
#   FIX 3: Restored try/except safety net around
#          _detect_reversal_exhaustion (was present in v5.5, dropped
#          in v5.6).
# =================================================================

import math
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class RegimeInfo:
    regime: str
    mode: str
    confidence: float
    adx: float
    bb_width: float
    atr_percentile: float

@dataclass
class IndicatorResult:
    indicator: str
    category: str
    signal: str
    weight: float
    detail: Dict[str, Any] = field(default_factory=dict)
    is_missing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {"indicator": self.indicator, "category": self.category,
             "signal": self.signal, "weight": self.weight, "is_missing": self.is_missing}
        d.update(self.detail)
        return d

class MIKACompressor:
    ADX_TREND_THRESHOLD          = 25.0
    ADX_WEAK_TREND_THRESHOLD     = 20.0
    ADX_RANGE_THRESHOLD          = 20.0
    BB_COMPRESSION_THRESHOLD     = 5.0
    BB_EXPANSION_THRESHOLD       = 15.0
    BB_WIDTH_SQUEEZE_PERCENTILE   = 20.0   # FIX: percentile-based squeeze detection
    BB_WIDTH_EXPANSION_PERCENTILE = 80.0   # FIX: percentile-based expansion detection
    ATR_HIGH_VOL_PERCENTILE      = 75.0
    ATR_LOW_VOL_PERCENTILE       = 25.0
    RSI_OVERBOUGHT               = 70.0
    RSI_OVERSOLD                 = 30.0
    RSI_TRENDING_MIDLINE         = 50.0
    STOCH_OVERBOUGHT             = 80.0
    STOCH_OVERSOLD               = 20.0
    VOLUME_HIGH_RATIO            = 1.2
    VOLUME_LOW_RATIO             = 0.6
    VOLUME_BREAKOUT_THRESHOLD    = 1.5
    EMA_SLOPE_WEIGHT_BONUS       = 0.05
    ALIGNMENT_FULL               = 0.80
    ALIGNMENT_STRONG             = 0.60
    ALIGNMENT_PARTIAL            = 0.40
    MIN_ACTIVE_CATEGORIES        = 4
    LEVEL_PROXIMITY_PCT          = 0.15
    CONV_NO_TRADE   = 15.0
    CONV_PROBE      = 30.0
    CONV_SMALL      = 50.0
    CONV_NORMAL     = 70.0
    MAX_TOTAL_PENALTY = 0.35

    def __init__(self, tv_data: Dict[str, Any], twelve_data: Optional[Dict[str, Any]] = None):
        self.tv_data    = tv_data
        self.twelve_data = twelve_data or {}

    def _sf(self, v: Any, default: float = 0.0) -> float:
        if v is None or v == "N/A" or v == "":
            return default
        try:
            f = float(str(v).replace(",", "").strip())
        except (ValueError, TypeError):
            return default
        if math.isnan(f) or math.isinf(f):
            return default
        return f

    def _ss(self, v: Any, default: str = "N/A") -> str:
        if v is None:
            return default
        if isinstance(v, bool):
            return "true" if v else "false"
        s = str(v).strip()
        return s if s else default

    def _sb(self, v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        return str(v).lower() in ("true", "yes", "1", "t", "y")

    def _is_missing(self, value: Any) -> bool:
        if value is None:
            return True
        if value == "N/A":
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return True
        if isinstance(value, (int, float)):
            return False
        if isinstance(value, str) and value.replace(".", "").replace("-", "").isdigit():
            return False
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return False
        if isinstance(value, str) and len(value.strip()) > 0:
            return False
        return False

    def _detect_regime(self) -> RegimeInfo:
        adx      = self._sf(self.tv_data.get("adx"), 0.0)
        bb_width = self._sf(self.twelve_data.get("bb_width"), 100.0)
        atr      = self._sf(self.tv_data.get("atr"), 0.0)
        price    = self._sf(self.tv_data.get("price"), 1.0)
        price    = price if price > 0 else 1.0
        supplied = self.tv_data.get("atr_percentile")
        atr_pct  = self._sf(supplied, 50.0) if supplied is not None else min(100.0, (atr / price) * 1000)
        pine_regime = self._ss(self.tv_data.get("market_regime"), "").upper()
        t_votes = r_votes = vol_votes = 0
        if adx > self.ADX_TREND_THRESHOLD:       t_votes   += 2
        elif adx > self.ADX_WEAK_TREND_THRESHOLD: t_votes  += 1
        else:                                      r_votes  += 1
        # FIX: BB width scaling bug. Fixed absolute thresholds (5% / 15%) assumed
        # a scale that real FX bb_width values (typically 0.02%-1.5% on 4h) never
        # reach, which meant almost every bar registered as "compressed" regardless
        # of actual volatility. Use bb_width_percentile (computed relative to that
        # symbol's own trailing history in compute_indicators_for_symbol) when
        # available, since it self-calibrates per symbol/timeframe. Falls back to
        # the legacy absolute thresholds only if the percentile field is absent
        # (e.g. older Pine webhook payloads that don't send it yet).
        bbw_pctile_raw = self.tv_data.get("bb_width_percentile")
        if bbw_pctile_raw is not None and not self._is_missing(bbw_pctile_raw):
            bbw_pctile = self._sf(bbw_pctile_raw, 50.0)
            if bbw_pctile < self.BB_WIDTH_SQUEEZE_PERCENTILE:      r_votes   += 2
            elif bbw_pctile > self.BB_WIDTH_EXPANSION_PERCENTILE:  vol_votes += 2
            else:                                                   r_votes   += 1
        else:
            if bb_width < self.BB_COMPRESSION_THRESHOLD:    r_votes   += 2
            elif bb_width > self.BB_EXPANSION_THRESHOLD:    vol_votes += 2
            else:                                            r_votes   += 1
        if atr_pct > self.ATR_HIGH_VOL_PERCENTILE:  vol_votes += 2
        elif atr_pct < self.ATR_LOW_VOL_PERCENTILE: r_votes   += 1
        else:                                         t_votes   += 1
        vote_map  = {"TRENDING": t_votes, "RANGING": r_votes, "VOLATILE": vol_votes}
        max_votes = max(vote_map.values())
        if max_votes == 0:
            regime, confidence = "TRANSITION", 30.0
        else:
            regime     = max(vote_map, key=vote_map.get)
            total      = t_votes + r_votes + vol_votes
            confidence = (max_votes / total * 100) if total > 0 else 50.0
        if "TRENDING BULL" in pine_regime:   regime, confidence = "BULL_TREND",  85.0
        elif "TRENDING BEAR" in pine_regime: regime, confidence = "BEAR_TREND",  85.0
        elif "RANGING"       in pine_regime: regime, confidence = "RANGING",     70.0
        mode = ("TREND_FOLLOW" if regime in ("BULL_TREND", "BEAR_TREND")
                else "MEAN_REVERT" if regime == "RANGING"
                else "CAUTION")
        return RegimeInfo(regime, mode, confidence, adx, bb_width, atr_pct)
    
    def _analyze_rsi(self, regime: RegimeInfo) -> IndicatorResult:
        rsi_raw = self.twelve_data.get("rsi")
        rsi = self._sf(rsi_raw, 50.0)
        is_missing = self._is_missing(rsi_raw)
        signal = "NEUTRAL"
        weight = 0.60
        if not is_missing:
            if regime.regime in ("BULL_TREND", "BEAR_TREND"):
                if regime.regime == "BULL_TREND":
                    if rsi > self.RSI_TRENDING_MIDLINE:
                        signal, weight = "BULLISH", 0.85
                    elif rsi < (100 - self.RSI_TRENDING_MIDLINE):
                        signal, weight = "BEARISH", 0.85
                else:
                    if rsi < self.RSI_TRENDING_MIDLINE:
                        signal, weight = "BEARISH", 0.85
                    elif rsi > (100 - self.RSI_TRENDING_MIDLINE):
                        signal, weight = "BULLISH", 0.85
            elif regime.regime == "RANGING":
                if rsi < self.RSI_OVERSOLD:  signal = "BULLISH"
                elif rsi > self.RSI_OVERBOUGHT: signal = "BEARISH"
            else:
                if rsi < self.RSI_OVERSOLD:  signal = "BULLISH"
                elif rsi > self.RSI_OVERBOUGHT: signal = "BEARISH"
        return IndicatorResult("RSI", "MOMENTUM", signal, weight, {"value": rsi}, is_missing)

    def _analyze_stochastic(self, regime: RegimeInfo) -> IndicatorResult:
        k_raw = self.twelve_data.get("stoch_k")
        d_raw = self.twelve_data.get("stoch_d")
        k = self._sf(k_raw, 50.0)
        d = self._sf(d_raw, k)
        is_missing = self._is_missing(k_raw) or self._is_missing(d_raw)
        signal, weight = "NEUTRAL", 0.55
        if not is_missing:
            if regime.mode == "MEAN_REVERT":
                if k < self.STOCH_OVERSOLD:  signal, weight = "BULLISH", 0.65
                elif k > self.STOCH_OVERBOUGHT: signal, weight = "BEARISH", 0.65
            else:
                if k > 50 and k > d:   signal = "BULLISH"
                elif k < 50 and k < d: signal = "BEARISH"
        return IndicatorResult("STOCHASTIC", "MOMENTUM", signal, weight, {"k": k, "d": d}, is_missing)

    def _analyze_macd(self, regime: RegimeInfo) -> IndicatorResult:
        hist_raw = self.twelve_data.get("macd_histogram")
        line_raw = self.twelve_data.get("macd")
        sig_raw  = self.twelve_data.get("macd_signal")
        hist = self._sf(hist_raw, 0.0)
        line = self._sf(line_raw, 0.0)
        sig  = self._sf(sig_raw, 0.0)
        is_missing = self._is_missing(hist_raw) or self._is_missing(line_raw) or self._is_missing(sig_raw)
        signal = "NEUTRAL"
        if not is_missing:
            if hist > 0 and line > sig:   signal = "BULLISH"
            elif hist < 0 and line < sig: signal = "BEARISH"
        return IndicatorResult("MACD", "MOMENTUM", signal, 0.85, {"hist": hist}, is_missing)

    def _analyze_bollinger_bands(self, regime: RegimeInfo) -> IndicatorResult:
        price    = self._sf(self.tv_data.get("price"), 0.0)
        bb_upper_raw = self.twelve_data.get("bb_upper")
        bb_lower_raw = self.twelve_data.get("bb_lower")
        bb_width_raw = self.twelve_data.get("bb_width")
        bb_upper = self._sf(bb_upper_raw, 0.0)
        bb_lower = self._sf(bb_lower_raw, 0.0)
        bb_width = self._sf(bb_width_raw, 100.0)
        is_missing = self._is_missing(bb_upper_raw) or self._is_missing(bb_lower_raw)
        signal = "NEUTRAL"
        if not is_missing and price > 0:
            if regime.mode == "MEAN_REVERT":
                if bb_lower > 0 and price <= bb_lower * 1.02: signal = "BULLISH"
                elif bb_upper > 0 and price >= bb_upper * 0.98: signal = "BEARISH"
            else:
                if bb_upper > 0 and price > bb_upper: signal = "BULLISH"
                elif bb_lower > 0 and price < bb_lower: signal = "BEARISH"
        return IndicatorResult("BOLLINGER", "VOLATILITY", signal, 0.85,
                               {"bb_width": bb_width}, is_missing)

    def _analyze_ema(self, regime: RegimeInfo) -> IndicatorResult:
        e21_raw = self.tv_data.get("ema_21")
        e50_raw = self.tv_data.get("ema_50")
        e200_raw = self.tv_data.get("ema_200")
        e21   = self._sf(e21_raw, 0.0)
        e50   = self._sf(e50_raw, 0.0)
        e200  = self._sf(e200_raw, 0.0)
        align = self._ss(self.tv_data.get("ema_align"), "")
        slope = self._ss(self.tv_data.get("ema_slope"), "").upper()
        is_missing = self._is_missing(e21_raw) or self._is_missing(e50_raw) or self._is_missing(e200_raw)
        signal, weight = "NEUTRAL", 0.85
        if not is_missing:
            if e21 > e50 > e200:   signal = "BULLISH"
            elif e21 < e50 < e200: signal = "BEARISH"
            if align == "BULL":    signal = "BULLISH"
            elif align == "BEAR":  signal = "BEARISH"
            slope_confirms = ((signal == "BULLISH" and slope == "UP") or
                              (signal == "BEARISH" and slope == "DOWN"))
            if slope_confirms:
                weight = min(1.0, weight + self.EMA_SLOPE_WEIGHT_BONUS)
            elif ((signal == "BULLISH" and slope == "DOWN") or
                  (signal == "BEARISH" and slope == "UP")):
                weight = max(0.50, weight - self.EMA_SLOPE_WEIGHT_BONUS)
        return IndicatorResult("EMA", "TREND", signal, weight, {"slope": slope}, is_missing)

    def _analyze_vwap(self, regime: RegimeInfo) -> IndicatorResult:
        price      = self._sf(self.tv_data.get("price"), 0.0)
        w_vwap_raw = self.tv_data.get("weekly_vwap")
        m_vwap_raw = self.tv_data.get("monthly_vwap")
        pvwap      = self._ss(self.tv_data.get("price_vs_vwap"), "")
        w_vwap = self._sf(w_vwap_raw, 0.0)
        m_vwap = self._sf(m_vwap_raw, 0.0)
        is_missing = self._is_missing(w_vwap_raw) or self._is_missing(m_vwap_raw)
        signal, weight, confirm = "NEUTRAL", 0.85, 0
        if pvwap == "ABOVE":   signal = "BULLISH"; confirm += 1
        elif pvwap == "BELOW": signal = "BEARISH"; confirm += 1
        if not self._is_missing(w_vwap_raw) and w_vwap > 0:
            confirm += 1 if (price > w_vwap) == (signal == "BULLISH") else -1
        if not self._is_missing(m_vwap_raw) and m_vwap > 0:
            confirm += 1 if (price > m_vwap) == (signal == "BULLISH") else -1
        if confirm >= 2:
            weight = 0.90
        elif confirm <= -2:
            weight = 0.90
            signal = "BEARISH" if signal == "BULLISH" else "BULLISH"
        return IndicatorResult("VWAP", "STRUCTURE", signal, weight,
                               {"confirmation": confirm}, is_missing)

    def _analyze_pivot(self, regime: RegimeInfo) -> IndicatorResult:
        price   = self._sf(self.tv_data.get("price"), 0.0)
        pivot_raw = self.tv_data.get("weekly_pivot")
        r1_raw = self.tv_data.get("weekly_r1")
        s1_raw = self.tv_data.get("weekly_s1")
        pivot   = self._sf(pivot_raw, 0.0)
        r1      = self._sf(r1_raw, 0.0)
        s1      = self._sf(s1_raw, 0.0)
        nearest = self._ss(self.tv_data.get("nearest_level"), "")
        is_missing = self._is_missing(pivot_raw)
        signal  = "NEUTRAL"
        if not is_missing and pivot > 0:
            if price > pivot:
                signal = "BULLISH"
            elif price < pivot:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"
            if nearest in ("R1", "R2", "R3") and r1 > 0 and price <= r1 * 1.01:
                signal = "BEARISH"
            elif nearest in ("S1", "S2", "S3") and s1 > 0 and price >= s1 * 0.99:
                signal = "BULLISH"
        return IndicatorResult("PIVOT", "STRUCTURE", signal, 0.60,
                               {"nearest": nearest}, is_missing)

    def _analyze_structure(self, regime: RegimeInfo) -> IndicatorResult:
        bias      = self._ss(self.tv_data.get("structure_bias"), "")
        hh_hl     = self._ss(self.tv_data.get("hh_hl_pred"), "")
        fail_up   = self._sb(self.tv_data.get("lon_fail_up"))   or self._sb(self.tv_data.get("ny_fail_up"))
        fail_down = self._sb(self.tv_data.get("lon_fail_down")) or self._sb(self.tv_data.get("ny_fail_down"))
        is_missing = self._is_missing(self.tv_data.get("structure_bias"))
        signal, weight = "NEUTRAL", 1.0
        if bias == "BULL":   signal = "BULLISH"
        elif bias == "BEAR": signal = "BEARISH"
        if hh_hl in ("HH", "HL"):
            if signal == "NEUTRAL": signal = "BULLISH"
            elif signal == "BEARISH": signal = "NEUTRAL"
        elif hh_hl == "LL":
            if signal == "NEUTRAL": signal = "BEARISH"
            elif signal == "BULLISH": signal = "NEUTRAL"
        both_confirmed = (bias in ("BULL", "BEAR") and
                          hh_hl in ("HH", "HL", "LL") and
                          signal != "NEUTRAL")
        if fail_up and not fail_down:
            if signal == "BULLISH" and both_confirmed: weight = 0.70
            else: signal = "BEARISH"
        elif fail_down and not fail_up:
            if signal == "BEARISH" and both_confirmed: weight = 0.70
            else: signal = "BULLISH"
        return IndicatorResult("STRUCTURE", "STRUCTURE", signal, weight,
                               {"bias": bias}, is_missing)

    def _analyze_volume(self, regime: RegimeInfo) -> IndicatorResult:
        vol_ratio_raw = self.tv_data.get("volume_ratio")
        vol_ratio  = self._sf(vol_ratio_raw, 1.0)
        raw_signal = self._sf(self.tv_data.get("raw_signal"), 0.0)
        is_missing = self._is_missing(vol_ratio_raw)
        signal     = "NEUTRAL"
        if not is_missing:
            if vol_ratio > self.VOLUME_HIGH_RATIO:
                signal = "BULLISH" if raw_signal > 0 else "BEARISH" if raw_signal < 0 else "NEUTRAL"
            elif vol_ratio < self.VOLUME_LOW_RATIO:
                signal = "NEUTRAL"
        return IndicatorResult("VOLUME", "VOLUME", signal, 0.70,
                               {"ratio": vol_ratio}, is_missing)

    def _analyze_adx_di(self, regime: RegimeInfo) -> IndicatorResult:
        adx_raw = self.tv_data.get("adx")
        diplus_raw = self.tv_data.get("diplus")
        diminus_raw = self.tv_data.get("diminus")
        adx     = self._sf(adx_raw, 0.0)
        diplus  = self._sf(diplus_raw, 0.0)
        diminus = self._sf(diminus_raw, 0.0)
        is_missing = self._is_missing(adx_raw) or self._is_missing(diplus_raw) or self._is_missing(diminus_raw)
        signal  = "NEUTRAL"
        weight  = 0.85 if adx >= self.ADX_RANGE_THRESHOLD else 0.50
        if not is_missing and diplus > diminus > 0:  signal = "BULLISH"
        elif not is_missing and diminus > diplus > 0: signal = "BEARISH"
        return IndicatorResult("ADX_DI", "TREND", signal, weight, {"adx": adx}, is_missing)

    def _analyze_ichimoku(self, regime: RegimeInfo) -> IndicatorResult:
        price = self._sf(self.tv_data.get("price"), 0.0)
        conv_raw = self.twelve_data.get("ichimoku_conversion")
        if self._is_missing(conv_raw):
            conv_raw = self.tv_data.get("ichimoku_conversion")
        base_raw = self.twelve_data.get("ichimoku_base")
        if self._is_missing(base_raw):
            base_raw = self.tv_data.get("ichimoku_base")
        span_a_raw = self.twelve_data.get("ichimoku_span_a")
        if self._is_missing(span_a_raw):
            span_a_raw = self.tv_data.get("ichimoku_span_a")
        span_b_raw = self.twelve_data.get("ichimoku_span_b")
        if self._is_missing(span_b_raw):
            span_b_raw = self.tv_data.get("ichimoku_span_b")

        conv   = self._sf(conv_raw, 0.0)
        base   = self._sf(base_raw, 0.0)
        span_a = self._sf(span_a_raw, 0.0)
        span_b = self._sf(span_b_raw, 0.0)

        has_conv_base = (conv > 0 and base > 0)
        has_span = (span_a > 0 and span_b > 0)
        is_missing = (not has_conv_base) and (not has_span)

        signal = "NEUTRAL"
        weight = 0.60
        if has_conv_base:
            if price > conv and price > base:
                signal = "BULLISH"
            elif price < conv and price < base:
                signal = "BEARISH"
        if has_span:
            if span_a > span_b and price > span_a:
                weight = 0.85
            elif span_b > span_a and price < span_a:
                weight = 0.85
        return IndicatorResult("ICHIMOKU", "TREND", signal, weight, {}, is_missing)

    def _analyze_volume_profile(self, regime: RegimeInfo) -> IndicatorResult:
        above_vah_raw = self.tv_data.get("price_above_vah")
        below_val_raw = self.tv_data.get("price_below_val")
        price_at_poc_raw = self.tv_data.get("price_at_poc")
        above_vah   = self._sb(above_vah_raw)
        below_val   = self._sb(below_val_raw)
        price_at_poc = self._sb(price_at_poc_raw)
        is_missing = (self._is_missing(above_vah_raw) or self._is_missing(below_val_raw))
        signal, weight = "NEUTRAL", 0.85
        if not is_missing:
            if regime.regime == "BULL_TREND":
                if below_val:   signal = "BULLISH"
                elif above_vah: signal = "BULLISH"
            elif regime.regime == "BEAR_TREND":
                if above_vah:   signal = "BEARISH"
                elif below_val: signal = "BEARISH"
            else:
                if below_val:   signal = "BULLISH"
                elif above_vah: signal = "BEARISH"
            if price_at_poc:
                signal, weight = "NEUTRAL", 0.90
        return IndicatorResult("VOLUME_PROFILE", "STRUCTURE", signal, weight, {}, is_missing)

    def _volume_institutional(self, regime: RegimeInfo) -> Dict[str, Any]:
        vol = self._sf(self.tv_data.get("volume_ratio"), 1.0)
        return {
            "volume_ratio":          vol,
            "volume_confirms_trend": (vol > self.VOLUME_HIGH_RATIO and
                                      regime.regime in ("BULL_TREND", "BEAR_TREND")),
            "very_high_volume":      vol > 1.8,
            "low_volume":            vol < self.VOLUME_LOW_RATIO,
        }

    def _detect_breakout_fakeout(self, regime: RegimeInfo) -> Dict[str, Any]:
        return {
            "fakeout_detected":      self._sb(self.tv_data.get("fakeout_detected")),
            "fakeout_bull":          self._sb(self.tv_data.get("fakeout_bull")),
            "fakeout_bear":          self._sb(self.tv_data.get("fakeout_bear")),
            "failed_breakout_above": self._sb(self.tv_data.get("failed_breakout_above")),
            "failed_breakout_below": self._sb(self.tv_data.get("failed_breakout_below")),
            "liquidity_sweep_bull":  self._sb(self.tv_data.get("liquidity_sweep_bull")),
            "liquidity_sweep_bear":  self._sb(self.tv_data.get("liquidity_sweep_bear")),
            "bear_pin":              self._sb(self.tv_data.get("bear_pin")),
            "bull_pin":              self._sb(self.tv_data.get("bull_pin")),
        }

    def _compute_exhaustion_from_indicators(self, regime: RegimeInfo) -> float:
        # This is now supplemented by the externally provided exhaustion scores.
        # We keep it as a fallback but mainly rely on the scores from tv_data.
        score = 0.0
        rsi = self._sf(self.twelve_data.get("rsi"), 50.0)
        if regime.regime == "BULL_TREND" and rsi > 75:
            score += 2.5
        elif regime.regime == "BULL_TREND" and rsi > 70:
            score += 1.5
        elif regime.regime == "BEAR_TREND" and rsi < 25:
            score += 2.5
        elif regime.regime == "BEAR_TREND" and rsi < 30:
            score += 1.5
        stoch_k = self._sf(self.twelve_data.get("stoch_k"), 50.0)
        if regime.regime == "BULL_TREND" and stoch_k > 85:
            score += 1.5
        elif regime.regime == "BEAR_TREND" and stoch_k < 15:
            score += 1.5
        macd_hist = self._sf(self.twelve_data.get("macd_histogram"), 0.0)
        macd_line = self._sf(self.twelve_data.get("macd"), 0.0)
        if regime.regime == "BULL_TREND" and macd_hist > 0 and macd_line != 0 and macd_hist < macd_line * 0.3:
            score += 1.5
        elif regime.regime == "BEAR_TREND" and macd_hist < 0 and macd_line != 0 and abs(macd_hist) < abs(macd_line) * 0.3:
            score += 1.5
        price    = self._sf(self.tv_data.get("price"), 0.0)
        bb_upper = self._sf(self.twelve_data.get("bb_upper"), 0.0)
        bb_lower = self._sf(self.twelve_data.get("bb_lower"), 0.0)
        if bb_upper > 0 and price > bb_upper * 1.01:
            score += 1.5
        elif bb_lower > 0 and price < bb_lower * 0.99:
            score += 1.5
        adx = self._sf(self.tv_data.get("adx"), 0.0)
        if adx > 40:
            score += 1.0
        return min(score, 10.0)

    def _detect_reversal_exhaustion(self, regime: RegimeInfo) -> Dict[str, Any]:
        # FIX 3: restored try/except safety net (was in v5.5, dropped in v5.6)
        try:
            bs_pine  = self._sf(self.tv_data.get("bearish_exhaustion_score"), 0.0)
            bus_pine = self._sf(self.tv_data.get("bullish_exhaustion_score"), 0.0)
            # Fallback to computed if both are zero
            if bs_pine == 0.0 and bus_pine == 0.0:
                computed = self._compute_exhaustion_from_indicators(regime)
                if regime.regime == "BULL_TREND":
                    bus = computed
                    bs  = 0.0
                elif regime.regime == "BEAR_TREND":
                    bs  = computed
                    bus = 0.0
                else:
                    bs  = computed * 0.5
                    bus = computed * 0.5
            else:
                bs  = bs_pine
                bus = bus_pine
        except Exception as e:
            print(f"[COMPRESSOR] _detect_reversal_exhaustion error: {e}")
            bs  = 0.0
            bus = 0.0

        if bus > bs:
            direction, score = "BULLISH_REVERSAL_RISK", bus
        elif bs > bus:
            direction, score = "BEARISH_REVERSAL_RISK", bs
        else:
            direction, score = "NEUTRAL_REVERSAL_RISK", 0

        if   score >= 7: verdict, override = "STRONG_REVERSAL_RISK",  "NO_TRADE"
        elif score >= 5: verdict, override = "WAIT_CONFIRMATION",      "WAIT_CONFIRMATION"
        elif score >= 3: verdict, override = "MODERATE_REVERSAL_RISK", "REDUCE_SIZE"
        else:            verdict, override = "CONTINUATION",           "NONE"

        return {
            "verdict":         verdict,
            "action_override": override,
            "risk_score":      score,
            "direction":       direction,
            "flags": {
                "bearish_rsi_div":   self._sb(self.tv_data.get("bearish_rsi_div")),
                "bullish_rsi_div":   self._sb(self.tv_data.get("bullish_rsi_div")),
                "bearish_macd_div":  self._sb(self.tv_data.get("bearish_macd_div")),
                "bullish_macd_div":  self._sb(self.tv_data.get("bullish_macd_div")),
                "overextended":      self._sb(self.tv_data.get("overextended")),
                "volume_absorption": self._sb(self.tv_data.get("volume_absorption")),
            }
        }

    def _detect_squeeze(self, regime: RegimeInfo) -> Dict[str, Any]:
        bbw        = self._sf(self.twelve_data.get("bb_width"), 100.0)
        price      = self._sf(self.tv_data.get("price"), 0.0)
        vol_ratio  = self._sf(self.tv_data.get("volume_ratio"), 1.0)
        bb_upper   = self._sf(self.twelve_data.get("bb_upper"), 0.0)
        bb_lower   = self._sf(self.twelve_data.get("bb_lower"), 0.0)
        # FIX: same percentile-based recalibration as _detect_regime — bbw's
        # absolute value is meaningless without knowing the symbol's normal
        # range, so prefer the self-calibrated percentile when available.
        bbw_pctile_raw = self.tv_data.get("bb_width_percentile")
        if bbw_pctile_raw is not None and not self._is_missing(bbw_pctile_raw):
            bbw_pctile = self._sf(bbw_pctile_raw, 50.0)
            is_squeeze = bbw_pctile < self.BB_WIDTH_SQUEEZE_PERCENTILE
        else:
            is_squeeze = bbw < self.BB_COMPRESSION_THRESHOLD
        if is_squeeze and price > 0:
            vol_confirms = vol_ratio > self.VOLUME_BREAKOUT_THRESHOLD
            if bb_upper > 0 and price > bb_upper and vol_confirms:
                direction = "BULLISH"
            elif bb_lower > 0 and price < bb_lower and vol_confirms:
                direction = "BEARISH"
            else:
                direction = "PENDING"
        else:
            direction = "NONE"
        return {
            "is_squeeze":         is_squeeze,
            "breakout_direction": direction,
            "bb_width":           bbw,
        }
    
    def _cross_validate(self, results: List[IndicatorResult],
                        regime: RegimeInfo) -> Tuple[Dict[str, Any], int, int]:
        tier_map = {
            "STRUCTURE": "TIER1", "VOLUME_PROFILE": "TIER1",
            "EMA": "TIER2", "VWAP": "TIER2", "MACD": "TIER2", "ADX_DI": "TIER2",
            "RSI": "TIER3", "STOCHASTIC": "TIER3", "BOLLINGER": "TIER3",
            "PIVOT": "TIER3", "ICHIMOKU": "TIER3", "VOLUME": "TIER3",
        }
        tier_weights = {"TIER1": 3.0, "TIER2": 2.0, "TIER3": 1.0}
        w_bull = w_bear = w_neut = 0.0
        missing_count = 0
        for r in results:
            if r.is_missing:
                missing_count += 1
                continue
            tw = tier_weights.get(tier_map.get(r.indicator, "TIER3"), 1.0)
            w  = r.weight * tw
            if r.signal   == "BULLISH": w_bull += w
            elif r.signal == "BEARISH": w_bear += w
            else:                       w_neut += w
        total = w_bull + w_bear + w_neut
        total_expected = len(results)
        return {
            "weighted_bull":      w_bull,
            "weighted_bear":      w_bear,
            "weighted_neutral":   w_neut,
            "directional_weight": w_bull + w_bear,
            "total_weighted":     total,
            "missing_count":      missing_count,
            "total_expected":     total_expected,
        }, missing_count, total_expected

    def multi_indicator_alignment(self, results: List[IndicatorResult], missing_count: int) -> Dict[str, Any]:
        cat_map = {
            "STRUCTURE": "STRUCTURE", "EMA": "TREND", "VWAP": "STRUCTURE",
            "ADX_DI": "TREND", "MACD": "MOMENTUM", "RSI": "MOMENTUM",
            "STOCHASTIC": "MOMENTUM", "BOLLINGER": "VOLATILITY",
            "PIVOT": "STRUCTURE", "VOLUME": "VOLUME",
            "ICHIMOKU": "TREND", "VOLUME_PROFILE": "STRUCTURE",
        }
        cats: Dict[str, Dict] = {}
        for r in results:
            if r.is_missing:
                continue
            cat = cat_map.get(r.indicator, "OTHER")
            cats.setdefault(cat, {"BULL": 0.0, "BEAR": 0.0, "NEUTRAL": 0.0, "active": False})
            if r.signal   == "BULLISH": cats[cat]["BULL"] += r.weight
            elif r.signal == "BEARISH": cats[cat]["BEAR"] += r.weight
            else:                       cats[cat]["NEUTRAL"] += r.weight
            cats[cat]["active"] = True
        bull_cats = bear_cats = 0
        active_cats = 0
        for cat, d in cats.items():
            if not d["active"]:
                continue
            active_cats += 1
            if d["BULL"] > d["BEAR"]:   bull_cats += 1
            elif d["BEAR"] > d["BULL"]: bear_cats += 1
        if active_cats < self.MIN_ACTIVE_CATEGORIES:
            if active_cats >= 2:
                status = "PARTIAL_ALIGNMENT"
                pct = 50.0
            else:
                status = "CONFLICTED"
                pct = 30.0
            majority = "NEUTRAL"
        else:
            if bull_cats > bear_cats:
                majority, agreeing, total = "BULLISH", bull_cats, bull_cats + bear_cats
            elif bear_cats > bull_cats:
                majority, agreeing, total = "BEARISH", bear_cats, bull_cats + bear_cats
            else:
                majority, agreeing, total = "NEUTRAL", 0, bull_cats + bear_cats
            pct = (agreeing / total * 100) if total > 0 else 50.0
            if   pct >= self.ALIGNMENT_FULL   * 100: status = "FULL_ALIGNMENT"
            elif pct >= self.ALIGNMENT_STRONG * 100: status = "STRONG_ALIGNMENT"
            elif pct >= self.ALIGNMENT_PARTIAL * 100: status = "PARTIAL_ALIGNMENT"
            else:                                      status = "CONFLICTED"
        return {
            "status":             status,
            "alignment_pct":      round(pct, 1),
            "majority_direction": majority,
            "bull_categories":    bull_cats,
            "bear_categories":    bear_cats,
            "active_categories":  active_cats,
            "min_required":       self.MIN_ACTIVE_CATEGORIES,
        }

    def _disagreement_penalty(self, analyses: List[IndicatorResult]) -> float:
        active = [r for r in analyses if not r.is_missing]
        sigs = {r.indicator: r.signal for r in active}
        struct = sigs.get("STRUCTURE", "NEUTRAL")
        ema    = sigs.get("EMA",       "NEUTRAL")
        rsi    = sigs.get("RSI",       "NEUTRAL")
        penalty = 0.0
        if (struct not in ("NEUTRAL",) and ema not in ("NEUTRAL",) and struct != ema):
            penalty += 0.12
        if (rsi not in ("NEUTRAL",) and ema not in ("NEUTRAL",) and rsi != ema):
            penalty += 0.05
        return min(penalty, 0.20)

    def _make_decision(self, regime: RegimeInfo,
                       cross_val: Dict, alignment: Dict,
                       squeeze: Dict, exhaustion: Dict,
                       breakout: Dict, analyses: List[IndicatorResult],
                       missing_ratio: float) -> Dict[str, Any]:
        w_bull  = cross_val["weighted_bull"]
        w_bear  = cross_val["weighted_bear"]
        dir_w   = cross_val["directional_weight"]
        total_w = cross_val["total_weighted"]
        neut_w  = cross_val["weighted_neutral"]
        net     = w_bull - w_bear
        base_bias = "BULLISH" if net > 0 else "BEARISH" if net < 0 else "NEUTRAL"
        base_strength = abs(net) / max(dir_w, 1e-9)
        raw_conv  = abs(net) / dir_w   if dir_w   > 0 else 0.0
        tot_conv  = abs(net) / total_w if total_w > 0 else 0.0
        conviction = raw_conv * 0.70 + tot_conv * 0.30
        if total_w > 0:
            neut_ratio  = neut_w / total_w
            neut_factor = 1.0 - min(0.25, neut_ratio * 0.20)
            conviction  = self._clamp(conviction * neut_factor)
        missing_penalty = missing_ratio * 0.35
        total_penalty = missing_penalty
        disagreement_penalty = self._disagreement_penalty(analyses)
        total_penalty += disagreement_penalty
        trend_penalty = 0.0
        if (regime.regime in ("BULL_TREND", "BEAR_TREND") and
            ((regime.regime == "BULL_TREND" and net < 0) or
             (regime.regime == "BEAR_TREND" and net > 0))):
            trend_penalty = 0.10
            total_penalty += trend_penalty
        total_penalty = min(total_penalty, self.MAX_TOTAL_PENALTY)
        conviction = self._clamp(conviction * (1.0 - total_penalty))
        alignment_pct = alignment["alignment_pct"]
        alignment_factor = 0.85 + (alignment_pct / 100.0 * 0.15)
        conviction = self._clamp(conviction * alignment_factor)
        vol_info = self._volume_institutional(regime)
        if vol_info["volume_confirms_trend"]:
            conviction = self._clamp(conviction + 0.06)
        bias = base_bias
        if bias == "NEUTRAL" and net != 0:
            bias = "NEUTRAL"
        conv_pct = conviction * 100
        def _size_action(b: str, pct: float) -> str:
            if b == "NEUTRAL" or pct < self.CONV_NO_TRADE: return "NO_TRADE"
            prefix = "BUY" if b == "BULLISH" else "SELL"
            if   pct < self.CONV_PROBE:  return f"{prefix}_PROBE"
            elif pct < self.CONV_SMALL:  return f"{prefix}_SMALL"
            elif pct < self.CONV_NORMAL: return f"{prefix}_NORMAL"
            else:                        return f"{prefix}_FULL"
        action = _size_action(bias, conv_pct)
        reasoning = []
        price = self._sf(self.tv_data.get("price"), 0.0)
        r1    = self._sf(self.tv_data.get("weekly_r1"), 0.0)
        s1    = self._sf(self.tv_data.get("weekly_s1"), 0.0)
        nearest = self._ss(self.tv_data.get("nearest_level"), "")
        near_resistance = (nearest in ("R1","R2","R3") and r1 > 0 and price > 0 and
                           abs(price - r1) / price * 100 <= self.LEVEL_PROXIMITY_PCT)
        near_support    = (nearest in ("S1","S2","S3") and s1 > 0 and price > 0 and
                           abs(price - s1) / price * 100 <= self.LEVEL_PROXIMITY_PCT)
        if (bias == "BULLISH" and near_resistance) or (bias == "BEARISH" and near_support):
            conviction  = self._clamp(conviction * 0.85)
            conv_pct    = conviction * 100
            action      = _size_action(bias, conv_pct)
            reasoning.append(f"Price near {nearest} — conviction reduced to {conv_pct:.0f}%.")
        if squeeze["is_squeeze"] and squeeze["breakout_direction"] == "PENDING":
            if alignment["status"] == "FULL_ALIGNMENT" and conv_pct > 65:
                action = "PREPARE_BREAKOUT"
                reasoning.append("Strong alignment in compression — prepare for breakout.")
            else:
                action = "WAIT_FOR_BREAKOUT"
                reasoning.append("Market compressed — wait for directional breakout.")
        SIZE_DOWN = {
            "FULL": "NORMAL", "NORMAL": "SMALL", "SMALL": "PROBE", "PROBE": "NO_TRADE"
        }
        rev_override = exhaustion["action_override"]
        if rev_override == "NO_TRADE":
            action = "NO_TRADE"
            reasoning.append(f"Strong reversal risk (score {exhaustion['risk_score']}) — {exhaustion['direction']}.")
        elif rev_override == "WAIT_CONFIRMATION":
            # FIX (carried from v5.6): always trigger the confirmation warning,
            # regardless of conviction level.
            action = "WAIT_CONFIRMATION"
            reasoning.append(f"Moderate reversal risk (score {exhaustion['risk_score']}) — wait for confirmation.")
        elif rev_override == "REDUCE_SIZE":
            if "_" in action:
                prefix, suffix = action.rsplit("_", 1)
                new_suffix = SIZE_DOWN.get(suffix, "NO_TRADE")
                action = f"{prefix}_{new_suffix}" if new_suffix != "NO_TRADE" else "NO_TRADE"
            reasoning.append(f"Reversal risk (score {exhaustion['risk_score']}) — position size reduced.")
        if (breakout["fakeout_detected"] and
                action not in ("NO_TRADE", "WAIT_FOR_BREAKOUT",
                               "PREPARE_BREAKOUT", "WAIT_CONFIRMATION")):
            action = "NO_TRADE"
            reasoning.append("Fakeout detected — stay out.")
        if not reasoning:
            reasoning.append(f"{alignment['status']} ({alignment_pct:.0f}% alignment) — conviction {conv_pct:.1f}% — {bias}.")
            if vol_info["volume_confirms_trend"]:
                reasoning.append("Institutional volume confirms direction.")
            if exhaustion["verdict"] == "CONTINUATION":
                reasoning.append("No major exhaustion detected.")
        if missing_ratio > 0.3:
            reasoning.insert(0, f"⚠️ {cross_val['missing_count']}/{cross_val['total_expected']} indicators missing — conviction penalized by {missing_penalty*100:.0f}%.")
        return {
            "bias":                bias,
            "base_bias":           base_bias,
            "base_strength":       round(base_strength, 3),
            "conviction_pct":      round(conv_pct, 1),
            "action":              action,
            "total_penalty_applied": round(total_penalty * 100, 1),
            "penalty_breakdown": {
                "missing": round(missing_penalty * 100, 1),
                "disagreement": round(disagreement_penalty * 100, 1),
                "trend_conflict": round(trend_penalty * 100, 1),
                "max_allowed": self.MAX_TOTAL_PENALTY * 100,
            },
            "near_resistance":     near_resistance,
            "near_support":        near_support,
            "nearest_level":       nearest,
            "reversal_risk_score": exhaustion["risk_score"],
            "reversal_risk_dir":   exhaustion["direction"],
            "reasoning":           reasoning,
        }

    @staticmethod
    def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    def compress(self) -> Dict[str, Any]:
        regime   = self._detect_regime()
        analyses = [
            self._analyze_structure(regime),
            self._analyze_ema(regime),
            self._analyze_vwap(regime),
            self._analyze_adx_di(regime),
            self._analyze_macd(regime),
            self._analyze_rsi(regime),
            self._analyze_stochastic(regime),
            self._analyze_bollinger_bands(regime),
            self._analyze_pivot(regime),
            self._analyze_volume(regime),
            self._analyze_ichimoku(regime),
            self._analyze_volume_profile(regime),
        ]
        cross_val, missing_count, total_expected = self._cross_validate(analyses, regime)
        missing_ratio = missing_count / total_expected if total_expected > 0 else 0.0
        squeeze       = self._detect_squeeze(regime)
        alignment     = self.multi_indicator_alignment(analyses, missing_count)
        exhaustion    = self._detect_reversal_exhaustion(regime)
        breakout      = self._detect_breakout_fakeout(regime)
        vol_info      = self._volume_institutional(regime)
        decision      = self._make_decision(regime, cross_val, alignment, squeeze,
                                            exhaustion, breakout, analyses, missing_ratio)
        lines = [
            "***MIKA COMPRESSOR v5.6.1 (Divergence + Exhaustion + Swing-Detection Fix)***",
            f"SYMBOL: {self._ss(self.tv_data.get('symbol'), 'UNKNOWN')} | "
            f"PRICE: {self._sf(self.tv_data.get('price'), 0.0):.5f} | "
            f"TF: {self._ss(self.tv_data.get('timeframe'), 'H1')}",
            "",
            f"REGIME: {regime.regime} | Mode: {regime.mode} | "
            f"Confidence: {regime.confidence:.0f}%",
            f"ADX: {regime.adx:.1f} | BB Width: {regime.bb_width:.2f}% | "
            f"ATR Pct: {regime.atr_percentile:.0f}%",
            "",
            f"DATA QUALITY: {missing_count}/{total_expected} indicators missing ({missing_ratio*100:.0f}%)",
            f"Active Categories: {alignment['active_categories']}/{alignment['min_required']} required",
            "",
            f"BASE BIAS: {decision['base_bias']} (strength={decision['base_strength']:.2f})",
            f"FINAL BIAS: {decision['bias']} | Conviction: {decision['conviction_pct']}%",
            f"TOTAL PENALTY: {decision['total_penalty_applied']}% (max {self.MAX_TOTAL_PENALTY*100:.0f}%)",
            f"  • Missing: {decision['penalty_breakdown']['missing']}%",
            f"  • Disagreement: {decision['penalty_breakdown']['disagreement']}%",
            f"  • Trend conflict: {decision['penalty_breakdown']['trend_conflict']}%",
            f"ALIGNMENT: {alignment['status']} ({alignment['alignment_pct']:.0f}%) → factor={0.85 + alignment['alignment_pct']/100*0.15:.2f}x",
            f"  Bull cats: {alignment['bull_categories']} / Bear cats: {alignment['bear_categories']}",
            "",
            "INDICATOR VOTES:",
        ]
        for r in analyses:
            emoji = "🟢" if r.signal == "BULLISH" else "🔴" if r.signal == "BEARISH" else "⚪"
            missing_flag = " ❌ MISSING" if r.is_missing else ""
            lines.append(f"  {emoji} {r.indicator:<16} {r.signal:<8} w={r.weight:.2f}{missing_flag}")
        lines += [
            "",
            f"VOLUME: ratio={vol_info['volume_ratio']:.2f}x | "
            f"confirms_trend={vol_info['volume_confirms_trend']} | "
            f"low={vol_info['low_volume']}",
            f"SQUEEZE: {squeeze['is_squeeze']} | "
            f"direction={squeeze['breakout_direction']} | "
            f"BB width={squeeze['bb_width']:.2f}%",
            f"EXHAUSTION: {exhaustion['verdict']} (score={exhaustion['risk_score']:.0f}) | "
            f"override={exhaustion['action_override']}",
            f"FAKEOUT: {breakout['fakeout_detected']} | "
            f"bull_trap={breakout['fakeout_bull']} | "
            f"bear_trap={breakout['fakeout_bear']}",
            "",
            f"DECISION: {decision['action']}",
        ]
        for r in decision["reasoning"]:
            lines.append(f"  • {r}")
        return {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "symbol":     self._ss(self.tv_data.get("symbol"), "UNKNOWN"),
            "price":      self._sf(self.tv_data.get("price"), 0.0),
            "timeframe":  self._ss(self.tv_data.get("timeframe"), "H1"),
            "version":    "5.6.1",
            "regime": {
                "regime":       regime.regime,
                "mode":         regime.mode,
                "confidence":   round(regime.confidence, 1),
                "adx":          round(regime.adx, 1),
                "bb_width":     round(regime.bb_width, 2),
                "atr_percentile": round(regime.atr_percentile, 1),
            },
            "data_quality": {
                "missing_count": missing_count,
                "total_expected": total_expected,
                "missing_ratio": round(missing_ratio, 2),
            },
            "signals": {
                "bullish_score": round(cross_val["weighted_bull"], 2),
                "bearish_score": round(cross_val["weighted_bear"], 2),
                "net_score":     round(cross_val["weighted_bull"] - cross_val["weighted_bear"], 2),
                "conviction_pct": decision["conviction_pct"],
                "bias":           decision["bias"],
                "base_bias":      decision["base_bias"],
                "base_strength":  decision["base_strength"],
            },
            "volume":             vol_info,
            "breakout":           breakout,
            "alignment":          alignment,
            "squeeze":            squeeze,
            "exhaustion":         exhaustion,
            "decision":           decision,
            "summary_text":       "\n".join(lines),
        }


# =================================================================
# compute_indicators_for_symbol – pure pandas/numpy
# =================================================================
def compute_indicators_for_symbol(sym_df: pd.DataFrame,
                                   divergence_lookback: int = 40) -> Dict[str, Any]:
    """
    Compute all indicators needed by MIKA Compressor, including divergence flags
    and exhaustion scores.

    FIX (v5.6.1): divergence_lookback widened default 20 -> 40, and swing
    detection rewritten as TRAILING (right-aligned) instead of centered,
    so the current/live bar CAN be classified as a swing high/low.
    """
    close = sym_df["Close"].squeeze()
    high = sym_df["High"].squeeze()
    low = sym_df["Low"].squeeze()
    volume = sym_df["Volume"].squeeze()
    price = close.iloc[-1]

    # helper EMA
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    if math.isnan(rsi): rsi = float('nan')
    # Store full RSI series for divergence
    rsi_series = 100 - (100 / (1 + rs))

    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    macd_line_val = macd_line.iloc[-1]
    signal_line_val = signal_line.iloc[-1]
    macd_hist_val = macd_hist.iloc[-1]
    for v in [macd_line_val, signal_line_val, macd_hist_val]:
        if math.isnan(v): v = float('nan')
    # Store MACD line series for divergence
    macd_line_series = macd_line

    # Bollinger Bands
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / sma20 * 100
    bb_upper_val = bb_upper.iloc[-1]
    bb_lower_val = bb_lower.iloc[-1]
    bb_width_val = bb_width.iloc[-1]
    for v in [bb_upper_val, bb_lower_val, bb_width_val]:
        if math.isnan(v): v = float('nan')

    # FIX: bb_width_percentile — self-calibrating squeeze/expansion signal.
    # Ranks the current bb_width against this symbol's own trailing history
    # instead of a fixed absolute % threshold, which broke because raw
    # bb_width scale varies enormously by symbol/timeframe (e.g. EURUSD 4h
    # typically 0.02%-1.5%, but the old thresholds assumed 5%-15%).
    BBW_PCTILE_LOOKBACK = 150
    bbw_hist = bb_width.iloc[-BBW_PCTILE_LOOKBACK:].dropna()
    if len(bbw_hist) >= 10 and not math.isnan(bb_width_val):
        bb_width_percentile = float((bbw_hist < bb_width_val).sum()) / len(bbw_hist) * 100.0
    else:
        bb_width_percentile = 50.0  # neutral fallback when insufficient history

    # EMAs
    ema21 = ema(close, 21).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    ema200 = ema(close, 200).iloc[-1]

    ema_vals_valid = not any(math.isnan(v) for v in (ema21, ema50, ema200))
    if ema_vals_valid and ema21 > ema50 > ema200:
        ema_align = "BULL"
        structure_bias = "BULL"
    elif ema_vals_valid and ema21 < ema50 < ema200:
        ema_align = "BEAR"
        structure_bias = "BEAR"
    elif ema_vals_valid:
        ema_align = "NEUTRAL"
        structure_bias = "NEUTRAL"
    else:
        ema_align = "N/A"
        structure_bias = "N/A"

    # ADX (Wilder's, vectorized)
    high_shift = high.shift(1)
    low_shift = low.shift(1)
    tr1 = high - low
    tr2 = (high - high_shift).abs()
    tr3 = (low - low_shift).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    plus_dm = pd.Series(index=high.index, dtype=float)
    minus_dm = pd.Series(index=high.index, dtype=float)
    for i in range(1, len(high)):
        up = high.iloc[i] - high.iloc[i-1]
        down = low.iloc[i-1] - low.iloc[i]
        plus_dm.iloc[i] = up if (up > down and up > 0) else 0.0
        minus_dm.iloc[i] = down if (down > up and down > 0) else 0.0

    diplus_series = (plus_dm.rolling(14).mean() / atr.replace(0, np.nan) * 100)
    diminus_series = (minus_dm.rolling(14).mean() / atr.replace(0, np.nan) * 100)
    di_sum = (diplus_series + diminus_series).replace(0, np.nan)
    dx_series = (diplus_series - diminus_series).abs() / di_sum * 100
    adx_series = dx_series.rolling(14).mean()

    adx = adx_series.iloc[-1] if not adx_series.empty else 0
    diplus = diplus_series.iloc[-1] if not diplus_series.empty else 0
    diminus = diminus_series.iloc[-1] if not diminus_series.empty else 0
    if isinstance(adx, float) and math.isnan(adx): adx = 0
    if isinstance(diplus, float) and math.isnan(diplus): diplus = 0
    if isinstance(diminus, float) and math.isnan(diminus): diminus = 0

    # Stochastic
    low14 = low.rolling(window=14).min()
    high14 = high.rolling(window=14).max()
    stoch_k = 100 * ((close - low14) / (high14 - low14))
    stoch_d = stoch_k.rolling(window=3).mean()
    stoch_k_val = stoch_k.iloc[-1]
    stoch_d_val = stoch_d.iloc[-1]
    for v in [stoch_k_val, stoch_d_val]:
        if math.isnan(v): v = float('nan')

    # Volume ratio
    avg_vol = volume.rolling(window=20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / avg_vol if avg_vol and avg_vol > 0 else 1.0
    raw_signal = 1 if price > close.iloc[-2] else -1 if price < close.iloc[-2] else 0

    # ---------- FIX: Divergence detection (trailing / right-aligned swings) ----------
    def _detect_divergence(price_series, indicator_series, window=divergence_lookback,
                            swing_bars=3):
        """
        Returns (bearish_div, bullish_div) booleans.
        Bearish: price makes higher high, indicator makes lower high.
        Bullish: price makes lower low, indicator makes higher low.

        FIX (v5.6.1): Uses a TRAILING swing definition instead of a centered
        one. A bar `i` is a swing high if it is the max of bars
        [i - swing_bars + 1, i] (i.e. it's the highest close of itself and
        the `swing_bars - 1` bars immediately before it). This lets the
        LAST bar in the dataset (the current/live bar) qualify as a swing,
        which a centered rolling window can never do.
        """
        n = len(price_series)
        if n < window:
            window = n

        # Trailing (right-aligned) rolling max/min: bar i compared only to
        # bars at or before it -> works on the final/live bar.
        price_trail_high = price_series.rolling(swing_bars).max()
        price_trail_low  = price_series.rolling(swing_bars).min()
        ind_trail_high   = indicator_series.rolling(swing_bars).max()
        ind_trail_low    = indicator_series.rolling(swing_bars).min()

        price_highs = (price_series == price_trail_high)
        price_lows  = (price_series == price_trail_low)
        ind_highs   = (indicator_series == ind_trail_high)
        ind_lows    = (indicator_series == ind_trail_low)

        idx = price_series.index[-window:]
        price_high_idx = idx[price_highs.reindex(idx).fillna(False)]
        price_low_idx  = idx[price_lows.reindex(idx).fillna(False)]
        ind_high_idx   = idx[ind_highs.reindex(idx).fillna(False)]
        ind_low_idx    = idx[ind_lows.reindex(idx).fillna(False)]

        bearish = False
        bullish = False

        # Bearish divergence: price higher high, indicator lower high
        if len(price_high_idx) >= 2 and len(ind_high_idx) >= 2:
            ph1, ph2 = price_high_idx[-2], price_high_idx[-1]
            ih1, ih2 = ind_high_idx[-2], ind_high_idx[-1]
            if ih2 >= ph1 and ih1 <= ph2 and ph2 > ph1:
                if (price_series[ph2] > price_series[ph1] and
                    indicator_series[ih2] < indicator_series[ih1]):
                    bearish = True

        # Bullish divergence: price lower low, indicator higher low
        if len(price_low_idx) >= 2 and len(ind_low_idx) >= 2:
            pl1, pl2 = price_low_idx[-2], price_low_idx[-1]
            il1, il2 = ind_low_idx[-2], ind_low_idx[-1]
            if il2 >= pl1 and il1 <= pl2 and pl2 > pl1:
                if (price_series[pl2] < price_series[pl1] and
                    indicator_series[il2] > indicator_series[il1]):
                    bullish = True

        return bearish, bullish

    # Compute divergences
    rsi_bear_div, rsi_bull_div = _detect_divergence(close, rsi_series)
    macd_bear_div, macd_bull_div = _detect_divergence(close, macd_line_series)

    # ---------- Compute exhaustion scores ----------
    bullish_exhaustion_score = 0.0
    if not math.isnan(rsi_series.iloc[-1]) and rsi_series.iloc[-1] > 70:
        bullish_exhaustion_score += 2.0
    if not math.isnan(stoch_k_val) and stoch_k_val > 80:
        bullish_exhaustion_score += 1.5
    if not math.isnan(macd_hist_val) and macd_hist_val > 0 and macd_line_val != 0:
        if abs(macd_hist_val) < abs(macd_line_val) * 0.3:  # histogram shrinking
            bullish_exhaustion_score += 1.5
    if not math.isnan(bb_upper_val) and price > bb_upper_val * 1.01:
        bullish_exhaustion_score += 1.5
    if rsi_bear_div:
        bullish_exhaustion_score += 2.0
    if macd_bear_div:
        bullish_exhaustion_score += 1.5
    bullish_exhaustion_score = min(bullish_exhaustion_score, 10.0)

    bearish_exhaustion_score = 0.0
    if not math.isnan(rsi_series.iloc[-1]) and rsi_series.iloc[-1] < 30:
        bearish_exhaustion_score += 2.0
    if not math.isnan(stoch_k_val) and stoch_k_val < 20:
        bearish_exhaustion_score += 1.5
    if not math.isnan(macd_hist_val) and macd_hist_val < 0 and macd_line_val != 0:
        if abs(macd_hist_val) < abs(macd_line_val) * 0.3:
            bearish_exhaustion_score += 1.5
    if not math.isnan(bb_lower_val) and price < bb_lower_val * 0.99:
        bearish_exhaustion_score += 1.5
    if rsi_bull_div:
        bearish_exhaustion_score += 2.0
    if macd_bull_div:
        bearish_exhaustion_score += 1.5
    bearish_exhaustion_score = min(bearish_exhaustion_score, 10.0)

    overextended = (bullish_exhaustion_score > 5 or bearish_exhaustion_score > 5)

    def _na_if_nan(v):
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "N/A"
        except TypeError:
            pass
        return v

    webhook_payload = {
        "symbol": sym_df["Symbol"].iloc[0],
        "price": _na_if_nan(price),
        "timeframe": "4h",
        "adx": _na_if_nan(adx),
        "ema_21": _na_if_nan(ema21),
        "ema_50": _na_if_nan(ema50),
        "ema_200": _na_if_nan(ema200),
        "ema_align": ema_align,
        "ema_slope": "N/A",
        "weekly_vwap": "N/A",
        "monthly_vwap": "N/A",
        "price_vs_vwap": "N/A",
        "weekly_pivot": "N/A",
        "weekly_r1": "N/A",
        "weekly_s1": "N/A",
        "nearest_level": "N/A",
        "structure_bias": structure_bias,
        "hh_hl_pred": "N/A",
        "lon_fail_up": False,
        "lon_fail_down": False,
        "ny_fail_up": False,
        "ny_fail_down": False,
        "volume_ratio": _na_if_nan(vol_ratio),
        "raw_signal": raw_signal,
        "diplus": _na_if_nan(diplus),
        "diminus": _na_if_nan(diminus),
        "market_regime": "N/A",
        "bearish_exhaustion_score": bearish_exhaustion_score,
        "bullish_exhaustion_score": bullish_exhaustion_score,
        "bearish_rsi_div": rsi_bear_div,
        "bullish_rsi_div": rsi_bull_div,
        "bearish_macd_div": macd_bear_div,
        "bullish_macd_div": macd_bull_div,
        "overextended": overextended,
        "volume_absorption": False,
        "fakeout_detected": False,
        "fakeout_bull": False,
        "fakeout_bear": False,
        "failed_breakout_above": False,
        "failed_breakout_below": False,
        "liquidity_sweep_bull": False,
        "liquidity_sweep_bear": False,
        "bear_pin": False,
        "bull_pin": False,
        "atr_percentile": "N/A",
        "atr": "N/A",
        "bb_width": _na_if_nan(bb_width_val),
        "bb_width_percentile": round(bb_width_percentile, 1),
    }
    twelve_data = {
        "rsi": _na_if_nan(rsi),
        "stoch_k": _na_if_nan(stoch_k_val),
        "stoch_d": _na_if_nan(stoch_d_val),
        "macd": _na_if_nan(macd_line_val),
        "macd_signal": _na_if_nan(signal_line_val),
        "macd_histogram": _na_if_nan(macd_hist_val),
        "bb_upper": _na_if_nan(bb_upper_val),
        "bb_lower": _na_if_nan(bb_lower_val),
        "bb_width": _na_if_nan(bb_width_val),
        "ichimoku_conversion": "N/A",
        "ichimoku_base": "N/A",
        "ichimoku_span_a": "N/A",
        "ichimoku_span_b": "N/A",
    }
    return {"webhook": webhook_payload, "twelve": twelve_data}
