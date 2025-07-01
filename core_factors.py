# EquiLend Core Factors
# Consolidated short squeeze and securities lending factors

import pandas as pd
import numpy as np
from typing import Optional, Union

class ShortInterestMomentum:
    """Short Interest Momentum (SIM) - tracks accelerating short build-up"""
    
    def __init__(self, loan_col='On Loan Quantity Month Diff', fee_col='Fee All Month Diff (BPS)'):
        self.loan_col = loan_col
        self.fee_col = fee_col
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate SIM z-score"""
        loan_momentum = df[self.loan_col].pct_change()
        fee_momentum = df[self.fee_col].pct_change()
        
        combined = (loan_momentum + fee_momentum) / 2
        return (combined - combined.mean()) / combined.std()

class BorrowCostShock:
    """Borrow Cost Shock (BCS) - detects sudden fee spikes"""
    
    def __init__(self, fee_col='Fee All (BPS)', window=30):
        self.fee_col = fee_col
        self.window = window
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate BCS z-score based on rolling volatility"""
        fees = df[self.fee_col]
        daily_change = fees.diff()
        rolling_std = daily_change.rolling(self.window).std()
        
        return daily_change / rolling_std

class UtilizationPersistence:
    """Utilization Persistence (UPI) - persistent tight supply indicator"""
    
    def __init__(self, util_col='Active Utilization (%)', threshold=95, window=20):
        self.util_col = util_col
        self.threshold = threshold
        self.window = window
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate UPI based on high utilization persistence"""
        high_util = (df[self.util_col] >= self.threshold).astype(int)
        persistence = high_util.rolling(self.window).mean()
        
        return (persistence - persistence.mean()) / persistence.std()

class FeeTrendZScore:
    """Fee Trend Z-Score (FTZ) - detects under-the-radar fee drifts"""
    
    def __init__(self, fee_col='Fee All (BPS)', window=20):
        self.fee_col = fee_col
        self.window = window
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate FTZ based on fee slope trend"""
        fees = df[self.fee_col]
        
        def rolling_slope(series):
            x = np.arange(len(series))
            return np.polyfit(x, series, 1)[0] if len(series) == self.window else np.nan
        
        slopes = fees.rolling(self.window).apply(rolling_slope, raw=True)
        return (slopes - slopes.mean()) / slopes.std()

class DaysToCoverZ:
    """Days-To-Cover Z-Score (DTC_z) - short covering pressure indicator"""
    
    def __init__(self, si_col='Short Interest', volume_col='Average Daily Volume'):
        self.si_col = si_col
        self.volume_col = volume_col
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate DTC z-score"""
        dtc = df[self.si_col] / df[self.volume_col]
        return (dtc - dtc.mean()) / dtc.std()

class LocateProxyFactor:
    """Locate Proxy Factor (LPF) - proxy for locate availability"""
    
    def __init__(self, rerate_col='Re-Rate Ratio', b2b_col='B2B Loans'):
        self.rerate_col = rerate_col
        self.b2b_col = b2b_col
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate LPF based on re-rate ratio and B2B activity"""
        rerate_z = (df[self.rerate_col] - df[self.rerate_col].mean()) / df[self.rerate_col].std()
        b2b_z = (df[self.b2b_col] - df[self.b2b_col].mean()) / df[self.b2b_col].std()
        
        return (rerate_z + b2b_z) / 2

# Helper function for z-score calculation
def _z_score(series: pd.Series) -> pd.Series:
    """Calculate z-score for a pandas Series"""
    return (series - series.mean()) / series.std()

def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all core factors for a dataframe"""
    result = df.copy()
    
    # Initialize factor classes
    sim = ShortInterestMomentum()
    bcs = BorrowCostShock()
    upi = UtilizationPersistence()
    ftz = FeeTrendZScore()
    dtc = DaysToCoverZ()
    lpf = LocateProxyFactor()
    
    # Calculate factors (with error handling)
    try:
        result['SIM'] = sim.score(df)
    except Exception:
        result['SIM'] = np.nan
        
    try:
        result['BCS'] = bcs.score(df)
    except Exception:
        result['BCS'] = np.nan
        
    try:
        result['UPI'] = upi.score(df)
    except Exception:
        result['UPI'] = np.nan
        
    try:
        result['FTZ'] = ftz.score(df)
    except Exception:
        result['FTZ'] = np.nan
        
    try:
        result['DTC_z'] = dtc.score(df)
    except Exception:
        result['DTC_z'] = np.nan
        
    try:
        result['LPF'] = lpf.score(df)
    except Exception:
        result['LPF'] = np.nan
    
    return result
