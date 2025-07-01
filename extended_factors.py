# Extended Factors for EquiLend Securities Lending Analysis
# Additional models using external data sources

import pandas as pd
import numpy as np
from typing import Optional
import os
import requests

class BorrowCDSBasis:
    """Borrow-CDS Basis - captures credit-equity dislocations"""
    
    def __init__(self, fee_col='Fee All (BPS)', cds_col='CDS_5Y'):
        self.fee_col = fee_col
        self.cds_col = cds_col
    
    def fetch_cds_data(self, start_date='2019-01-01'):
        """Fetch CDS data from FRED API"""
        try:
            from fredapi import Fred
            fred = Fred(api_key=os.getenv('FRED_KEY'))
            return fred.get_series('BAMLCC0A0CM', start=start_date).rename('CDS_5Y')
        except Exception:
            return pd.Series(dtype=float, name='CDS_5Y')
    
    def score(self, df: pd.DataFrame, cds_series: Optional[pd.Series] = None) -> pd.Series:
        """Calculate Borrow-CDS basis signal"""
        if cds_series is None:
            cds_series = self.fetch_cds_data()
        
        merged = df.join(cds_series, on='Date', how='inner')
        if merged.empty:
            return pd.Series(index=df.index, dtype=float)
            
        basis = merged[self.fee_col] - merged[self.cds_col]
        return (basis - basis.mean()) / basis.std()

class OptionsSkewDivergence:
    """Options Skew Divergence - spots mis-priced hedging"""
    
    def __init__(self, fee_col='Fee All (BPS)', util_col='Active Utilization (%)', skew_col='CBOE_Skew'):
        self.fee_col = fee_col
        self.util_col = util_col
        self.skew_col = skew_col
    
    def fetch_skew_data(self, start_date='2019-01-01'):
        """Fetch CBOE skew data from FRED"""
        try:
            from fredapi import Fred
            fred = Fred(api_key=os.getenv('FRED_KEY'))
            return fred.get_series('SKEW', start=start_date).rename('CBOE_Skew')
        except Exception:
            return pd.Series(dtype=float, name='CBOE_Skew')
    
    def score(self, df: pd.DataFrame, skew_series: Optional[pd.Series] = None) -> pd.Series:
        """Calculate skew divergence signal"""
        if skew_series is None:
            skew_series = self.fetch_skew_data()
        
        merged = df.join(skew_series, on='Date', how='inner')
        if merged.empty:
            return pd.Series(index=df.index, dtype=float)
        
        z_fee = (merged[self.fee_col] - merged[self.fee_col].mean()) / merged[self.fee_col].std()
        z_skew = (merged[self.skew_col] - merged[self.skew_col].mean()) / merged[self.skew_col].std()
        
        return z_fee - z_skew  # Positive when fee rising faster than skew

class ETFFlowPressure:
    """ETF Flow Pressure - identifies arbitrage strains"""
    
    def __init__(self, loan_col='On Loan Quantity'):
        self.loan_col = loan_col
    
    def fetch_etf_flows(self, etf_ticker: str, start_date: str):
        """Fetch ETF flow data (placeholder - requires actual data source)"""
        # This would connect to your preferred data source (Polygon, etc.)
        return pd.Series(dtype=float, name='ETF_Flows')
    
    def score(self, df: pd.DataFrame, etf_flows: Optional[pd.Series] = None) -> pd.Series:
        """Calculate ETF flow pressure"""
        if etf_flows is None:
            return pd.Series(index=df.index, dtype=float)
        
        merged = df.join(etf_flows, on='Date', how='inner')
        if merged.empty:
            return pd.Series(index=df.index, dtype=float)
        
        loan_change = merged[self.loan_col].pct_change()
        flow_pressure = loan_change - merged['ETF_Flows']
        
        return (flow_pressure - flow_pressure.mean()) / flow_pressure.std()

class MacroLiquidityStress:
    """Macro Liquidity Stress - systemic stress overlay"""
    
    def __init__(self, util_col='Active Utilization (%)'):
        self.util_col = util_col
    
    def fetch_stress_data(self, start_date='2019-01-01'):
        """Fetch financial stress index from FRED"""
        try:
            from fredapi import Fred
            fred = Fred(api_key=os.getenv('FRED_KEY'))
            return fred.get_series('STLFSI2', start=start_date).rename('Stress_Index')
        except Exception:
            return pd.Series(dtype=float, name='Stress_Index')
    
    def score(self, df: pd.DataFrame, stress_series: Optional[pd.Series] = None) -> pd.Series:
        """Calculate macro stress overlay"""
        if stress_series is None:
            stress_series = self.fetch_stress_data()
        
        merged = df.join(stress_series, on='Date', how='inner')
        if merged.empty:
            return pd.Series(index=df.index, dtype=float)
        
        return merged[self.util_col] * merged['Stress_Index']

class ESGConstraintGauge:
    """ESG Constraint Gauge - supply limits from ESG considerations"""
    
    def __init__(self, lender_count_col='Lender Count', esg_score_col='ESG_Score'):
        self.lender_count_col = lender_count_col
        self.esg_score_col = esg_score_col
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ESG constraint signal"""
        if self.esg_score_col not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        # Lower lender diversity + high ESG risk = higher constraint
        lender_diversity = 1 / df[self.lender_count_col]
        esg_risk = -df[self.esg_score_col]  # Assuming higher score = better ESG
        
        constraint = lender_diversity * esg_risk
        return (constraint - constraint.mean()) / constraint.std()

class CrowdBuzzPulse:
    """Crowd Buzz Pulse - retail-driven squeeze detection"""
    
    def __init__(self, reddit_col='Reddit_Mentions', twitter_col='Twitter_Mentions'):
        self.reddit_col = reddit_col
        self.twitter_col = twitter_col
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate crowd buzz intensity"""
        buzz_cols = [col for col in [self.reddit_col, self.twitter_col] if col in df.columns]
        
        if not buzz_cols:
            return pd.Series(index=df.index, dtype=float)
        
        total_buzz = df[buzz_cols].sum(axis=1)
        buzz_velocity = total_buzz.pct_change()
        
        return (buzz_velocity - buzz_velocity.mean()) / buzz_velocity.std()

class EnhancedShortSqueezeV4:
    """Enhanced Short Squeeze Prediction v4 - combines multiple signals"""
    
    def __init__(self):
        self.weights = {
            'SIM': 0.18,
            'BCS': 0.16, 
            'UPI': 0.14,
            'FTZ': 0.14,
            'LPF': 0.14,
            'Reddit_Buzz': 0.12,
            'Uptick_Flag': 0.12
        }
    
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate enhanced short squeeze score"""
        score = pd.Series(0.0, index=df.index)
        
        for factor, weight in self.weights.items():
            if factor in df.columns:
                factor_score = df[factor].fillna(0)
                score += weight * factor_score
        
        return score

def compute_extended_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all extended factors for a dataframe"""
    result = df.copy()
    
    # Initialize extended factor classes
    borrow_cds = BorrowCDSBasis()
    skew_div = OptionsSkewDivergence()
    etf_pressure = ETFFlowPressure()
    macro_stress = MacroLiquidityStress()
    esg_constraint = ESGConstraintGauge()
    crowd_buzz = CrowdBuzzPulse()
    ssr_v4 = EnhancedShortSqueezeV4()
    
    # Calculate extended factors
    try:
        result['Borrow_CDS_Basis'] = borrow_cds.score(df)
    except Exception:
        result['Borrow_CDS_Basis'] = np.nan
    
    try:
        result['Options_Skew_Div'] = skew_div.score(df)
    except Exception:
        result['Options_Skew_Div'] = np.nan
    
    try:
        result['ETF_Flow_Pressure'] = etf_pressure.score(df)
    except Exception:
        result['ETF_Flow_Pressure'] = np.nan
    
    try:
        result['Macro_Stress'] = macro_stress.score(df)
    except Exception:
        result['Macro_Stress'] = np.nan
    
    try:
        result['ESG_Constraint'] = esg_constraint.score(df)
    except Exception:
        result['ESG_Constraint'] = np.nan
    
    try:
        result['Crowd_Buzz'] = crowd_buzz.score(df)
    except Exception:
        result['Crowd_Buzz'] = np.nan
    
    try:
        result['SSR_v4'] = ssr_v4.score(df)
    except Exception:
        result['SSR_v4'] = np.nan
    
    return result
