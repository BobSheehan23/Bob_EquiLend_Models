# Bob_EquiLend_Models

A comprehensive suite of quantitative models for securities lending and short squeeze analysis.

## 🚀 Quick Start

1. **Set up environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys in `.env`:**
   ```bash
   FRED_KEY=your_fred_api_key
   GEMINI_API_KEY=your_gemini_key
   POLYGON_KEY=your_polygon_key
   ```

3. **Run the main analysis:**
   ```bash
   jupyter lab nb/EquiLend_Consolidated_Playbook.ipynb
   ```

## 📁 Project Structure

```
Bob_EquiLend_Models/
├── 📊 nb/                           # Interactive notebooks
│   ├── EquiLend_Consolidated_Playbook.ipynb
│   └── Factor_Development.ipynb
├── 🐍 Python Modules/               # Reusable model code
│   ├── core_factors.py             # Core short squeeze factors
│   └── extended_factors.py         # Extended analysis models
├── 🏭 src/                          # Production scripts
│   └── daily_digest.py             # Automated report generation
├── 📋 docs/                         # Documentation
└── 🔧 data/                         # Data files (excluded from Git)
```

## 🧮 Available Models

### Core Factors
- **Short-Interest Momentum (SIM)** - Accelerating short build-up detection
- **Borrow Cost Shock (BCS)** - Sudden scarcity event identification
- **Utilization Persistence (UPI)** - Persistent tight supply monitoring
- **Fee Trend Z-Score (FTZ)** - Under-the-radar fee drift detection
- **Days-to-Cover Z (DTC_z)** - Short-covering pressure analysis
- **Locate Proxy Factor (LPF)** - Locate surge estimation

### Extended Models
- **Borrow-CDS Basis** - Credit-equity dislocation detection
- **Options Skew Divergence** - Hedge mis-pricing signals
- **ETF Flow Pressure** - Arbitrage strain monitoring
- **Macro Liquidity Stress** - Systemic stress overlay
- **ESG Constraint Gauge** - Supply limits from ESG mandates
- **Crowd Buzz Pulse** - Retail-driven squeeze detection
- **Enhanced Short Squeeze Prediction (SSR v4)** - Multi-factor squeeze scoring

## 🔄 Workflow

1. **Research & Development** → Use notebooks for interactive analysis
2. **Production Code** → Move stable functions to Python modules
3. **Daily Analysis** → Import modules into consolidated playbook
4. **Automation** → Use modules directly for scheduled processes

## 📊 Usage Examples

```python
# Import core factors
from core_factors import ShortInterestMomentum, BorrowCostShock

# Calculate factors
sim = ShortInterestMomentum()
sim_scores = sim.score(data)

# Generate daily digest
from daily_digest import generate_digest
digest = generate_digest(data)
```

## 🛠️ Dependencies

- pandas, numpy - Data manipulation
- matplotlib, seaborn - Visualization
- scikit-learn - Machine learning utilities
- requests - API data fetching
- python-dotenv - Environment management

## 📈 Performance

The hybrid notebook + module approach provides:
- ✅ Interactive analysis and visualization
- ✅ Clean, testable, reusable code
- ✅ Version control for production logic
- ✅ Automated execution capabilities

---

*Last updated: July 2025*
