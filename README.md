# Bob_EquiLend_Models

A comprehensive suite of quantitative models for securities lending and short squeeze analysis.

## 🚀 Quick Start

1. **Open the development environment:**
   ```bash
   jupyter lab Notebooks/Factor_Development.ipynb
   ```

2. **For comprehensive analysis:**
   ```bash
   jupyter lab Notebooks/EquiLend_Consolidated_Playbook.ipynb
   ```

3. **Browse model documentation:**
   - See `Docs/` folder for detailed model specifications
   - Review branded model documentation for each factor

## 📁 Project Structure

```
Bob_EquiLend_Models/
├── 📊 Notebooks/                    # Interactive notebooks
│   ├── EquiLend_Consolidated_Playbook.ipynb  # Main analysis workflow
│   └── Factor_Development.ipynb     # Factor development & testing
├── 📋 Docs/                         # Documentation & specifications
│   ├── Model documentation (30+ Word docs)
│   ├── Daily digest templates
│   └── Technical specifications
└── 📖 README.md                     # Project documentation
```

### 🔧 **Current Development Status:**

**✅ Working Components:**
- Interactive notebooks with embedded factor classes
- Comprehensive model documentation
- Factor development environment

**⚠️ In Development:**
- Modular Python structure (factors currently defined inline)
- Automated production scripts
- Data pipeline integration

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

1. **Research & Development** → Use `Factor_Development.ipynb` for interactive analysis
2. **Comprehensive Analysis** → Use `EquiLend_Consolidated_Playbook.ipynb` for full workflow
3. **Documentation Review** → Browse `Docs/` for detailed model specifications
4. **Future Modularization** → Extract stable factors to Python modules

## 📊 Current Implementation

**Factor Development Notebook** includes embedded classes for:

```python
# Core factor classes (defined inline)
class ShortInterestMomentum:
    """Short Interest Momentum (SIM) - tracks accelerating short build-up"""
    
class BorrowCostShock:
    """Borrow Cost Shock (BCS) - detects sudden fee spikes"""

# Usage in notebook
sim = ShortInterestMomentum()
sim_scores = sim.score(data)
```

**Documentation Available:**
- 30+ Word documents with detailed model specifications
- Technical implementation guides
- Daily digest templates

## 🛠️ Dependencies

**Core Libraries (install as needed):**
- `pandas`, `numpy` - Data manipulation and analysis
- `matplotlib`, `seaborn` - Visualization and plotting
- `jupyter` - Interactive notebook environment
- `datetime` - Date/time handling (built-in)

**Installation:**
```bash
pip install pandas numpy matplotlib seaborn jupyter
```

**Optional Enhancements:**
- `scikit-learn` - Machine learning utilities
- `requests` - API data fetching
- `python-dotenv` - Environment management

## 📈 Current Status

The notebook-based development approach provides:

- ✅ Interactive analysis and visualization
- ✅ Rapid prototyping and testing
- ✅ Self-contained factor development
- ✅ Comprehensive documentation
- ⚠️ Ready for modularization when needed

## 🚀 Next Steps

1. **Extract stable factors** to Python modules
2. **Add production scripts** for automated analysis
3. **Implement data pipeline** for live data integration
4. **Create requirements.txt** for dependency management

---

**Last updated: July 2025**
