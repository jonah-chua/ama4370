 # Workspace Cleanup & Latest Implementation Summary

## Cleanup Steps
- Archived all scripts and notebooks not containing the keyword "latest" (except for `binance_collector.py` and `collate_trades.ipynb`).
- Moved old scripts to `archive/scripts` and old notebooks to `archive/notebooks`.
- Moved all markdown documentation to `archive/md` for review and cleanup.
- Deleted outdated markdowns referencing non-existent files (e.g., `README_MODULAR.md`).
- Edited documentation to clarify only `profitview_latest_bot.py` and `Latest_test.py` are maintained; older scripts are archived.

## Latest Files (Main Workspace)
- `Latest_test.py`: Main order block detection, backtesting, and visualization script. Implements Pine Script logic in Python, with full parameterization and plotting.
- `profitview_latest_bot.py`: Production trading bot with robust position management, trailing stops, dynamic loss limits, and live parameter update endpoint.
- `binance_collector.py`: Data collection utility for fetching OHLCV data from Binance.
- `collate_trades.ipynb`: Notebook for trade collation and analysis.

## Implementation Highlights
- **Order Block Detection**: Symmetric pivot logic, breakout confirmation, candidate candle selection, volumetric strength calculation, overlap handling, and violation tracking.
- **Position Management**: Dynamic stop loss/take profit, trailing stop activation and update logic, holding period exits, and risk-based sizing.
- **Live Parameter Updates**: `/update_params` endpoint for runtime tuning of all major strategy parameters.
- **Accurate PnL Tracking**: Hybrid approach using actual fill prices and fees when available, with fallback to estimates.
- **Robustness**: Defensive programming, exponential backoff for market unavailability, and comprehensive logging.
- Only latest, production-ready files remain in the main folder; all legacy code and docs are archived for reference.

## Current Steps
- Currently testing in Profitview.

---

**Status:** Workspace cleaned and up-to-date as of November 13, 2025.
