# 🧠 AI Context Handover
**Last Updated:** 2026-02-02 15:00
**Status:** Stable (Optimization Complete)

## 🎯 Current Micro-Goal
Implemented system stability fixes to prevent the system from crashing when spread thresholds are low.

## 🏗 The "Mental Stack" (LIFO)
1. **System Stability Fixes**: Applied 60s cooldown to scanner, reduced CCXT candle fetch to 200, and added a semaphore (limit 2) to the Unified API to prevent OOM errors.
2. **Euro-Migration**: Moved deployment to Koyeb Frankfurt to bypass Binance geofencing.
3. **Architecture Consolidation**: Combined Backtester and Executor into `unified_bot.py`.

## 🚧 Active "Hot" Files
- `crypto_arb_bot/scanner.py` (Cooldown logic)
- `crypto_arb_bot/unified_bot.py` (Lifecycle & Concurrency logic)
- `crypto_arb_bot/backtester_agent.py` (Resource optimization)

## 💡 Key Architectural Decisions (Do Not Revert)
- **Decision**: Using `Koyeb` Frankfurt region for all exchange-facing services.
- **Decision**: Combining Backtester and Executor into ONE `unified_bot.py` service to save RAM/Free Tier credits.
- **Decision**: Using a 60s cooldown per symbol in the scanner to prevent n8n/API flooding.

## 📋 Immediate Next Steps (The To-Do List)
- [ ] Push local stability fixes to GitHub.
- [ ] Redeploy Unified Bot and Scanner on Koyeb.
- [ ] Re-activate n8n workflow.
- [ ] Monitor results at 0.005 (0.5%) or 0.01 (1.0%) threshold.

## 🐛 Known Bugs / Blockers
- None currently. The system was crashing due to load, which these fixes specifically address.
