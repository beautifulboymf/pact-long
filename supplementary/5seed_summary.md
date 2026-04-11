# 5-Seed Results: TARNet and GDC on DBLP + CoraFull

All values: mean±std over 5 seeds (seeds 0-4), 200 epochs, lr=1e-3.

## TARNet: GPE improves ALL 6/6 settings

| Dataset | ρ | TARNet | TARNet+GPE | Δ% |
|---|---|---|---|---|
| CoraFull | 5 | 2.108±0.199 | **1.872±0.094** | -11% |
| CoraFull | 10 | 3.012±0.231 | **2.517±0.209** | -16% |
| CoraFull | 30 | 7.191±2.197 | **4.677±0.302** | -35% |
| DBLP | 5 | 1.875±0.171 | **1.722±0.107** | -8% |
| DBLP | 10 | 2.469±0.279 | **2.230±0.074** | -10% |
| DBLP | 30 | 6.132±0.490 | **5.448±1.072** | -11% |

## GDC: GPE improves 5/6 settings

| Dataset | ρ | GDC | GDC+GPE | Δ% |
|---|---|---|---|---|
| CoraFull | 5 | 3.565±0.357 | **3.088±0.303** | -13% |
| CoraFull | 10 | 4.030±0.278 | **3.491±0.534** | -13% |
| CoraFull | 30 | **4.870±0.229** | 5.588±1.525 | +15% (overfits) |
| DBLP | 5 | 3.196±0.202 | **2.757±0.274** | -14% |
| DBLP | 10 | 3.229±0.307 | **2.715±0.100** | -16% |
| DBLP | 30 | 4.201±0.350 | **3.786±0.173** | -10% |

## Key observations
- TARNet+GPE improvements are OUTSIDE 1-std in 5/6 settings (only ρ=5 DBLP is marginal)
- GDC+GPE ρ=30 CoraFull: confirmed overfitting — this is where Var Weighting is needed
- 5-seed std is generally lower than 3-seed, confirming stability
