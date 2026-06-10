# Tier-1 sweep verdict

Verdict ladder: `gap_closed_mean >= 0.88` -> **closed**; `0.49 <= mean < 0.88` -> **partial_upper**; `0.20 <= mean < 0.49` -> **partial_lower**; `mean < 0.20` -> **insufficient**.

| Trainer | Scenario | gap_closed (mean ± std) | n_seeds | Verdict |
|---------|----------|--------------------------|---------|---------|
| ippo | minimal_specialization | 0.319 ± 0.182 | 4 ok | partial_lower |
| ippo | minimal_specialization | 0.206 ± 0.193 | 4 ok | partial_lower |
| ippo | minimal_specialization | 0.093 ± 0.096 | 4 ok | insufficient |
| ippo | minimal_specialization | 0.088 ± 0.099 | 4 ok | insufficient |
| ippo | minimal_specialization | -0.161 ± 0.048 | 4 ok | insufficient |
| ippo | minimal_specialization | -0.183 ± 0.047 | 4 ok | insufficient |
| ippo | minimal_specialization | -0.185 ± 0.062 | 4 ok | insufficient |
