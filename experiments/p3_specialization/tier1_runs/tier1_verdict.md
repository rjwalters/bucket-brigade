# Tier-1 sweep verdict

Verdict ladder: `gap_closed_mean >= 0.88` -> **closed**; `0.49 <= mean < 0.88` -> **partial_upper**; `0.20 <= mean < 0.49` -> **partial_lower**; `mean < 0.20` -> **insufficient**.

| Trainer | Scenario | gap_closed (mean ± std) | n_seeds | Verdict |
|---------|----------|--------------------------|---------|---------|
| ippo | minimal_specialization | 0.113 ± 0.074 | 3 ok | insufficient |
| influence | minimal_specialization | 0.108 ± 0.032 | 3 ok | insufficient |
| hca | minimal_specialization | 0.073 ± 0.126 | 3 ok | insufficient |
| lola | minimal_specialization | 0.004 ± 0.048 | 3 ok | insufficient |
