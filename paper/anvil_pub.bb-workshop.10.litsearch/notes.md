# Litsearch notes — anvil_pub.bb-workshop.10 (re-run)

Re-run after the v10 review (`anvil_pub.bb-workshop.10.review/comments.md`,
minor tagged `[related-work]`) and the v10 citation audit
(`anvil_pub.bb-workshop.10.audit/citation-audit.md`, 4 keys "unverified —
source not on disk"). `web_search: true` is set in the thread BRIEF
frontmatter, so the opt-in web path was active; every entry in
`candidates.bib` passed the resolver-verified-or-dropped contract via
`anvil/lib/cite.py::resolve()` (see the Web provenance table below).

## Verification of the four unverified refs.bib entries

All four resolved against live authoritative sources. **No corrections
required** — none of these entries reappear in `candidates.bib`; the
existing `refs.bib` entries stand as-is.

| refs.bib key | Identifier | Verdict | Detail |
|---|---|---|---|
| `ppo2017` | arXiv:1707.06347 | **CONFIRMED** | arXiv API: title, all 5 authors (Schulman, Wolski, Dhariwal, Radford, Klimov), year 2017 match exactly. Correctly cited as an arXiv preprint (never venue-published). |
| `mappo2022` | arXiv:2103.01955 | **CONFIRMED** | arXiv API confirms authors (Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu) and ID. The arXiv abs page's Comments field states "accepted by NeurIPS 2022 Datasets and Benchmarks", confirming refs.bib's venue and year 2022 (the arXiv record's own year is 2021, the v1 submission; refs.bib correctly cites the published version). Note: the arXiv title carries a comma ("Cooperative, Multi-Agent Games"); refs.bib's comma-free form matches the NeurIPS proceedings title — keep as-is. |
| `openspiel2019` | arXiv:1908.09453 | **CONFIRMED** | arXiv API: title, year 2019, and the full 27-author list match refs.bib (the resolver renders author 17 as "Vylder, Bart De" — a name-split artifact; refs.bib's "De Vylder, Bart" is the correct form). Correctly cited as an arXiv preprint (no conference venue). |
| `psro2017` | arXiv:1711.00832 | **CONFIRMED** | arXiv API: title, all 8 authors, year 2017 match. The abs page's Comments field says "Camera-ready copy of NIPS 2017 paper", confirming the NeurIPS 2017 `@inproceedings` venue claim. |

The `refs.bib` header comment ("not yet auditor-verified; re-check arXiv IDs
and venue names") can now be updated by the reviser to record that this
litsearch performed the live re-check for all 11 keys' 4 outstanding cases
(the 6 benchmark keys + `pettingzoo2021` were already point-for-point matched
against `refs/benchmark_comparison.md` by the v10 audit).

## Positioning summary

**Cluster 1 — exploitability-evaluation empirical follow-ups (reviewer lead
ii, first half).** The v10 §6 paragraph engages the tradition through its two
founding artifacts (`openspiel2019`, `psro2017`) but cites no empirical
follow-up. Two verified candidates fill that: `lanctot2023population`
(Lanctot et al. 2023) is the exploitability line's own later move toward
*benchmark-shaped* evaluation — population-based evaluation in repeated
rock-paper-scissors, explicitly framed as "a benchmark for multiagent
reinforcement learning" — and `li2024meta` (Li & Wellman 2024) is the current
methodological statement of how to evaluate deep MARL when equilibrium
behavior is the object of interest (meta-game / empirical game-theoretic
evaluation). Citing one or both in the §6 paragraph (around L1180, after the
PSRO/NashConv sentence) upgrades the engagement from "we know the two
founding papers" to "we know where that line went," and both *strengthen* the
paper's scoping sentence: the RPS benchmark is a single fixed game, and
meta-game evaluation estimates equilibria empirically post hoc — neither
supplies a closed-form parametric regime map as ground truth.

**Cluster 2 — games-with-known-equilibria as MARL testbeds (reviewer lead ii,
second half: "any other parametric-family game constructions").** The closest
things found, and neither is close. `krever2025guard` (GUARD, 2025)
*constructs* realistic two-player matrix and security games for benchmarking
— but for equilibrium-*computation* algorithms (Nash/Stackelberg solvers),
not for MARL training, and its games are generated instances, not an
interpretable parametric family with a derived phase diagram.
`christianos2022pareto` (Pareto-AC, 2022) is the modern representative of the
climbing/penalty-game tradition: it evaluates equilibrium *selection* by MARL
agents in fixed matrix games whose equilibria are known by inspection. That
tradition (originating with Claus & Boutilier 1998 — see Web leads; no
resolvable identifier) is the oldest form of "NE-ground-truth MARL
evaluation," and Bucket Brigade should acknowledge it in §6: it differs in
that its games are individually specified 2-agent stateless matrices, whereas
Bucket Brigade is an N-agent *stochastic* game family whose equilibrium
*regime* moves continuously under three interpretable environment scalars.
`papoudakis2020benchmarking` (optional, weakest of the five) documents the
standard evaluation practice in cooperative MADRL — benchmark scores with no
equilibrium ground truth — and could support §1's framing sentence if the
drafter wants a citation there; it is not needed for §6.

**Cluster 3 — Appendix A template sources (reviewer lead iii, the standing
declined item).** All four canonical sources for
`\subsection{Relationship to canonical game-theory templates}` are now
resolver-verified: `diekmann1985volunteer` for the Volunteer's Dilemma
paragraph (Diekmann 1985, J. Conflict Resolution 29(4):605–610 — the paper
that named the game), `ledyard19952` for the Public Goods paragraph (Ledyard
1995, Handbook of Experimental Economics ch. 2 — the canonical survey),
`skyrms2003stag` for the Stag Hunt paragraph (Skyrms, Cambridge UP — Crossref
dates the book record 2003; the print edition is conventionally cited 2004;
keep the resolver's 2003 unless the drafter prefers the print convention with
publisher metadata), and `shapley1953stochastic` for the Stochastic Game
paragraph (Shapley 1953, PNAS 39(10):1095–1100 — the "every formal result for
finite stochastic games" sentence currently rests on no citation). The
free-rider paragraph needs no new source; `diekmann1985volunteer` and
`ledyard19952` jointly cover the free-riding incentive it describes.

**What the search did NOT find — and why that is a positioning strength.**
Three targeted searches (parametric/matrix-game benchmarks with computable
NE; NashConv-style empirical evaluation; climbing/penalty-game cooperative
benchmarks) surfaced *no* published parametric cooperative-competitive MARL
environment family whose equilibrium structure is characterized in closed
form across the parameter space. Everything adjacent falls short on at least
one clause: OpenSpiel's library and the matrix-game tradition are fixed,
individually-specified games; GUARD generates games but for solver
benchmarking; population/meta-game evaluation estimates equilibria
empirically rather than analytically; Pareto-AC's testbeds are stateless
2-agent matrices. The §6 claim "to our knowledge it is the only published
parametric cooperative-competitive MARL family in which a closed-form regime
map ... supplies the ground truth" survived a live search and can stand —
the reviser may even cite the new candidates as the survey behind the
"to our knowledge" qualifier.

## Confirmed coverage

- The six canonical MARL benchmarks (§1, §6, Table 3): fully covered by the
  existing refs.bib entries, previously matched point-for-point against
  `refs/benchmark_comparison.md`.
- PPO / MAPPO algorithmic recipe (§4): covered; both keys now live-verified.
- Equilibrium-computation tradition's founding artifacts (§6): covered
  (`openspiel2019`, `psro2017`, both now live-verified).

## Identified gaps

- **Claus & Boutilier 1998** (climbing/penalty games; the original
  known-equilibrium MARL testbeds) has no resolvable DOI or arXiv ID — AAAI-98
  proceedings predate DOI registration. It is a Web lead (below), not a
  citation. If the author wants it cited in §6, they must supply BibTeX
  manually into `<thread>/refs/` or `refs.bib` (an authoritative PDF is
  hosted at cs.toronto.edu/~cebly; the citation is conventional and safe to
  hand-enter *by the author*, but is outside this role's write contract).
- **Primary PDFs still not on disk** for any cited paper (carried from v9
  NC-2 / v10 audit): verification here is metadata-level (live API + abs
  pages), not full-text claim-level. Unchanged status; acquisition remains an
  author task if the venue demands it.
- **Skyrms year convention**: Crossref says 2003 (online book record); the
  print edition is cited 2004 in most venues. Either is defensible; pick one
  and keep it consistent.
- **BibTeX cosmetics on merge** (drafter task, not gaps in coverage): the
  lib-generated key `ledyard19952` inherits the chapter-number "2." from the
  De Gruyter title — the drafter may rekey (e.g. `ledyard1995public`) and
  strip the "2. " prefix on merge, and may retype `skyrms2003stag` → `@book`
  and `ledyard19952` → `@incollection` (Kagel & Roth, eds., Princeton UP,
  pp. 111–194). The identifier-verified fields (author, year, DOI, pages)
  must not change.

## Re-run delta

This is the first litsearch sibling on this thread (no
`anvil_pub.bb-workshop.*.litsearch/` existed before); it was triggered by the
v10 review's `[related-work]` minor, which asked for exactly three things:
(i) resolver-verify `openspiel2019`/`psro2017`/`ppo2017`/`mappo2022` — done,
all four CONFIRMED, zero corrections; (ii) survey the exploitability line's
empirical follow-ups and other parametric-family constructions — done, five
verified candidates (`lanctot2023population`, `li2024meta`, `krever2025guard`,
`christianos2022pareto`, `papoudakis2020benchmarking`), plus the finding that
no closer parametric-family prior exists; (iii) source the Appendix A named
templates — done, four verified candidates (`diekmann1985volunteer`,
`ledyard19952`, `skyrms2003stag`, `shapley1953stochastic`). Open after this
re-run: the Claus & Boutilier lead (author promotion required) and the
primary-PDF acquisition item (unchanged).

## Web provenance

| bib key | identifier | resolver |
|---|---|---|
| `lanctot2023population` | arXiv:2303.03196 | arxiv |
| `li2024meta` | arXiv:2405.00243 | arxiv |
| `krever2025guard` | arXiv:2505.14547 | arxiv |
| `christianos2022pareto` | arXiv:2209.14344 | arxiv |
| `papoudakis2020benchmarking` | arXiv:2006.07869 | arxiv |
| `diekmann1985volunteer` | doi:10.1177/0022002785029004003 | crossref |
| `ledyard19952` | doi:10.1515/9780691213255-004 | crossref |
| `skyrms2003stag` | doi:10.1017/CBO9781139165228 | crossref |
| `shapley1953stochastic` | doi:10.1073/pnas.39.10.1095 | crossref |

(The four re-verified refs.bib keys are author-supplied entries and therefore
do not appear here, per the provenance contract.)

## Web leads (unverified)

| title | authors | year | URL as found | reason unresolved |
|---|---|---|---|---|
| The Dynamics of Reinforcement Learning in Cooperative Multiagent Systems (AAAI-98; introduces the climbing and penalty games) | Caroline Claus, Craig Boutilier | 1998 | https://www.cs.toronto.edu/~cebly/Papers/multirl2.pdf | no identifier extractable (AAAI-98 proceedings paper; no DOI registered, no arXiv ID) |
