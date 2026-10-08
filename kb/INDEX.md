# KB Index — what was extracted and where

## Pass 1 (done)
- All 39 PDFs in `media/` → `kb/raw/*.txt` via `pdftotext -layout` (39/39 OK, 5.7 MB).
- Marathi scanned PDFs OCR'd locally (Tesseract `mar+eng`, 200 DPI): `1. Paddy.ocr.txt` (8 pp), `advisory-marathi.ocr.txt` (14 pp) via `kb/ocr_marathi.py`.

## Structured rules (done)
- `rules/varieties.json` — 12 rice varieties (duration, grain, window, districts, yield, resistance) + district sowing windows. Sources: Konkan DSS report, Agresco 2018/2019/2023, Paddy bulletin.
- `rules/fert_pest.json` — fertilizer (11 rules: zone NPK doses, briquettes, INM/biofertilizer, DSR, rice-rice/rice-sugarcane systems), pesticide (19 rules: fungicides for sheath blight/blast/BLB, **brown spot (4 in-label MUP rules: Ediphenphos, Azoxystrobin+Mancozeb, Picoxystrobin+Isoprothiolane, Carbendazim 5% GR), leaf scald (3 off-label IRRI-actives rules: Mancozeb, Validamycin, Ediphenphos — no Indian label claim exists)**, stem-borer insecticides incl. ETL spray, botanicals), weeds (4 rules incl. pretilachlor). Sources: Konkan DSS, Agresco 2018-2023, Marathi OCR (cited as `1. Paddy.ocr` / `advisory-marathi.ocr` with `\n`-based line numbers), CIB&RC MUP fungicides 31.03.2026 (`cibrc/mup_fungicides_31.03.2026`).
- `rules/ipm.json` — 10 non-chemical IPM actions (water management, mechanical/cono weeding, Tephrosia green manure, ETL scouting, resistant varieties BM-4 / Ratnagiri-7 / Karjat-184 / Karjat-8). Types: monitoring/cultural/mechanical/variety. **Chemical-free by contract** — the recommender returns this list before any pesticide (IPM-first). Sources: Marathi OCR, Agresco 2019, Konkan DSS. Citation drift tracked by `ml/evals/rule_validator.py`.
- `rules/registry.json` — CIB&RC legality/registration layer for all 22 rule chemicals (19 pesticide + 4 herbicide). Built by `kb/build_registry.py` from official GOI PDFs in `media/cibrc/` (text in `kb/raw/cibrc/`): formulations list (as on 31.03.2026), banned/refused/restricted list (as on 31.07.2026), Major Uses of Pesticides. Status precedence `banned > restricted > registered > not_listed`. **Dichlorvos is `banned`** (S.O. 1196 (E), 20.03.2020) → hard-gated out of recommendations; Oxyfluorfen `restricted` (potato/groundnut only — not rice); **Mancozeb and Azoxystrobin+Mancozeb are `restricted`** (crop-specific: Guava/Jowar/Tapioca — not rice, note explains rice use unaffected). All entries carry cited lines (`cibrc/<name>`); drift tracked by the validator (section B/F). Brands/prices deliberately excluded. Re-run `python kb/build_registry.py` after editing rule chemicals.
- `tables/rainfall_konkan_2026.csv` — 2026 seasonal rainfall departure by district (IMD).

## Pending / known gaps
| Gap | Action | Where |
|---|---|---|
| ~~Brown_Spot, Leaf_Scald have NO cited chemical rule~~ **DONE** — Brown_Spot: 4 in-label MUP rules (Ediphenphos L666, Azoxystrobin+Mancozeb L2927, Picoxystrobin+Isoprothiolane L4648, Carbendazim 5% GR L215); Leaf_Scald: 3 off-label IRRI-actives rules (Mancozeb L1056, Validamycin L2559, Ediphenphos L666) with transparent off-label notes | — | kb/rules/fert_pest.json |
| DWE 2020-24 district-wise rice area/production/yield tables | Parse `kb/raw/DWE_*.txt` → `tables/dwe_rice_*.csv` | local (regex, next) |
| IMD daily rainfall for anomaly (risk engine) | Use `rf_p25_*_clm.nc` climatology + IMD API/backend feed | ml/src |

## Rules for contributors
1. Never add a chemical/dose without `source_pdf` + `line`/`page`.
2. If not in KB → recommender must `abstain` and flag `needs_expert`.
3. `kb/raw` is append-only — never edit extracted text in place.
4. `ipm.json` stays chemical-free — no chemical name, no dose ever (validator enforces).
5. After any rule edit run `python ml/evals/rule_validator.py --update-snapshot`.
6. Adding/renaming a pesticide or herbicide chemical requires re-running `python kb/build_registry.py` — the registry must cover every rule chemical (validator section F fails otherwise). A `banned` registry status permanently blocks that chemical from recommendations.