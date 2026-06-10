# Echoes in the Sky
Research repository for analyzing Trump-related discourse on Bluesky and comparing it with a thematic analysis of executive orders.
![Overall Workflow](Figures/Supplementary_Figure_Overall.jpg)

The repository includes:
- [Bluesky_Data](Bluesky_Data): sample Bluesky data, microtopics, and theme-generation outputs. The full dataset is available upon reasonable request for research purposes.
- [EO_Data](EO_Data): executive-order theme-generation outputs
- [Code_Scripts](Code_Scripts): collection, analysis, and figure-generation scripts
> **📌 Supplementary Material**  
> **[SUPPLEMENTARY_MATERIAL.pdf](SUPPLEMENTARY_MATERIAL.pdf)**: Supplementary methods, analyses, and supporting documentation for the study.

---

## Bluesky Data

The public Bluesky release in this repository includes:

- [Bluesky_SampleData.csv](Bluesky_Data/Bluesky_SampleData.csv): 100 randomly selected English-language posts with selected columns
- [Microtopics.csv](Bluesky_Data/Microtopics.csv): full microtopic output
- [Microtopics_top100.csv](Bluesky_Data/Microtopics_top100.csv): highest-frequency microtopics
- [ThemeGeneration_Step1_Initial_Codes.csv](Bluesky_Data/ThemeGeneration_Step1_Initial_Codes.csv): initial codebook
- [ThemeGeneration_Step2_Themes.csv](Bluesky_Data/ThemeGeneration_Step2_Themes.csv): consolidated theme definitions
- [ThemeGeneration_Step3_Annotation.csv](Bluesky_Data/ThemeGeneration_Step3_Annotation.csv): microtopic-to-theme annotations

Collection summary for the underlying full corpus:

- Query set: `trump`, `maga`, `trumpism`, `potus45`, `america first`, `keep america great`, `Trumpocalypse`, `Republican`, `Drumpf`, `45th President`
- Total collected posts: 38,467,701
- Available features in the full dataset: 304
- Unique authors: 1,364,569

The full post-level dataset is not stored in this repository. It may be available for research use upon reasonable request.

## Executive Orders Data

The repository also includes a separate executive-order thematic analysis in [EO_Data](EO_Data):

- [EO_ThemeGeneration_Step1_Initial_Codes.csv](EO_Data/EO_ThemeGeneration_Step1_Initial_Codes.csv)
- [EO_ThemeGeneration_Step2_Themes.csv](EO_Data/EO_ThemeGeneration_Step2_Themes.csv)
- [EO_ThemeGeneration_Step3_Annotation.csv](EO_Data/EO_ThemeGeneration_Step3_Annotation.csv)

These files cover 258 executive orders issued between 2025-01-20 and 2026-05-01 and are intended to support comparison between policy activity and Bluesky discourse.

## Code Scripts

Most scripts live in [Code_Scripts](Code_Scripts). They fall into a few groups:

- data collection
- DuckDB and parquet analysis
- sentiment analysis and figure generation
- executive-order and Bluesky comparison analysis
- Granger-causality and structural-break analysis

A short script guide is available in [Code_Scripts/README.md](Code_Scripts/README.md).

## Setup

This repo does not currently ship a reproducible environment lockfile, but the checked-in scripts mainly rely on the packages listed in [requirements.txt](requirements.txt).

Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Most analysis scripts will still require local research data in paths such as `local_data/03_Github_data/...`.

For the Bluesky collection script specifically, set credentials through environment variables instead of editing the file:

```bash
export BLUESKY_HANDLE="your-handle.bsky.social"
export BLUESKY_APP_PASSWORD="your-app-password"
```

## Notes for Reuse

- Many scripts assume specific parquet, CSV, and output directory layouts from the original research environment.
- Several figure scripts write derived outputs next to the local datasets they analyze.
- Topic labels, theme labels, and generated summaries should be treated as research outputs that may require manual validation.
- The repository is best understood as a transparent release of data artifacts plus working analysis code, rather than a polished software package.
- For questions, contact: kylewang@udel.edu

## Citation

If you use the microtopics workflow, please also cite:

Salloum, A., Quelle, D., Iannucci, L., Bovet, A., and Kivela, M. (2025). *Politics and polarization on Bluesky*. arXiv. [https://arxiv.org/abs/2506.03443](https://arxiv.org/abs/2506.03443)

If you use the theme-generation workflow, please also cite:

Wang, Q., Erqsous, M., Barner, K. E., and Mauriello, M. L. (2025). *LATA: A Pilot Study on LLM-Assisted Thematic Analysis of Online Social Network Data Generation Experiences*. *Proceedings of the ACM on Human-Computer Interaction*, 9(2), Article CSCW124. [https://dl.acm.org/doi/abs/10.1145/3711022](https://dl.acm.org/doi/abs/10.1145/3711022)

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
