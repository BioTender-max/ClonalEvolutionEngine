# ClonalEvolutionEngine

**Tumor Clonal Evolution from Multi-Region Sequencing**

A pure-Python pipeline for reconstructing tumor evolutionary history from multi-region sequencing data.

## Features
- Cancer cell fraction (CCF) estimation from VAF + copy number + purity
- Clonal hierarchy reconstruction (phylogenetic tree from CCF ordering)
- Subclone detection (Gaussian mixture model clustering)
- Driver mutation timing (early vs late based on CCF)
- Evolutionary fitness estimation (subclone growth rate)

## Results
- 20 tumors × 5 regions, 200 somatic mutations per tumor
- Mean subclone count: 3.0 per tumor
- Mean clonal fraction: 0.242
- Mean tumor purity: 0.737 ± 0.090
- Mean evolutionary fitness: 1.135 ± 1.088
- Early driver mutations: 58.4, Late: 92.2

## Usage
```bash
pip install numpy scipy matplotlib
python clonal_evolution_engine.py
```

## Tags
`clonal-evolution` `tumor-heterogeneity` `subclone` `phylogenetic-tree` `cancer-evolution` `ccf`
