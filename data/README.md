# Data layout

The training code expects the following files. Large or third-party datasets are not bundled with the repository.

```text
data/
├── gdsc/
│   ├── gdsc_expr.csv
│   ├── gdsc_sens.csv
│   ├── gdsc_smiles.csv
│   └── gdsc_targets.csv
├── gse112274/
│   ├── GSE112274_expr.csv
│   ├── GSE112274_label.csv
│   ├── target_smiles.csv
│   └── target_targets.csv
└── pathways/
    └── h.all.v2026.1.Hs.symbols.gmt
```

## File conventions

### `gdsc/gdsc_expr.csv`

Expression matrix with **cell lines as rows** and **genes as columns**. The first column is used as the row index / cell-line identifier.

### `gdsc/gdsc_sens.csv`

The first three columns are interpreted as:

```text
cell, drug, value
```

The default configuration assumes larger `value` means greater sensitivity. For each drug, the top 10% and bottom 10% of available source samples are used as sensitive (`1`) and resistant (`0`) examples.

### `gdsc/gdsc_smiles.csv`

The first two columns are interpreted as:

```text
drug, smiles
```

### `gdsc/gdsc_targets.csv`

The first two columns are interpreted as:

```text
drug, target
```

Multiple rows per drug are allowed.

### `gse112274/GSE112274_expr.csv`

GSE112274 single-cell expression matrix with **cells as rows** and **genes as columns**. The first column is used as the cell identifier.

### `gse112274/GSE112274_label.csv`

The first two columns are interpreted as:

```text
cell, label
```

where labels are integer response classes (`0` = resistant, `1` = sensitive) used for retrospective evaluation.

### `gse112274/target_smiles.csv`

The first two columns are interpreted as:

```text
drug, smiles
```

The file should contain a `Gefitinib` entry. In the original pipeline logic, target-specific metadata is used when the target drug is not already available in the GDSC drug metadata.

### `gse112274/target_targets.csv`

The first two columns are interpreted as:

```text
drug, target
```

Include one row per known Gefitinib target.

### `pathways/h.all.v2026.1.Hs.symbols.gmt`

GMT pathway collection. The default filename follows the Hallmark gene-set file used in the supplied research script. If you use another GMT collection, rename it to this filename or update `resolve_paths()` in the source code.

## Gene identifiers

The pipeline uppercases gene symbols, strips suffixes after the first `.`, removes surrounding whitespace, and keeps the first occurrence of duplicated normalized gene names.
