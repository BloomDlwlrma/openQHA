# Citing openQHA

Two ready-made entries:

| file | for |
|---|---|
| [`cite_openQHA.bib`](cite_openQHA.bib) | BibTeX — LaTeX, Zotero, JabRef |
| [`cite_openQHA.ris`](cite_openQHA.ris) | RIS — EndNote, Mendeley, Word |

## Before you use them

We have not yet published a paper describing openQHA. Until then, please cite the software directly using the following BibTeX entry:

```bibtex
@misc{openqha,
    title = {{openQHA}: conformational free energy for small organic molecules on a machine-learned potential},
    author = {Zhang, Shiwei},  % replace with actual author list when known
    url = {https://github.com/BloomDlwlrma/openQHA},
    urldate = {2026-09-04},
    version = {0.1},
    year = {2026},
    month = sep,
    note = {Accessed: Sep 4, 2026},
}
```
If you use openQHA in your work, we encourage you to cite it in the main text of your paper, not only in the supporting information, to ensure proper discoverability by search engine or database.

## What else to cite

The methods openQHA is built on are listed in the repository
[`README.md`](../../README.md#citation) — CREST/iMTD-GC, RMSD metadynamics, GFN2-xTB,
MACE and MACE-OFF23, and the quasi-harmonic literature. Cite the ones your run used.

**Data sets are cited only if you use them.** Whether curatedQM9 enters a calculation is
a choice made with `f7_mode`, and under the default (`drop_all`) none of its data is
used. See the Data section of the repository README.
