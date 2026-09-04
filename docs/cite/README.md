# Citing openQHA

Two ready-made entries:

| file | for |
|---|---|
| [`cite_openQHA.bib`](cite_openQHA.bib) | BibTeX — LaTeX, Zotero, JabRef |
| [`cite_openQHA.ris`](cite_openQHA.ris) | RIS — EndNote, Mendeley, Word |

## Before you use them

**The author line and the URL are placeholders.** They read `{openQHA developers}` and
`https://github.com/<org>/openQHA` because this repository does not record a real author
list or a public URL yet, and a citation naming the wrong author is worse than no
citation at all. Fill both in, and set `version`/`ET` to the release you actually used.

There is no openQHA paper yet, so both entries cite the **software**. Replace them with
the paper when one exists.

## What else to cite

The methods openQHA is built on are listed in the repository
[`README.md`](../../README.md#citation) — CREST/iMTD-GC, RMSD metadynamics, GFN2-xTB,
MACE and MACE-OFF23, and the quasi-harmonic literature. Cite the ones your run used.

**Data sets are cited only if you use them.** Whether curatedQM9 enters a calculation is
a choice made with `f7_mode`, and under the default (`drop_all`) none of its data is
used. See the Data section of the repository README.
