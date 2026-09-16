# 01: Table module: sections and column comments

**What to build:** a `.dat` can hold several named tables. The writer takes an ordered set of sections, each with its rows and an optional column schema of the Property-file shape (`{column: (Type, unit or None, doc)}`), and writes for each a `[section]` line, one comment line per column (`Type, unit: doc`), the `# col col ...` header, the rows and a blank line; a column the schema does not know is written without a comment and returned, so a caller's test can catch it. The reader gives `{section: rows}` back with today's value fidelity; a file without any `[section]` line reads as one unnamed section; the single-table reader keeps working on such files and refuses a sectioned file by naming its sections. An empty section reads as `[]`.

**Blocked by:** None (can start immediately).

**Status:** done 2026-09-16

- [x] write three sections (one empty), read back `{section: rows}` equal to what was written, ints/floats/nan/inf/booleans/NA/quoted strings intact
- [x] every header column has a comment line above the header when the schema knows it; unknown columns are returned by the writer
- [x] a section-less file written by the existing single-table writer reads back unchanged through both readers
- [x] the single-table reader on a sectioned file raises, naming the sections
- [x] a string cell beginning with `[` is quoted, so no row can be mistaken for a section line
- [x] unit test beside the existing round-trip test, one defect story per check, under a second
