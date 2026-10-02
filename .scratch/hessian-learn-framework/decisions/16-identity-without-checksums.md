# Identity without checksums: the sha256 machinery comes out, the seed material stays

Type: task
Status: open
Blocked by: None.
Part of: [hessian-learn-framework](../map.md)

## Question / work

User ruling (2026-10-02, grilled to a signed contract): remove every sha256 use the
project adds of its own -- the checksum-as-correspondence machinery -- and keep exactly
one exception, the sha256 that serves as PRNG seed material (`frame_seed`,
`valid_probes`, `frame_draw`); upstream mace/CREST code is never touched ("mace/crest
native", Q14=(a)). The "version locking" half is scoped to the process layer -- the
two-side commit-locking deployment discipline, the fork-by-URL requirement, the docs pin
guidance -- while the product identity stays: `check_fork`'s refusals and the Record's
fork/package commit fields (Q9=A). The Tianhe GitHub linkage goes: install-side default
and docs here, checkout remotes on-site.

Promises this ruling knowingly overwrites (dated notes follow in 16c): decision 04's
"kept: CONFIG_SHA256 + non-weight hashes"; the 2026-10-02 record-slimming ruling's field
table; the campaign's "both sides size+sha equal" transfer rule; the two test pins on
`CONFIG_SHA256`.

Sequencing: slices 16a-16d land locally (no Tianhe touch) while the two arms run; 16e is
the single deferred on-site pass, gated on ticket 15 (the arms' Records + receipts) and
on 16a-16d.

**Postscript (2026-10-02, from the 16e grilling; the sentences above stay as
written).** The on-site execution face changed against 16e's first write: the landing
is an overlay of the new-code ZIPs (`unzip -o` + merge; downloaded from GitHub after
the workstation pushes; no `.git` is carried or updated), and Tianhe gets no operator
git action -- no receipts, no writes, no tree surgery; the "checkout remotes on-site"
item retires (nothing is stripped; a stale, "modified" site status is expected and
harmless). The receipts are content probes plus the workstation-side record of the
pushed heads; the wait-for-the-arms gate became "as soon as the ZIPs are on site,
avoiding each arm's stage-1 -> stage-2 handoff moment". The full delta list is
[16e](../implementation/16e-the-tianhe-landing.md)'s Revision note.

## Answer

<!-- resolver: append what was done + evidence; set Status: resolved once 16a-16e have landed; one line to the map -->
