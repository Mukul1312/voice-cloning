---
description: Teach the pasted fastbook/ML section gently + full depth, then document to Notion
argument-hint: [optional: "go deeper" or a concept to focus on]
---

Teach the fastbook (or ML) section the user just pasted — or the concept named in $ARGUMENTS — following the **teaching-style** memory:

- **Gentle pace:** small steps, plain English, tiny concrete examples (real numbers / 2×2 grids), JS / engineer analogies, frequent "does this click?" check-ins.
- **Full depth (never sacrificed):** cover every code cell (decoded, often line by line) and every advanced concept. Never skip code or hard detail for the sake of gentleness — hard bits get MORE care, not less.
- If $ARGUMENTS contains "go deeper", dig further into the code / math / nuance of that bit.
- Tie back to the GGS voice-cloning goal only where it's genuinely relevant (don't force it).

THEN document the concept(s) to the Notion **ML Journey** database (data source `6703affa-3621-448a-bfc5-432db79d6167`) via `notion-create-pages`, in the same plain-English style: one-line idea → plain explanation (with a tiny example) → why it matters → code + plain read → analogy. Set Topics, Status=Understood, Date=today, Source.

End by inviting the next section.
