# The Drug-Design Pipeline, Explained Like You're 5

*Simple words + everyday analogies. No jargon without a translation.*

---

## The Big Picture

Imagine a **kitchen where robots invent new cookie recipes**.

1. You show the robots a cookie you already love.
2. A robot **studies** what makes that cookie tasty (which chips, which crunch, which sweetness).
3. Another robot **invents** thousands of new cookie ideas that keep those tasty parts but change the rest.
4. A third robot checks: *"can we actually buy the ingredients for this at the store?"* — and throws the impossible ones in the trash.
5. You get a neat list of the best, actually-bakeable recipes.

That's our pipeline. The "cookies" are **molecules** (tiny building blocks of drugs).

### A word you'll see everywhere: SMILES

A **SMILES** is just how you write a molecule as text. Like writing a cookie
recipe in one short line of code:

```
C   = a single carbon atom (methane)
CCO = two carbons + oxygen = alcohol (the stuff in drinks)
CC(=O)OC1=CC=CC=C1C(=O)O = aspirin
```

That's it. Computers like text, chemists like pictures. RDKit can turn one
into the other.

---

## The 5 Steps (your plan)

### Step 1 — You pick a molecule you like 🎯

You give the app a SMILES string (or a picture later) of a molecule that
already works — e.g. a known drug. This is your **target**.

**Analogy:** "I love grandma's cookies, I want new cookies that taste just as good."

### Step 2 — RDKit extracts the pharmacophore 🧩

**RDKit** is a free toolbox (Python library) that understands molecules the
way a calculator understands numbers.

A **pharmacophore** (say it: "far-ma-co-for") is the **shape of the important
parts** of a molecule — not the exact atoms, just the features that make it
work:

> **ELI5:** Imagine a key that opens a locked door. The key's *exact* metal
> doesn't matter. What matters is: one bump here, a notch there, a hole in
> the middle. That pattern of bumps-and-notches is the pharmacophore.
>
> Molecules "open" locks in your body (like proteins that cause disease) the
> same way. Different keys can open the same lock if they have the same pattern!

So Step 2 turns your favorite molecule into a list of features like:

```
- a place that gives away a hydrogen (a "sticky bit")
- a flat ring of atoms (a "bumpy disc")
- a spot with positive charge (a "magnet end")
- these must be roughly 5, 8, and 11 atoms apart
```

Any new molecule with the same *pattern* might work the same way — even if
it's built differently. That's the trick that lets us invent new drugs.

**In the app:** we'll run a few RDKit functions and draw a picture with the
features highlighted in color.

### Step 3 — REINVENT4 invents new molecules 🍳

**What it is:** an AI chef from AstraZeneca. You tell it your scoring rules
("I want molecules that: stick to the target, aren't too big, dissolve in
water...") and it invents *thousands* of new molecules, getting better each
round.

**How it learns — reinforcement learning (ELI5):**

> It's like teaching a dog tricks with treats. The AI proposes a molecule →
> your scoring rule gives it a score from 0 to 1 → high score = treat
> ("do more like THAT!") → low score = no treat ("try something else").
> After thousands of tries, it only makes high-scoring molecules.

Under the hood it's a language model (like a tiny autocomplete) that writes
SMILES text one letter at a time, plus the treat-system on top.

**The big catch (important!):** REINVENT4 is **command-line only**. No
buttons, no pictures. You:

1. Write a settings file in TOML format (a text file like `config.toml`)
2. Run one command: `reinvent -l log.txt config.toml`
3. Get back a **CSV/text file of SMILES strings** — just text, no visuals

That's exactly the gap our front-end app will fill: *we* build the buttons
and pictures, REINVENT4 stays the engine under the hood.

**Another catch:** it only *proposes* molecules. It never says "here's how
to make this in a lab" and it doesn't guarantee the molecule is safe or
works — that's what the next steps are for.

Also good to know: you must download the pre-trained "brain" files (priors)
separately from Zenodo — the GitHub repo is just the code.

**Modes it has** (all using the same treat-training idea):

| Mode | What it invents | Cookie analogy |
|---|---|---|
| Reinvent | totally new molecules from scratch | invent a whole new cookie |
| Mol2Mol | molecules *similar to* ones you give it | "tweak grandma's recipe a little" |
| LibInvent | just the decoration for a fixed core | "keep the cookie base, change the chips" |
| LinkInvent | a piece to join two halves | "glue two cookies together" |

For your plan (you have a reference molecule → invent new ones that keep the
pattern), **Mol2Mol** and classic **Reinvent with a pharmacophore score** are
the most relevant modes.

### Step 4 — AiZynthFinder checks "can we actually make it?" 🔬

**What it is:** another free tool from AstraZeneca. Given a molecule, it
works **backwards**: "to build this, I'd first build these two smaller
pieces... and to build those, these even simpler pieces..." until it reaches
pieces you can just **buy from a chemical catalog**.

**Analogy (ELI5):**

> You want to build a LEGO castle but you only have a big picture of it.
> You work backwards: "the tower = wall piece + roof piece", "the wall piece
> = 2x4 bricks + window bricks"... until you're left with standard bricks
> you can actually buy. If you get stuck on a weird brick that's sold
> nowhere — sorry, that castle stays a dream.

**How it works (a tiny bit more grown-up):** it uses **Monte Carlo tree
search** (try things in a smart random order, remember what worked) plus a
**neural network** that has learned common chemistry reactions. It recursively
breaks the molecule into simpler pieces until everything left is a purchasable
"building block" (also called *stock*).

**Same catch:** command-line + Python API only. No pretty interface.

**In our pipeline:** after REINVENT4 invents candidates, AiZynthFinder runs
on each one. Un-buildable ones get thrown out. Buildable ones get a
"route found ✓" stamp and the number of steps ("you'd need 3 baking steps").

### Step 5 — Ranked report 📋

The app shows a sortable list: best score first, with a ✓/✗ for "makeable",
a picture of each molecule, its scores, and (if found) the recipe route from
AiZynthFinder.

---

## The Tools, Summarized

| Tool | Job | One-line analogy | Interface |
|---|---|---|---|
| **REINVENT4** | invents new molecules | chef inventing recipes | CLI only (TOML config → CSV out) |
| **AiZynthFinder** | checks makeability | "can I buy these ingredients?" | CLI / Python only |
| **RDKit** | chemistry toolbox | molecule calculator | Python library |
| **GenUI** | *reference* for pretty front-end | the "nice kitchen counter" we study, not copy | web app |
| **DrugEx / SMILES-RNN** | alternative inventors | other chef schools (same job as REINVENT) | CLI / Python |

### About the alternatives

**DrugEx** and **SMILES-RNN** do the same *job* as REINVENT4 (invent
molecules with reinforcement learning / language models) but with different
training recipes. You'd only reach for them to *compare* results ("do
different chefs agree?") — for a hobby pipeline, one inventor (REINVENT4) is
plenty.

### About GenUI

GenUI is a published **web front-end** for de novo molecule generation —
proof that the "REINVENT4 but with buttons" idea works. We use it as
**inspiration**: how does it show molecules? how does it run jobs in the
background? We build our own, simpler version.

**Update:** we checked it properly. It's a big Django + Postgres + Redis +
Docker platform from 2022 that wraps a different AI chef (DrugEx) — running
it would be like buying a whole restaurant just to peek at the kitchen's
floor plan. So: cloned for reference only, never run. Details in
`docs/Genui_NOTES.md`. Bonus: RDKit (which we already have) draws molecule
pictures by itself, so the "pretty preview" part is easy without GenUI.

---

## The Full Flow (one picture)

```
 [You] give reference molecule (SMILES)
        │
        ▼
 Step 2: RDKit  ──►  pharmacophore pattern (bumps & notches list)
        │
        ▼
 Step 3: REINVENT4 (RL loop)
   invent 100 molecules → score them vs. pattern + rules
   keep best → invent 100 more, better → repeat N rounds
        │
        ▼
 Step 4: AiZynthFinder  ──►  each molecule: buildable? route? steps?
        │                     ✗ trash the impossible ones
        ▼
 Step 5: Ranked report
   (picture, score, makeable?, route, number of steps)
```

---

## Honest Reality Checks (so no surprises)

- **Scoring by pharmacophore match** needs 3D shapes; for a hobby start we can
  approximate with simpler 2D feature similarity and upgrade later.
- **AiZynthFinder needs reaction + stock data files** (~GBs, downloaded once).
- **REINVENT4 needs PyTorch** (heavy, ~2GB+ with CPU wheels) and the prior
  model files from Zenodo.
- This pipeline proposes ideas; it does **not** make real medicine. Testing
  happens in real labs, years later, by humans in white coats.
- Everything here is free & open source (Apache-2.0 / MIT licensed). For a
  hobby project this is fine; commercial drug discovery has extra rules.

---

## 🏁 The Plan Is Built!

All 5 steps work end-to-end in the web app — type a molecule, press
"🚀 Run whole pipeline", and about a minute later you get a ranked report
of the best, actually-makeable invented molecules (with CSV download).

## What's Installed So Far

- ✅ REINVENT4 cloned to `tools/REINVENT4` (v4.8.24) — see `tools/README.md`
- ✅ AiZynthFinder cloned to `tools/aizynthfinder` (v4.4.1) — see `tools/README.md`
- ✅ Both installed in their own Python 3.12 "sandboxes" (`.venv-reinvent`,
  `.venv-aizynth`) — REINVENT4 on CPU PyTorch, AiZynthFinder on light
  onnxruntime. Recreate with `sh ./scripts/setup_envs.sh`
- ⬜ Prior models from Zenodo
- ⬜ Reaction + stock data for AiZynthFinder (figshare, one command)
- ⬜ Our own app (front-end + pipeline glue) — the fun part
