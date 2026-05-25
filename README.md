# 🌿 IMPPATez

**Natural Product Informatics · Phytochemical Extraction · Full Batch Molecular Docking**

> Search the IMPPAT phytochemical database, prepare ligands and receptors, run batch docking, validate with RMSD, and visualize interactions — all from a single Streamlit interface.

**Supported engines:** AutoDock Vina | VinaXB | GNINA | Uni-Dock

---

## ✨ What it does

| | |
|---|---|
| 🌿 | **IMPPAT natural product search** — query the IMPPAT phytochemical database and collect compound records with SMILES for direct docking input |
| 🔬 | **Single & batch docking** — dock any number of natural product ligands against a prepared receptor |
| 🏗️ | **Guided receptor preparation** — download any PDB/CIF from RCSB, auto-scan HETATM records, strip solvent, add hydrogens, and convert to PDBQT |
| 🎯 | **Smart co-crystal ligand detection** — auto-scans drug-like HETATM records; shows a dropdown only when multiple candidates are present |
| ✏️ | **Flexible ligand input** — SMILES text, file upload (`.pdb` / `.sdf` / `.mol2`), or draw in Ketcher |
| 🧬 | **Heme-aware receptor preparation** — HEM/HEC/HEA/HEB stripped before OpenBabel, re-injected into PDBQT with correct AD4 atom types |
| ⚗️ | **Water, cofactor & metal control** — independently remove waters, keep metal ions (ZN, MG, CA, FE …), and keep/strip cofactors (HEM, FAD, NAD, ATP …) |
| ♻️ | **Redocking validation** — re-dock the co-crystal ligand from SMILES; calculates heavy-atom RMSD vs crystal pose with per-pose breakdown and 3D overlay |
| 🗺️ | **2D interaction diagrams** — custom SVG PoseView-style renderer + classic RDKit MolDraw2DSVG highlights |
| 📊 | **3D binding pocket viewer** — interacting residues as sticks with toggleable labels and adjustable distance cutoff |
| 📁 | **One-click ZIP** — receptor PDBQT, ligand SDFs & PDBQTs, all docked poses, 2D diagrams, score plot, and session report |

---

## 🚀 Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run imppatez.py
```

---

## 🔩 External Tools

Some workflows require command-line tools in addition to the Python packages:

- **Open Babel** (`obabel`) — hydrogen addition and PDBQT conversion
- **AutoDock Vina 1.2.7** — downloaded automatically on first launch
- **VinaXB / GNINA / Uni-Dock** — optional; provide binary path in the app
- **Docker** — only required when using the GNINA Docker image fallback

Put binaries on your `PATH`, or provide custom binary paths from within the app.

---

## 📦 Project Structure

```text
.
├── imppatez.py          # Streamlit user interface
├── core.py              # Streamlit-free docking/preparation helpers
├── requirements.txt     # Python dependencies
├── .gitignore           # GitHub upload hygiene
└── README.md
```

`core.py` is intentionally importable without Streamlit and contains all computation-focused protocol helpers: PDB cleaning, receptor preparation, ligand preparation, docking runners, RMSD calculation, interaction detection, and 2D diagram rendering.

---

## 📄 Citation

If you use IMPPATez in research, please cite the relevant underlying software and databases:

> **IMPPAT**
> Mohanraj et al., *Scientific Reports*, 2018
> DOI: https://doi.org/10.1038/s41598-018-22631-z

> **AutoDock Vina 1.2.7**
> Eberhardt et al., *Journal of Chemical Information and Modeling*, 2021
> DOI: https://doi.org/10.1021/acs.jcim.1c00203

> **RDKit**
> Landrum, G. (2023). RDKit: Open-source cheminformatics.
> https://www.rdkit.org

> **ProDy**
> Bakan et al., *Bioinformatics*, 2011
> DOI: https://doi.org/10.1093/bioinformatics/btr168

> **Meeko**
> https://github.com/forlilab/Meeko

> **Open Babel**
> O'Boyle et al., *Journal of Cheminformatics*, 2011
> DOI: https://doi.org/10.1186/1758-2946-3-33
