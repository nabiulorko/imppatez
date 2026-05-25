#!/usr/bin/env python3
"""
IMPPATez - Natural Product Informatics with Full Batch Molecular Docking
Merged from IMPPATez (app123.py) + Vina_batch.ipynb
Supports: AutoDock Vina 1.2.7 | VinaXB | GNINA | Uni-Dock
Added: RMSD calculation, co-crystal redocking with SMILES input, unlimited compound docking
"""

# ═══════════════════════════════════════════════════════════════════════════
#  📦 IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import urllib.parse
import re
import os
import shutil
import time
import zipfile
import io
import subprocess
import tempfile
import platform
import json as _json
import select
import pty
import glob
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, List, Optional, Tuple

# Cheminformatics
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolAlign

try:
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    STEREO_ENUM = True
except ImportError:
    STEREO_ENUM = False

# Visualization
import matplotlib.pyplot as plt
import numpy as np

# For RMSD calculation
from scipy.spatial import distance_matrix

# Optional: ProDy for RMSD
try:
    from prody import parsePDB, calcCenter, writePDB, calcRMSD
    PRODY_AVAILABLE = True
except ImportError:
    PRODY_AVAILABLE = False

# Optional: Meeko for PDBQT
try:
    from meeko import MoleculePreparation
    try:
        from meeko import PDBQTWriterLegacy as _MeekoWriter
        MEEKO_LEGACY = True
    except ImportError:
        MEEKO_LEGACY = False
    MEEKO_AVAILABLE = True
except ImportError:
    MEEKO_AVAILABLE = False

# Dimorphite-DL for protonation
try:
    from dimorphite_dl import protonate_smiles as _dimorphite_protonate
    DIMORPHITE_AVAILABLE = True
except ImportError:
    DIMORPHITE_AVAILABLE = False

# Base URLs / constants
BASE = "https://cb.imsc.res.in/imppat"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_DIR = "outputs"

# Protocol core loader. The GitHub layout keeps core.py beside app3.py, while
# older local experiments may still keep it under anyone-docking-main.
APP_DIR = Path(__file__).resolve().parent
ACD_DIR = APP_DIR / "anyone-docking-main"
ACD_CORE_AVAILABLE = False
_ACD_CORE = None
for _core_dir in (APP_DIR, ACD_DIR):
    try:
        if not (_core_dir / "core.py").exists():
            continue
        if str(_core_dir) not in sys.path:
            sys.path.insert(0, str(_core_dir))
        import core as _ACD_CORE
        ACD_CORE_AVAILABLE = True
        break
    except Exception:
        ACD_CORE_AVAILABLE = False
        _ACD_CORE = None

# Metal constants (from Vina_batch)
METAL_RESNAMES = {
    "MG","ZN","CA","MN","FE","CU","CO","NI","CD","HG","NA","K","HO",
    "LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","ER","TM","YB","LU",
}
METAL_CHARGES = {
    "MG":2.0,"ZN":2.0,"CA":2.0,"MN":2.0,"FE":3.0,
    "CU":2.0,"CO":2.0,"NI":2.0,"CD":2.0,"HG":2.0,"HO":3.0,
    "LA":3.0,"CE":3.0,"PR":3.0,"ND":3.0,"PM":3.0,"SM":3.0,
    "EU":3.0,"GD":3.0,"TB":3.0,"DY":3.0,"ER":3.0,"TM":3.0,
    "YB":3.0,"LU":3.0,"NA":1.0,"K":1.0,
}
_NO_REINJECT = {"HO","LA","CE","PR","ND","PM","SM","EU","GD","TB","DY","ER","TM","YB","LU"}
EXCLUDE_IONS = set(
    "HOH,WAT,DOD,SOL,NA,CL,K,CA,MG,ZN,MN,FE,CU,CO,NI,CD,HG,HO,"
    "LA,CE,PR,ND,PM,SM,EU,GD,TB,DY,ER,TB,YB,LU".split(",")
)
GLYCAN_NAMES = {"NAG","BMA","MAN","FUC","GAL","GLC","SIA","NGA","FUL","GLA","BGC","A2G","LAT","MAL"}
COFACTOR_NAMES = {"ATP","ADP","AMP","GTP","GDP","GMP","NAD","NAP","NDP","FAD","FMN","HEM","HEC","HEA",
                  "GOL","PEG","EDO","MPD","PGE","PG4","SO4","PO4","SUL","PHO","IHP","TTP","CTP","UTP",
                  "COA","SAM","SAH","EPE","MES","TRS","ACT","ACY"}
HEME_RESNAMES = {"HEM","HEC","HEA","HEB","HDD","HDM"}
_MIN_LIG_ATOMS = 4
_BACKBONE = {"N","CA","C","O"}
BUFFER_RESNAMES = {
    "GOL", "EDO", "PEG", "PGE", "PG4", "MPD", "DMS", "DMSO", "ACT",
    "ACY", "ACE", "TRS", "MES", "EPE", "BME", "SO4", "PO4", "NO3",
    "SCN", "FMT", "IPA", "EOH", "MOH", "CL", "BR", "IOD",
}
WATER_RESNAMES = {"HOH", "WAT", "DOD", "SOL"}

# Page config
st.set_page_config(
    page_title="IMPPATez — Natural Product Docking",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    :root {
        --bg:#FFFFFF; --bg-subtle:#F6F8FA; --bg-card:#F0F4F8;
        --border:#D0D7DE; --accent:#0969DA; --success:#1A7F37;
        --warn:#9A6700; --text:#24292F; --text-muted:#57606A;
    }
    .main { background: linear-gradient(180deg, #ffffff 0%, #f6fbf6 100%); }
    .title-container {
        text-align: center; padding: 1rem 0;
        background: linear-gradient(90deg, #1a3d20 0%, #2e7d32 60%, #43a047 100%);
        border-radius: 12px; margin-bottom: 2rem; color: white;
    }
    .step-card {
        background: linear-gradient(135deg, #f0fbf0 0%, #e4f5e5 100%);
        border: 1.5px solid #a5d6a7; border-radius: 16px;
        padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(46,125,50,0.12);
    }
    .step-title {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 700;
        color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px;
    }
    .step-heading {
        font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem;
        font-weight: 700; color: #1a3d20; margin-bottom: 16px;
    }
    .result-pill {
        display: inline-block; background: #DDF4FF; border: 1px solid #54AEFF;
        color: #0550AE; border-radius: 20px; padding: 2px 12px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; margin: 2px;
    }
    .success-pill {
        display: inline-block; background: #DAFBE1; border: 1px solid #1A7F37;
        color: #1A7F37; border-radius: 20px; padding: 4px 14px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    }
    .warn-pill {
        display: inline-block; background: #FFF8C5; border: 1px solid #9A6700;
        color: #9A6700; border-radius: 20px; padding: 4px 14px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    }
    .stat-card {
        background: white; border-radius: 10px; padding: 1rem; text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05); border: 1px solid #e0e0e0;
    }
    .stat-number { font-size: 2rem; font-weight: 800; color: #2e7d32; }
    .score-best { font-family: 'IBM Plex Mono', monospace; font-size: 2.4rem; color: #1A7F37; font-weight: 600; }
    .score-unit { font-size: 1rem; color: var(--text-muted); }
    .log-box {
        background: var(--bg-subtle); border: 1px solid var(--border); border-radius: 6px;
        padding: 12px 16px; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
        max-height: 220px; overflow-y: auto; white-space: pre-wrap;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #43a047); color: white;
        border: none; font-weight: 600; border-radius: 8px;
        padding: 0.5rem 1.5rem; transition: all 0.3s ease;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #1b5e20, #2e7d32); transform: translateY(-2px); }
    .engine-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 700;
    }
    hr { margin: 1rem 0; border: none; border-top: 2px dashed #c8e6c9; }
    .rmsd-excellent { color: #1A7F37; font-weight: bold; }
    .rmsd-good { color: #9A6700; font-weight: bold; }
    .rmsd-poor { color: #D1242F; font-weight: bold; }
    .current-receptor-card {
        background: linear-gradient(135deg, #f8fbff 0%, #eef7f0 100%);
        border: 1px solid #b7d9c0; border-left: 5px solid #2e7d32;
        border-radius: 10px; padding: 14px 16px; margin: 14px 0;
        box-shadow: 0 3px 14px rgba(46,125,50,0.08);
    }
    .current-receptor-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px;
        margin-bottom: 4px; font-weight: 700;
    }
    .current-receptor-path {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.92rem;
        color: #1a3d20; overflow-wrap: anywhere;
    }
    .ligand-detect-card {
        display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr);
        gap: 14px; background: linear-gradient(135deg, #ffffff 0%, #edf8f2 100%);
        border: 1px solid #9bd2ad; border-radius: 12px; padding: 14px;
        margin: 12px 0; box-shadow: 0 4px 18px rgba(9,105,218,0.08);
    }
    .ligand-detect-item {
        background: rgba(255,255,255,0.78); border: 1px solid rgba(46,125,50,0.14);
        border-radius: 8px; padding: 12px;
    }
    .ligand-detect-label {
        font-family: 'IBM Plex Mono', monospace; color: var(--text-muted);
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.3px;
        font-weight: 700; margin-bottom: 5px;
    }
    .ligand-detect-value {
        color: #163b22; font-size: 1.02rem; font-weight: 700;
        overflow-wrap: anywhere;
    }
    @media (max-width: 760px) {
        .ligand-detect-card { grid-template-columns: 1fr; }
    }
    .rmsd-card {
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
        border-radius: 12px; padding: 16px; margin: 12px 0;
        border-left: 4px solid #2e7d32;
    }
    .rmsd-value {
        font-size: 1.8rem; font-weight: 700; font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  HEADER (Centered)
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="title-container">
    <h1 style="color: white; margin: 0;">🌿 IMPPATez</h1>
    <p style="color: #c8e6c9; margin: 0;">Natural Product Informatics · Phytochemical Extraction · Full Batch Molecular Docking</p>
    <p style="color: #a5d6a7; margin: 0; font-size: 0.8rem;">AutoDock Vina 1.2.7 | VinaXB | GNINA | Uni-Dock | RMSD Validation with SMILES Input</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def timestamp_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session

@st.cache_resource
def get_session():
    return create_robust_session()

def safe_name(s):
    s = (s or "").strip() or "mol"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s

def _pill(text, kind="info"):
    cls = {
        "info": "result-pill",
        "success": "success-pill",
        "warn": "warn-pill",
    }.get(kind, "result-pill")
    return f'<span class="{cls}">{text}</span>'

def receptor_session_tag(receptor_path):
    if receptor_path:
        stem = Path(str(receptor_path)).stem.strip()
        stem = re.sub(r"^receptor_", "", stem, flags=re.IGNORECASE)
        if re.fullmatch(r"[0-9A-Za-z]{4}", stem):
            return stem.upper()
        if stem:
            return safe_name(stem)
    return "receptor"

def default_docking_session_name(receptor_path=None):
    return f"imppatez_{receptor_session_tag(receptor_path)}_{timestamp_tag()}"

def assert_file_ok(path, min_bytes=100, msg=""):
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        raise ValueError(msg or f"File missing/too small: {path}")

def unique_workdir(base_path):
    """Return a non-existing path by adding a numeric suffix when needed."""
    base = Path(base_path)
    if not base.exists():
        return base
    for i in range(2, 1000):
        candidate = base.with_name(f"{base.name}_{i}")
        if not candidate.exists():
            return candidate
    return base.with_name(f"{base.name}_{timestamp_tag()}")

def _count_pdb_atoms(path):
    try:
        with open(path) as f:
            return sum(1 for line in f if line.startswith(("ATOM", "HETATM")))
    except Exception:
        return None

def _format_hetatm_label(row):
    resname = str(row.get("resname") or "").strip().upper() or "HETATM"
    chain = str(row.get("chain") or "").strip() or "-"
    resid = row.get("resid", "")
    return f"{resname}:{chain}:{resid}"

def fmt2(value, suffix=""):
    """Format numeric results consistently for UI and reports."""
    try:
        if value is None or value == "":
            return "N/A"
        return f"{float(value):.2f}{suffix}"
    except Exception:
        return str(value)

def round_numeric_results(rows, digits=2):
    """Return table rows with numeric result columns rounded for display."""
    rounded = []
    for row in rows or []:
        new_row = dict(row)
        for key, value in new_row.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                new_row[key] = round(float(value), digits)
        rounded.append(new_row)
    return rounded

def format_receptor_preparation_log(rec_result):
    """Create a clean user-facing receptor preparation report."""
    if not rec_result:
        return ""

    center = rec_result.get("center") or (
        rec_result.get("cx"), rec_result.get("cy"), rec_result.get("cz")
    )
    hetatm_rows = rec_result.get("hetatm_table") or []
    reference_rows = [r for r in hetatm_rows if str(r.get("action", "")).lower() == "reference"]
    removed_rows = [r for r in hetatm_rows if str(r.get("action", "")).lower() == "remove"]
    water_removed = [
        r for r in removed_rows
        if str(r.get("type_guess", "")).lower() == "water"
        or str(r.get("resname", "")).upper() in WATER_RESNAMES
    ]
    ref = reference_rows[0] if reference_rows else None
    final_atoms = rec_result.get("n_atoms") or _count_pdb_atoms(rec_result.get("receptor_pdb"))

    lines = [
        "✓ PDB structure pre-cleaned:",
        "   • ANISOU records removed",
        "   • Optimal alternate locations retained",
        "",
        f"✓ Total atoms parsed: {rec_result.get('total_atoms') or 'recorded in source structure'}",
        "",
    ]

    if ref:
        lines.extend([
            "🎯 Reference ligand identified:",
            f"   • Ligand ID: {rec_result.get('cocrystal_ligand_id') or _format_hetatm_label(ref)}",
            f"   • Chain: {ref.get('chain') or '-'}",
            f"   • Residue number: {ref.get('resid') or '-'}",
            f"   • Ligand atoms: {ref.get('n_atoms') or '-'}",
            "",
        ])
    elif rec_result.get("cocrystal_ligand_id"):
        lines.extend([
            "🎯 Reference ligand identified:",
            f"   • Ligand ID: {rec_result.get('cocrystal_ligand_id')}",
            "",
        ])
    else:
        lines.extend([
            "🎯 Reference ligand identified:",
            "   • None detected; manual docking center was used",
            "",
        ])

    try:
        cx, cy, cz = [float(v) for v in center]
        center_label = "Auto-detected docking center:" if ref else "Manual docking center:"
        lines.extend([
            f"📍 {center_label}",
            f"   • X = {cx:.2f}",
            f"   • Y = {cy:.2f}",
            f"   • Z = {cz:.2f}",
            "",
        ])
    except Exception:
        pass

    lines.extend([
        "🔑 PoseView2 ligand reference:",
        f"   • {rec_result.get('cocrystal_ligand_id') or 'Not available'}",
        "",
        "🧾 HETATM processing summary:",
        f"   • Reference ligand retained: {_format_hetatm_label(ref) if ref else 'None'}",
        (
            "   • Unwanted HETATM entries removed: "
            f"{len(removed_rows)} entries, {len(water_removed)} water molecules"
        ),
        "",
    ])

    if removed_rows:
        lines.append("🧹 Removed entries:")
        preview = removed_rows[:12]
        for row in preview:
            lines.append(f"   • {_format_hetatm_label(row)}")
        if len(removed_rows) > len(preview):
            lines.append(f"   • ... {len(removed_rows) - len(preview)} more entries")
        lines.append("")

    lines.extend([
        f"✓ Final receptor atom count: {final_atoms or 'available in prepared receptor file'}",
        "✓ Polar hydrogens added successfully",
        "✓ Receptor converted to PDBQT format",
        "✓ Receptor PDBQT generation completed",
        "✓ Docking grid box and configuration files generated successfully",
    ])
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
#  RMSD CALCULATION FUNCTIONS (from app11.py - enhanced)
# ═══════════════════════════════════════════════════════════════════════════

def extract_ligand_coords_from_pdb(pdb_file, ligand_resname=None, ligand_chain=None, ligand_resid=None):
    """Extract ligand coordinates from PDB file."""
    coords = []
    atoms = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                if ligand_resname:
                    resname = line[17:20].strip()
                    if resname != ligand_resname:
                        continue
                if ligand_chain is not None and line[21].strip() != str(ligand_chain).strip():
                    continue
                if ligand_resid is not None:
                    try:
                        if int(line[22:26]) != int(ligand_resid):
                            continue
                    except Exception:
                        continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    atom_name = line[12:16].strip()
                    atoms.append(atom_name)
                except:
                    continue
    return np.array(coords), atoms

def calculate_rmsd(coords1, coords2):
    """Calculate RMSD between two sets of coordinates (no alignment)."""
    if len(coords1) != len(coords2):
        min_len = min(len(coords1), len(coords2))
        coords1 = coords1[:min_len]
        coords2 = coords2[:min_len]
    if len(coords1) == 0:
        return None
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def calculate_rmsd_with_alignment(coords1, coords2):
    """Calculate RMSD after optimal superposition (Kabsch algorithm)."""
    if len(coords1) != len(coords2):
        min_len = min(len(coords1), len(coords2))
        coords1 = coords1[:min_len]
        coords2 = coords2[:min_len]
    if len(coords1) == 0:
        return None
    
    # Center the coordinates
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    
    # Calculate optimal rotation using Kabsch algorithm
    H = np.dot(coords1_centered.T, coords2_centered)
    try:
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
        if d < 0:
            Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    except np.linalg.LinAlgError:
        R = np.eye(3)
    
    coords2_aligned = np.dot(coords2_centered, R)
    diff = coords1_centered - coords2_aligned
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def _heavy_mol(mol):
    """Remove hydrogens while preserving the conformer coordinates."""
    if mol is None:
        return None
    try:
        return Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        return Chem.RemoveHs(mol)

def _assign_template_bonds(template, mol):
    """Use the user-provided SMILES to make PDB/SDF atom mapping reliable."""
    if mol is None:
        return None
    if template is None:
        return _heavy_mol(mol)
    try:
        templ = _heavy_mol(template)
        target = _heavy_mol(mol)
        assigned = AllChem.AssignBondOrdersFromTemplate(templ, target)
        Chem.SanitizeMol(assigned)
        return assigned
    except Exception:
        return _heavy_mol(mol)

def _first_sdf_mol(sdf_path):
    """Return the first valid molecule from an SDF file."""
    supp = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    return next((m for m in supp if m is not None), None)

def _sdf_mols(sdf_path):
    """Return all valid molecules from an SDF file."""
    supp = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    return [m for m in supp if m is not None]

def _pose_affinity(mol):
    """Extract docking affinity from molecule properties."""
    if mol is None:
        return None
    for prop in ("minimizedAffinity", "affinity", "docking_score", "ENERGY"):
        if mol.HasProp(prop):
            try:
                return float(re.findall(r"[-+]?\d*\.?\d+", mol.GetProp(prop))[0])
            except Exception:
                pass
    return None

def _direct_rmsd_for_match(ref_mol, docked_mol, ref_match, docked_match):
    """Calculate RMSD for a specific atom mapping."""
    ref_conf = ref_mol.GetConformer()
    docked_conf = docked_mol.GetConformer()
    sq = []
    for ref_idx, docked_idx in zip(ref_match, docked_match):
        rp = ref_conf.GetAtomPosition(int(ref_idx))
        dp = docked_conf.GetAtomPosition(int(docked_idx))
        sq.append((rp.x - dp.x) ** 2 + (rp.y - dp.y) ** 2 + (rp.z - dp.z) ** 2)
    if not sq:
        return None
    return float(np.sqrt(np.mean(sq)))

def _rmsd_against_reference(ref_mol, docked_mol, query=None):
    """Calculate RMSD between reference and docked molecule using substructure matching."""
    if ref_mol is None or docked_mol is None:
        return None
    
    if query is not None:
        ref_matches = ref_mol.GetSubstructMatches(query, uniquify=False, maxMatches=512)
        docked_matches = docked_mol.GetSubstructMatches(query, uniquify=False, maxMatches=512)
        if ref_matches and docked_matches:
            best = float('inf')
            for ref_match in ref_matches:
                for docked_match in docked_matches:
                    val = _direct_rmsd_for_match(ref_mol, docked_mol, ref_match, docked_match)
                    if val is not None and val < best:
                        best = val
            if best < float('inf'):
                return best
    
    ref_heavy = _heavy_mol(ref_mol)
    docked_heavy = _heavy_mol(docked_mol)
    
    if ref_heavy is None or docked_heavy is None:
        return None
    
    ref_conf = ref_heavy.GetConformer()
    docked_conf = docked_heavy.GetConformer()
    
    ref_coords = []
    for atom in ref_heavy.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = ref_conf.GetAtomPosition(atom.GetIdx())
        ref_coords.append([pos.x, pos.y, pos.z])
    
    docked_coords = []
    for atom in docked_heavy.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = docked_conf.GetAtomPosition(atom.GetIdx())
        docked_coords.append([pos.x, pos.y, pos.z])
    
    ref_coords = np.array(ref_coords)
    docked_coords = np.array(docked_coords)
    
    if len(ref_coords) != len(docked_coords) or len(ref_coords) == 0:
        return None
    
    return calculate_rmsd_with_alignment(ref_coords, docked_coords)

def calculate_rmsd_between_ligands(ref_ligand_path, docked_ligand_path, ligand_resname, ligand_smiles=None, return_details=False):
    """Calculate heavy-atom redocking RMSD between reference PDB and docked SDF."""
    try:
        template = Chem.MolFromSmiles(ligand_smiles) if ligand_smiles else None
        
        ref_mol = Chem.MolFromPDBFile(str(ref_ligand_path), sanitize=False, removeHs=False, proximityBonding=True)
        docked_mols = _sdf_mols(docked_ligand_path)

        if ref_mol is None:
            return None
        
        if not docked_mols:
            return None

        ref_mol = _assign_template_bonds(template, ref_mol)
        query = _heavy_mol(template) if template is not None else None

        pose_details = []
        for pose_idx, mol in enumerate(docked_mols, start=1):
            try:
                docked_mol = _assign_template_bonds(template, mol)
                val = _rmsd_against_reference(ref_mol, docked_mol, query)
                if val is not None:
                    pose_details.append({
                        "pose": pose_idx,
                        "rmsd": val,
                        "affinity": _pose_affinity(mol),
                    })
            except Exception as e:
                if hasattr(st, "debug"):
                    st.debug(f"RMSD error for pose {pose_idx}: {e}")
                continue

        if not pose_details:
            return None
        
        best_pose = min(pose_details, key=lambda d: d["rmsd"])
        top_pose = pose_details[0]
        
        if return_details:
            return {
                "top_pose_rmsd": top_pose["rmsd"],
                "best_pose_rmsd": best_pose["rmsd"],
                "best_pose": best_pose["pose"],
                "pose_rmsds": pose_details,
            }
        return top_pose["rmsd"]
        
    except Exception as e:
        if hasattr(st, "debug"):
            st.debug(f"RMSD calculation failed: {e}")
        return None

def calculate_rmsd_prody(ref_pdb, docked_pdb, ligand_resname):
    """Calculate RMSD using ProDy if available."""
    if not PRODY_AVAILABLE:
        return None
    try:
        ref = parsePDB(ref_pdb)
        docked = parsePDB(docked_pdb)
        ref_lig = ref.select(f"resname {ligand_resname}")
        docked_lig = docked.select(f"resname {ligand_resname}")
        if ref_lig is None or docked_lig is None:
            return None
        rmsd = calcRMSD(ref_lig, docked_lig)
        return rmsd
    except Exception as e:
        if hasattr(st, "debug"):
            st.debug(f"ProDy RMSD calculation failed: {e}")
        return None

def get_rmsd_validation_message(rmsd):
    """Return validation message based on RMSD value."""
    if rmsd is None:
        return "❓ Could not calculate RMSD", "unknown"
    elif rmsd < 1.5:
        return f"✅ EXCELLENT! RMSD = {rmsd:.2f} Å (Docking protocol is highly reliable)", "excellent"
    elif rmsd < 2.0:
        return f"✅ GOOD! RMSD = {rmsd:.2f} Å (Docking protocol is reliable)", "good"
    elif rmsd < 3.0:
        return f"⚠️ ACCEPTABLE - RMSD = {rmsd:.2f} Å (Protocol may need optimization)", "acceptable"
    else:
        return f"❌ POOR - RMSD = {rmsd:.2f} Å (Docking protocol needs significant improvement)", "poor"

def quick_rmsd_check_debug(smiles_file, sdf_file, smiles_string=None):
    """Quick RMSD check function for debugging/verification."""
    try:
        if smiles_string:
            template = Chem.MolFromSmiles(smiles_string)
        else:
            template = None
        
        ref_mol = Chem.MolFromMol2File(smiles_file, sanitize=False) if smiles_file.endswith('.mol2') else Chem.MolFromPDBFile(smiles_file, sanitize=False)
        docked_mol = _first_sdf_mol(sdf_file)
        if ref_mol is None or docked_mol is None:
            return None
        ref_mol = _assign_template_bonds(template, ref_mol)
        query = _heavy_mol(template) if template is not None else None
        return _rmsd_against_reference(ref_mol, docked_mol, query)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════
#  BOX & CONFIG FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def ligand_box_from_coords(coords, padding=8.0, min_size=12.0, max_size=24.0):
    """Create a compact redocking box around the crystallographic ligand."""
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return None
    center = coords.mean(axis=0)
    span = coords.max(axis=0) - coords.min(axis=0)
    size = np.clip(span + float(padding), float(min_size), float(max_size))
    return tuple(float(x) for x in (*center, *size))

def write_vina_config(config_path, cx, cy, cz, sx, sy, sz):
    with open(config_path, "w") as f:
        f.write(
            f"center_x = {cx:.4f}\n"
            f"center_y = {cy:.4f}\n"
            f"center_z = {cz:.4f}\n"
            f"size_x = {float(sx):.2f}\n"
            f"size_y = {float(sy):.2f}\n"
            f"size_z = {float(sz):.2f}\n"
        )

def write_box_pdb(filename, cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx/2, sy/2, sz/2
    corners = [(cx+dx, cy+dy, cz+dz) for dx in (-hx,hx) for dy in (-hy,hy) for dz in (-hz,hz)]
    with open(filename, "w") as f:
        for i, (x,y,z) in enumerate(corners, 1):
            f.write(f"HETATM{i:5d}  C   BOX A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("CONECT    1    2    3    5\nCONECT    2    1    4    6\nCONECT    3    1    4    7\nCONECT    4    2    3    8\nCONECT    5    1    6    7\nCONECT    6    2    5    8\nCONECT    7    3    5    8\nCONECT    8    4    6    7\n")

def _split_sdf_blocks(sdf_text):
    blocks = []
    for block in sdf_text.split("$$$$"):
        if block.strip():
            blocks.append(block.rstrip() + "\n$$$$\n")
    return blocks

def render_redocking_pose_viewer(
    original_ligand_pdb,
    docked_sdf,
    height=520,
    receptor_pdb=None,
    redock_result=None,
    key_prefix="redock",
):
    """Render an Anyone Can Dock-style redocking pose browser."""
    if not original_ligand_pdb or not docked_sdf:
        return
    if not os.path.exists(original_ligand_pdb) or not os.path.exists(docked_sdf):
        return
    try:
        with open(original_ligand_pdb, "r") as f:
            original_pdb = f.read()
        with open(docked_sdf, "r") as f:
            docked_sdf_text = f.read()
        receptor_text = ""
        if receptor_pdb and os.path.exists(receptor_pdb):
            with open(receptor_pdb, "r") as f:
                receptor_text = f.read()
    except Exception as e:
        st.warning(f"Could not load pose viewer files: {e}")
        return

    sdf_blocks = _split_sdf_blocks(docked_sdf_text)
    if not sdf_blocks:
        st.warning("No redocked poses found in SDF.")
        return

    pose_count = len(sdf_blocks)
    pose_idx = 0

    selected_sdf = sdf_blocks[pose_idx]
    pose_rows = ((redock_result or {}).get("rmsd_details") or {}).get("pose_rmsds") or []
    pose_row = next((r for r in pose_rows if int(r.get("pose", -1)) == pose_idx + 1), None)
    if pose_row is None and pose_idx < len(pose_rows):
        pose_row = pose_rows[pose_idx]

    score = None
    rmsd = None
    if pose_row:
        score = pose_row.get("affinity")
        rmsd = pose_row.get("rmsd")
    if score is None and redock_result:
        scores = parse_all_poses(redock_result.get("docked_pdbqt"), docked_sdf, st.session_state.get("current_engine", "VINA"))
        score = next((p.get("affinity") for p in scores if p.get("pose") == pose_idx + 1), None)
    if rmsd is None and redock_result and pose_idx == 0:
        rmsd = redock_result.get("rmsd")

    viewer_id = f"redock_viewer_{key_prefix}_{pose_idx}_{int(time.time() * 1000)}"
    html = f"""
    <div style="border:1px solid #d0d7de;border-radius:8px;overflow:hidden;background:#ffffff;">
      <div id="{viewer_id}" style="width:100%;height:{height}px;"></div>
      <div style="display:flex;gap:18px;align-items:center;padding:8px 12px;font-family:IBM Plex Sans,Arial,sans-serif;font-size:13px;border-top:1px solid #d0d7de;background:#f6f8fa;">
        <span><span style="display:inline-block;width:12px;height:12px;background:#7c3aed;border-radius:2px;margin-right:6px;"></span>Co-crystal pose</span>
        <span><span style="display:inline-block;width:12px;height:12px;background:#2563eb;border-radius:2px;margin-right:6px;"></span>Redocked pose</span>
      </div>
    </div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      var originalPdb = {_json.dumps(original_pdb)};
      var dockedSdf = {_json.dumps(selected_sdf)};
      function initRedockViewer(attempt) {{
        if (!window.$3Dmol) {{
          if (attempt < 50) {{
            window.setTimeout(function() {{ initRedockViewer(attempt + 1); }}, 100);
          }}
          return;
        }}
        var el = document.getElementById("{viewer_id}");
        if (!el) return;
        el.innerHTML = "";
        var viewer = $3Dmol.createViewer("{viewer_id}", {{backgroundColor: "white"}});
        var modelIndex = 0;
        viewer.addModel(originalPdb, "pdb");
        var crystalModel = modelIndex;
        viewer.setStyle({{model: crystalModel}}, {{
          stick: {{color: "#7c3aed", radius: 0.22}},
          sphere: {{color: "#7c3aed", scale: 0.18}}
        }});
        modelIndex += 1;
        viewer.addModel(dockedSdf, "sdf");
        var dockedModel = modelIndex;
        viewer.setStyle({{model: dockedModel}}, {{
          stick: {{color: "#2563eb", radius: 0.28}},
          sphere: {{color: "#2563eb", scale: 0.20}}
        }});
        viewer.zoomTo({{model: dockedModel}});
        viewer.render();
      }}
      initRedockViewer(0);
    </script>
    """
    components.html(html, height=height + 48)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            f"⬇ Redocked pose {pose_idx + 1} (.sdf)",
            data=selected_sdf,
            file_name=f"redocked_pose_{pose_idx + 1}.sdf",
            mime="chemical/x-mdl-sdfile",
            key=f"{key_prefix}_dl_pose_sdf_{pose_idx}",
            use_container_width=True,
        )
    with col_dl2:
        if redock_result and redock_result.get("docked_pdbqt") and os.path.exists(redock_result["docked_pdbqt"]):
            with open(redock_result["docked_pdbqt"], "rb") as f:
                st.download_button(
                    "⬇ All redocked poses (.pdbqt)",
                    data=f,
                    file_name=os.path.basename(redock_result["docked_pdbqt"]),
                    mime="chemical/x-pdbqt",
                    key=f"{key_prefix}_dl_all_pdbqt",
                    use_container_width=True,
                )

# ═══════════════════════════════════════════════════════════════════════════
#  REDOCKING FUNCTION WITH SMILES INPUT
# ═══════════════════════════════════════════════════════════════════════════

def redock_cocrystal_ligand_from_smiles(rec_result, cocrystal_smiles, cocrystal_resname, 
                                         engine, dock_bin, dock_params, workdir, source_pdb=None,
                                         cocrystal_chain=None, cocrystal_resid=None):
    """Redock a co-crystal ligand from SMILES string to validate docking protocol."""
    try:
        receptor_pdb = source_pdb or rec_result.get('raw_pdb') or rec_result.get('receptor_pdb')
        original_coords, original_atoms = extract_ligand_coords_from_pdb(
            receptor_pdb, cocrystal_resname, cocrystal_chain, cocrystal_resid
        )
        
        if len(original_coords) == 0:
            loc = f" chain {cocrystal_chain} resid {cocrystal_resid}" if cocrystal_chain or cocrystal_resid else ""
            st.error(f"Could not extract co-crystal ligand '{cocrystal_resname}'{loc} from original PDB file: {receptor_pdb}")
            return None
                
        # Write original ligand to separate PDB file
        original_ligand_pdb = str(workdir / f"original_{cocrystal_resname}.pdb")
        with open(receptor_pdb, 'r') as f_in, open(original_ligand_pdb, 'w') as f_out:
            for line in f_in:
                if line.startswith(('ATOM', 'HETATM')):
                    resname = line[17:20].strip()
                    chain_ok = cocrystal_chain is None or line[21].strip() == str(cocrystal_chain).strip()
                    try:
                        resid_ok = cocrystal_resid is None or int(line[22:26]) == int(cocrystal_resid)
                    except Exception:
                        resid_ok = False
                    if resname == cocrystal_resname and chain_ok and resid_ok:
                        f_out.write(line)
            f_out.write("END\n")
        
        lig_name = f"cocrystal_{cocrystal_resname}"
        lig_prep = prepare_ligand_vina_batch(cocrystal_smiles, lig_name, 7.4, workdir)
        
        if not lig_prep['success']:
            st.error("Failed to prepare co-crystal ligand from SMILES")
            return None
        
        lig_input = lig_prep["sdf"] if engine == "UNIDOCK" else lig_prep["pdbqt"]
        out_prefix = str(workdir / f"redock_{cocrystal_resname}")
        config_file = rec_result["config_file"]
        
        # Optionally use a compact box around the crystal ligand
        redock_box = None
        if dock_params.get("use_crystal_box", True):
            redock_box = ligand_box_from_coords(
                original_coords,
                padding=dock_params.get("redock_box_padding", 8.0),
                min_size=dock_params.get("redock_min_box_size", 12.0),
                max_size=dock_params.get("redock_max_box_size", 24.0),
            )
            if redock_box:
                cx, cy, cz, sx, sy, sz = redock_box
                config_file = str(workdir / f"redock_{cocrystal_resname}.box.txt")
                box_pdb = str(workdir / f"redock_{cocrystal_resname}.box.pdb")
                write_vina_config(config_file, cx, cy, cz, sx, sy, sz)
                write_box_pdb(box_pdb, cx, cy, cz, sx, sy, sz)
        
        # Run docking
        docked_pdbqt, docked_sdf, dock_log = dock_one_ligand(
            engine, dock_bin,
            rec_result["receptor_pdbqt"],
            lig_input, config_file,
            out_prefix, dock_params
        )
        
        if not docked_sdf or not os.path.exists(docked_sdf):
            st.error("Docking failed - no output file")
            return None
        
        # Calculate RMSD
        rmsd_details = calculate_rmsd_between_ligands(
            original_ligand_pdb, docked_sdf, cocrystal_resname, cocrystal_smiles, return_details=True
        )
        
        if isinstance(rmsd_details, dict):
            rmsd = rmsd_details.get("top_pose_rmsd")
        else:
            rmsd = rmsd_details

        pose_scores = parse_all_poses(docked_pdbqt, docked_sdf, engine)
        pose_affinities = {
            p.get("pose"): p.get("affinity")
            for p in pose_scores
            if p.get("pose") is not None
        }
        binding_affinity = pose_affinities.get(1)
        if binding_affinity is None and pose_scores:
            binding_affinity = pose_scores[0].get("affinity")

        if isinstance(rmsd_details, dict):
            for row in rmsd_details.get("pose_rmsds") or []:
                if row.get("affinity") is None:
                    row["affinity"] = pose_affinities.get(row.get("pose"))
            rmsd_details["binding_affinity"] = binding_affinity
        
        # Try ProDy RMSD as alternative
        rmsd_prody = None
        if PRODY_AVAILABLE:
            rmsd_prody = calculate_rmsd_prody(original_ligand_pdb, docked_sdf, cocrystal_resname)
        
        # Get atom counts
        try:
            template = Chem.MolFromSmiles(cocrystal_smiles)
            n_heavy_docked = template.GetNumHeavyAtoms() if template else 0
        except:
            n_heavy_docked = 0
        
        return {
            'success': True,
            'rmsd': rmsd,
            'rmsd_prody': rmsd_prody,
            'docked_sdf': docked_sdf,
            'docked_pdbqt': docked_pdbqt,
            'original_ligand_pdb': original_ligand_pdb,
            'original_smiles': cocrystal_smiles,
            'original_resname': cocrystal_resname,
            'ligand_name': lig_name,
            'log': dock_log,
            'binding_affinity': binding_affinity,
            'n_atoms_original': len(original_coords),
            'n_atoms_docked': n_heavy_docked,
            'rmsd_method': 'heavy_atom_symmetry_aware_top_pose',
            'rmsd_details': rmsd_details if isinstance(rmsd_details, dict) else None,
            'redock_box': redock_box,
            'redock_config_file': config_file,
        }
    except Exception as e:
        st.error(f"Redocking failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
#  RCSB PDB SEARCH & DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

def search_rcsb_pdb(query, top_n=20):
    """Search RCSB PDB database for structures matching query."""
    try:
        if re.match(r'^[0-9a-zA-Z]{4}$', query.upper()):
            url = f"https://data.rcsb.org/rest/v1/core/entry/{query.upper()}"
            response = get_session().get(url, timeout=15)
            if response.status_code == 200:
                entry = response.json()
                return [{
                    "pdb_id": query.upper(),
                    "title": entry.get("struct", {}).get("title", ""),
                    "resolution": entry.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
                    "method": entry.get("exptl", [{}])[0].get("method", ""),
                    "organism": entry.get("rcsb_entity_source_organism", [{}])[0].get("common_name", ""),
                    "deposition_date": entry.get("rcsb_accession_info", {}).get("deposit_date", "")
                }]
        
        payload = {
            "query": {"type": "terminal", "service": "full_text", "parameters": {"value": query}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": top_n}, "results_verbosity": "compact", "sort": [{"sort_by": "score", "direction": "desc"}]},
        }
        r = get_session().post("https://search.rcsb.org/rcsbsearch/v2/query", json=payload, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, dict):
            return []
        result_set = data.get("result_set", [])
        if not result_set:
            return []
        results = []
        for hit in result_set:
            try:
                pdb_id = hit.get("identifier", "") if isinstance(hit, dict) else str(hit) if hit else ""
                if not pdb_id:
                    continue
                detail_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
                try:
                    r2 = get_session().get(detail_url, timeout=10)
                    if r2.status_code == 200:
                        entry = r2.json()
                        results.append({
                            "pdb_id": pdb_id,
                            "title": entry.get("struct", {}).get("title", "")[:120],
                            "resolution": entry.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0] or "N/A",
                            "method": entry.get("exptl", [{}])[0].get("method", "") or "N/A",
                            "organism": entry.get("rcsb_entity_source_organism", [{}])[0].get("common_name", "") or "N/A",
                            "deposition_date": entry.get("rcsb_accession_info", {}).get("deposit_date", "")
                        })
                    else:
                        results.append({"pdb_id": pdb_id, "title": "", "resolution": "N/A", "method": "N/A", "organism": "N/A", "deposition_date": ""})
                except Exception:
                    results.append({"pdb_id": pdb_id, "title": "", "resolution": "N/A", "method": "N/A", "organism": "N/A", "deposition_date": ""})
            except Exception:
                continue
        return results
    except Exception as e:
        st.error(f"RCSB search error: {str(e)}")
        return []

def download_pdb_direct(pdb_id, output_path):
    """Direct download of PDB file from RCSB by ID."""
    try:
        pdb_id = pdb_id.upper().strip()
        urls = [
            f"https://files.rcsb.org/download/{pdb_id}.pdb",
            f"https://models.rcsb.org/{pdb_id}.pdb",
            f"https://www.rcsb.org/pdb/files/{pdb_id}.pdb",
            f"https://files.rcsb.org/download/{pdb_id}.cif",
        ]
        for url in urls:
            try:
                response = get_session().get(url, timeout=30)
                if response.status_code == 200:
                    content = response.text
                    if content.strip().startswith(("HEADER", "ATOM", "HETATM", "CRYST1", "REMARK", "data_")):
                        with open(output_path, "w") as f:
                            f.write(content)
                        if os.path.getsize(output_path) > 1000:
                            return True
            except Exception:
                continue
        return False
    except Exception as e:
        st.error(f"Download error for {pdb_id}: {str(e)}")
        return False

def get_pdb_info(pdb_id):
    """Get information about a PDB entry."""
    try:
        pdb_id = pdb_id.upper().strip()
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        response = get_session().get(url, timeout=10)
        if response.status_code == 200:
            entry = response.json()
            return {
                "title": entry.get("struct", {}).get("title", ""),
                "resolution": entry.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
                "method": entry.get("exptl", [{}])[0].get("method", ""),
                "organism": entry.get("rcsb_entity_source_organism", [{}])[0].get("common_name", ""),
                "deposition_date": entry.get("rcsb_accession_info", {}).get("deposit_date", "")
            }
        return None
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════
#  CO-CRYSTAL LIGAND DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _safe_resname(x):
    return (x or "").strip()

def _hetatm_key(resname, chain, resid):
    return f"{str(resname).strip().upper()}|{str(chain or '').strip() or '_'}|{int(resid)}"

def _guess_hetatm_type(resname, n_atoms):
    rn = _safe_resname(resname).upper()
    if rn in WATER_RESNAMES:
        return "water"
    if rn in METAL_RESNAMES:
        return "metal"
    if rn in HEME_RESNAMES:
        return "heme/cofactor"
    if rn in COFACTOR_NAMES:
        return "cofactor"
    if rn in BUFFER_RESNAMES or n_atoms <= 3:
        return "buffer/ion"
    if n_atoms >= _MIN_LIG_ATOMS:
        return "ligand"
    return "other"

def _default_hetatm_action(type_guess, n_atoms):
    if type_guess == "water":
        return "remove"
    if type_guess == "metal":
        return "keep"
    if "cofactor" in type_guess:
        return "keep"
    if type_guess == "ligand":
        return "reference" if n_atoms >= _MIN_LIG_ATOMS else "remove"
    return "remove"

def _collect_hetatm_residues(atoms):
    het = atoms.select("hetatm")
    if het is None:
        return []
    rows = []
    for res in het.getHierView().iterResidues():
        rn = _safe_resname(res.getResname()).upper()
        ch = (res.getChid() or "").strip()
        ri = int(res.getResnum())
        n_atoms = int(res.numAtoms())
        type_guess = _guess_hetatm_type(rn, n_atoms)
        sel = f"resname {rn} and resid {ri} and chain {ch}" if ch else f"resname {rn} and resid {ri}"
        res_atoms = atoms.select(sel)
        if res_atoms is None:
            continue
        cx, cy, cz = (float(v) for v in calcCenter(res_atoms))
        action = _default_hetatm_action(type_guess, n_atoms)
        rows.append({
            "key": _hetatm_key(rn, ch, ri),
            "resname": rn,
            "chain": ch,
            "resid": ri,
            "n_atoms": n_atoms,
            "type_guess": type_guess,
            "default_action": action,
            "action": action,
            "sel_str": sel,
            "cx": cx,
            "cy": cy,
            "cz": cz,
        })
    rank = {
        "ligand": 0,
        "heme/cofactor": 1,
        "cofactor": 2,
        "metal": 3,
        "buffer/ion": 4,
        "other": 5,
        "water": 6,
    }
    rows.sort(key=lambda d: (rank.get(d["type_guess"], 9), -d["n_atoms"], d["resname"], d["chain"], d["resid"]))
    seen_reference = False
    for row in rows:
        if row["type_guess"] == "ligand":
            if not seen_reference:
                row["default_action"] = row["action"] = "reference"
                seen_reference = True
            else:
                row["default_action"] = row["action"] = "remove"
    return rows

def _choose_ligand_candidates(ligands):
    if not ligands:
        return [], None, "no_ligand"
    chain_a = [row for row in ligands if str(row.get("chain", "")).strip().upper() == "A"]
    resnames = {str(row.get("resname", "")).strip().upper() for row in ligands}
    chains = [str(row.get("chain", "")).strip().upper() or "_" for row in ligands]
    one_per_chain = all(chains.count(c) == 1 for c in set(chains))
    if len(ligands) == 1:
        return ligands, ligands[0], "single_ligand"
    if len(resnames) == 1 and len(chain_a) == 1 and one_per_chain:
        return chain_a, chain_a[0], "homo_multimer_chain_a"
    if len(chain_a) > 1:
        return chain_a, None, "multiple_chain_a_ligands"
    return ligands, None, "multiple_ligands"

def _collect_removable_ligands(atoms, exclude_glycans=True, exclude_cofactors=True):
    excl = EXCLUDE_IONS | HEME_RESNAMES | METAL_RESNAMES
    if exclude_glycans:
        excl |= GLYCAN_NAMES
    if exclude_cofactors:
        excl |= COFACTOR_NAMES
    het = atoms.select("hetatm and not water")
    if het is None:
        return []
    results = []
    for res in het.getHierView().iterResidues():
        rn = _safe_resname(res.getResname()).upper()
        if rn in excl or res.numAtoms() <= _MIN_LIG_ATOMS:
            continue
        if _BACKBONE.issubset(set(res.getNames())):
            continue
        ch = (res.getChid() or "").strip()
        ri = res.getResnum()
        sel = (f"resname {rn} and resid {ri} and chain {ch}" if ch else f"resname {rn} and resid {ri}")
        lig_atoms = atoms.select(sel)
        if lig_atoms is None or lig_atoms.numAtoms() == 0:
            continue
        cx, cy, cz = (float(v) for v in calcCenter(lig_atoms))
        results.append({
            "resname": rn, "chain": ch, "resid": ri,
            "n_atoms": lig_atoms.numAtoms(),
            "cx": cx, "cy": cy, "cz": cz, "sel_str": sel
        })
    results.sort(key=lambda d: (-d["n_atoms"], d["chain"] != "A"))
    return results

def detect_cocrystal_ligand(pdb_path):
    if not PRODY_AVAILABLE:
        return {"found": False, "error": "ProDy not installed"}
    if ACD_CORE_AVAILABLE:
        try:
            working_path = str(pdb_path)
            if _ACD_CORE.is_cif_file(working_path):
                tmp_pdb = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb").name
                conv = _ACD_CORE.convert_cif_to_pdb(working_path, tmp_pdb)
                if conv.get("success"):
                    working_path = conv["pdb_path"]
                else:
                    return {"found": False, "error": conv.get("error", "CIF conversion failed")}
            working_path = _ACD_CORE._clean_pdb_file(working_path)
            atoms = parsePDB(working_path)
            if atoms is None:
                return {"found": False, "error": "Could not parse PDB"}

            rows = _ACD_CORE._collect_hetatm_residues(atoms)
            ligands = [r for r in rows if str(r.get("type_guess", "")).lower() == "ligand"]
            if not ligands:
                return {"found": False, "error": "No drug-like ligand found"}
            candidates, chosen, reason = _choose_ligand_candidates(ligands)
            if chosen is None:
                chosen = candidates[0]
            return {
                "found": True,
                "resname": chosen["resname"],
                "chain": chosen["chain"],
                "resid": chosen["resid"],
                "n_atoms": chosen["n_atoms"],
                "center": (chosen["cx"], chosen["cy"], chosen["cz"]),
                "sel_str": chosen.get("sel_str"),
                "key": chosen["key"],
                "message": (
                    f"{chosen['resname']} (chain {chosen['chain'] or '-'}, "
                    f"resid {chosen['resid']}, {chosen['n_atoms']} atoms)"
                ),
                "all_candidates": candidates,
                "selection_reason": reason,
                "hetatm_rows": [
                    {k: r.get(k) for k in ("key", "resname", "chain", "resid", "n_atoms", "type_guess", "default_action")}
                    for r in rows
                ],
            }
        except Exception as e:
            return {"found": False, "error": str(e)}
    try:
        atoms = parsePDB(pdb_path)
        if atoms is None:
            return {"found": False, "error": "Could not parse PDB"}
        candidates = _collect_removable_ligands(atoms)
        if not candidates:
            return {"found": False, "error": "No drug-like ligand found"}
        best = candidates[0]
        return {
            "found": True,
            "resname": best["resname"],
            "chain": best["chain"],
            "resid": best["resid"],
            "n_atoms": best["n_atoms"],
            "center": (best["cx"], best["cy"], best["cz"]),
            "sel_str": best.get("sel_str"),
            "key": _hetatm_key(best["resname"], best["chain"], best["resid"]),
            "message": f"{best['resname']} (chain {best['chain']}, resid {best['resid']}, {best['n_atoms']} atoms)",
            "all_candidates": candidates,
        }
    except Exception as e:
        return {"found": False, "error": str(e)}

# ═══════════════════════════════════════════════════════════════════════════
#  DOCKING ENGINE BINARIES
# ═══════════════════════════════════════════════════════════════════════════

VINA_SEARCH_PATHS = {
    "VINA":    ["/content/vina",    "./vina",    "vina",    "/usr/local/bin/vina"],
    "VINAXB":  ["/content/vinaXB",  "./vinaXB",  "vinaXB",  "./vina_xb"],
    "GNINA":   ["/content/gnina",   "./gnina",   "gnina",   "/usr/local/bin/gnina"],
    "UNIDOCK": ["/content/unidock", "./unidock", "unidock", "/usr/local/bin/unidock"],
}

def _file_is_executable(path):
    return os.path.isfile(path) and os.access(path, os.X_OK) and os.path.getsize(path) > 1000

def find_docking_binary(engine, custom=""):
    engine = (engine or "").upper().strip()
    if custom and custom.strip():
        cand = custom.strip()
        if engine == "GNINA" and cand.startswith("docker:"):
            ok, msg = docker_is_ready()
            if ok:
                return cand, cand
            raise RuntimeError(f"Docker GNINA requested but Docker is not ready: {msg}")
        if _file_is_executable(cand):
            return cand, "custom"
        raise RuntimeError(f"Custom binary not found or not executable: {cand}")
    for cand in VINA_SEARCH_PATHS.get(engine, []):
        if _file_is_executable(cand):
            return cand, cand
    if engine == "GNINA":
        ok, msg = docker_is_ready()
        if ok:
            return "docker:gnina/gnina", "Docker image gnina/gnina"
    return None, None

def check_obabel():
    return shutil.which("obabel") is not None

def docker_is_ready():
    if shutil.which("docker") is None:
        return False, "Docker command not found"
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return True, "Docker is running"
        return False, (r.stderr or r.stdout or "Docker is not running").strip()
    except Exception as e:
        return False, str(e)

def _is_docker_gnina(dock_bin):
    return isinstance(dock_bin, str) and dock_bin.startswith("docker:")

def _docker_work_path(path):
    abs_path = os.path.abspath(str(path))
    cwd = os.path.abspath(os.getcwd())
    try:
        rel = os.path.relpath(abs_path, cwd)
        if not rel.startswith(".."):
            return "/work/" + rel.replace(os.sep, "/")
    except Exception:
        pass
    return "/work/" + os.path.basename(abs_path)

# ═══════════════════════════════════════════════════════════════════════════
#  RECEPTOR PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def prepare_receptor_vina_batch(raw_pdb, workdir, cx, cy, cz, sx, sy, sz):
    if ACD_CORE_AVAILABLE:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        reference_key = ""
        try:
            info = st.session_state.get("cocrystal_info") or {}
            reference_key = info.get("key") or ""
        except Exception:
            reference_key = ""

        hetatm_policy = None
        try:
            rows = _ACD_CORE.scan_hetatm_residues(str(raw_pdb))
            if rows:
                hetatm_policy = {}
                ligand_keys = {
                    row["key"]
                    for row in rows
                    if str(row.get("type_guess", "")).lower() == "ligand"
                }
                if not reference_key and len(ligand_keys) == 1:
                    reference_key = next(iter(ligand_keys))
                for row in rows:
                    key = row["key"]
                    action = str(row.get("default_action", "remove")).lower()
                    if key == reference_key:
                        action = "reference"
                    elif key in ligand_keys:
                        action = "remove"
                    hetatm_policy[key] = action
        except Exception:
            hetatm_policy = None

        ui_center_mode = st.session_state.get("box_center_mode", "")
        center_mode = "manual" if ui_center_mode == "Enter XYZ manually" else ("auto" if reference_key else "manual")
        res = _ACD_CORE.prepare_receptor(
            str(raw_pdb),
            workdir,
            center_mode=center_mode,
            manual_xyz=(cx, cy, cz),
            box_size=(sx, sy, sz),
            hetatm_policy=hetatm_policy,
            reference_hetatm_key=reference_key,
        )
        if not res.get("success"):
            raise RuntimeError(res.get("error", "Anyone Can Dock receptor preparation failed"))
        return {
            "success": True,
            "raw_pdb": str(raw_pdb),
            "receptor_pdb": res.get("rec_fh"),
            "receptor_pdbqt": res.get("rec_pdbqt"),
            "config_file": res.get("config_txt"),
            "box_pdb": res.get("box_pdb"),
            "ligand_pdb_path": res.get("ligand_pdb_path"),
            "cocrystal_ligand_id": res.get("cocrystal_ligand_id"),
            "n_atoms": res.get("n_atoms"),
            "hetatm_table": res.get("hetatm_table", []),
            "center": (res.get("cx"), res.get("cy"), res.get("cz")),
            "box_size": (res.get("sx"), res.get("sy"), res.get("sz")),
            "log": ["✓ Anyone Can Dock receptor protocol"] + res.get("log", []),
        }

    workdir = Path(workdir)
    log = []
    receptor_pdb = str(workdir / "receptor_atoms.pdb")
    rec_nometal  = str(workdir / "receptor_nometal.pdb")
    receptor_fh  = str(workdir / "rec.pdb")
    receptor_pdbqt = str(workdir / "rec.pdbqt")
    config_file  = str(workdir / "rec.box.txt")
    box_pdb      = str(workdir / "rec.box.pdb")
    
    if PRODY_AVAILABLE:
        try:
            atoms = parsePDB(raw_pdb)
            all_lig = _collect_removable_ligands(atoms)
            if all_lig:
                excl_expr = " or ".join(f"({d['sel_str']})" for d in all_lig)
                sel_str = f"not ({excl_expr}) and not water"
            else:
                sel_str = "not water"
            rec_atoms = atoms.select(sel_str)
            if rec_atoms is not None and rec_atoms.numAtoms() > 0:
                writePDB(receptor_pdb, rec_atoms)
                log.append(f"✓ Receptor selected ({rec_atoms.numAtoms()} atoms)")
            else:
                shutil.copy(raw_pdb, receptor_pdb)
                log.append("⚠ ProDy selection returned no atoms — using raw PDB")
        except Exception as e:
            shutil.copy(raw_pdb, receptor_pdb)
            log.append(f"⚠ ProDy error ({e}) — using raw PDB")
    else:
        shutil.copy(raw_pdb, receptor_pdb)
        log.append("ℹ ProDy not available — using raw PDB as receptor")
    
    metal_lines, clean_lines = [], []
    with open(receptor_pdb) as f:
        for line in f:
            field = line[:6].strip()
            if field in ("ATOM","HETATM") and line[17:20].strip().upper() in METAL_RESNAMES:
                metal_lines.append(line)
            else:
                clean_lines.append(line)
    with open(rec_nometal, "w") as f:
        f.writelines(clean_lines)
    if metal_lines:
        log.append(f"⚠ Stripped {len(metal_lines)} metal atoms before OpenBabel")
    
    r = subprocess.run(f'obabel "{rec_nometal}" -O "{receptor_fh}" -p 7.4 2>/dev/null', shell=True, capture_output=True, text=True)
    if not os.path.exists(receptor_fh) or os.path.getsize(receptor_fh) < 100:
        raise RuntimeError("Hydrogen addition failed (obabel)")
    log.append("✓ Polar hydrogens added to receptor (pH 7.4)")
    
    if metal_lines:
        lines = [l for l in open(receptor_fh).readlines() if l.strip() != "END"]
        lines.extend(metal_lines)
        lines.append("END\n")
        with open(receptor_fh, "w") as f:
            f.writelines(lines)
    
    r = subprocess.run(f'obabel "{rec_nometal}" -O "{receptor_pdbqt}" -xr --partialcharge gasteiger 2>/dev/null', shell=True, capture_output=True, text=True)
    if not os.path.exists(receptor_pdbqt) or os.path.getsize(receptor_pdbqt) < 100:
        raise RuntimeError("PDBQT conversion failed (obabel)")
    log.append("✓ Receptor converted to PDBQT")
    
    if metal_lines:
        pdbqt_lines = [l for l in open(receptor_pdbqt).readlines() if l.strip() != "END"]
        injected = 0
        for ml in metal_lines:
            try:
                resname = ml[17:20].strip().upper()
                if resname in _NO_REINJECT:
                    continue
                serial = int(ml[6:11])
                name = ml[12:16].strip()
                chain = ml[21] if len(ml) > 21 else "A"
                resid = int(ml[22:26])
                x, y, z = float(ml[30:38]), float(ml[38:46]), float(ml[46:54])
                charge = METAL_CHARGES.get(resname, 0.0)
                atype = resname.capitalize()
                pdbqt_lines.append(f"HETATM{serial:5d} {name:<4s} {resname:<3s} {chain}{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {charge:+.3f} {atype}\n")
                injected += 1
            except Exception:
                pass
        pdbqt_lines.append("END\n")
        with open(receptor_pdbqt, "w") as f:
            f.writelines(pdbqt_lines)
        if injected:
            log.append(f"✓ Re-injected {injected} metal atoms into PDBQT")
    
    write_box_pdb(box_pdb, cx, cy, cz, sx, sy, sz)
    write_vina_config(config_file, cx, cy, cz, sx, sy, sz)
    log.append(f"✓ Box: {sx}×{sy}×{sz} Å at ({cx:.2f},{cy:.2f},{cz:.2f})")
    log.append(f"✓ Config: {config_file}")
    
    try:
        rec_lines = open(receptor_fh).readlines()
        coord_lines = [l for l in rec_lines if l[:6].strip() in ("ATOM","HETATM")]
        all_blank = all((l[21]==" " if len(l)>21 else True) for l in coord_lines)
        if all_blank and coord_lines:
            fixed = []
            for l in rec_lines:
                if l[:6].strip() in ("ATOM","HETATM") and len(l) > 21:
                    l = l[:21] + "A" + l[22:]
                fixed.append(l)
            with open(receptor_fh, "w") as f:
                f.writelines(fixed)
            log.append("✓ Assigned chain A to blank-chain atoms")
    except Exception:
        pass
    
    return {
        "success": True,
        "raw_pdb": str(raw_pdb),
        "receptor_pdb": receptor_fh,
        "receptor_pdbqt": receptor_pdbqt,
        "config_file": config_file,
        "box_pdb": box_pdb,
        "log": log
    }

# ═══════════════════════════════════════════════════════════════════════════
#  LIGAND PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def ph_adjust_smiles(smiles_str, ph=7.4):
    if DIMORPHITE_AVAILABLE:
        try:
            prot_list = _dimorphite_protonate(smiles_str, ph_min=ph, ph_max=ph, max_variants=4)
            candidates = []
            for smi in prot_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
                net = int(sum(charges))
                candidates.append((smi, net))
            if candidates:
                candidates.sort(key=lambda x: abs(x[1]))
                return candidates[0][0]
        except Exception:
            pass
    return smiles_str

def build_3d_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
    except AttributeError:
        params = AllChem.ETKDG()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol

def write_pdbqt_meeko(mol3d, out_pdbqt):
    if not MEEKO_AVAILABLE:
        raise RuntimeError("Meeko not available")
    mol3d = Chem.Mol(mol3d)
    if any(a.GetNumImplicitHs() > 0 for a in mol3d.GetAtoms()):
        mol3d = Chem.AddHs(mol3d, addCoords=True)
    prep = MoleculePreparation()
    if MEEKO_LEGACY:
        setups = prep.prepare(mol3d)
        pdbqt_str, _, _ = _MeekoWriter.write_string(setups[0])
    else:
        prep.prepare(mol3d)
        pdbqt_str = prep.write_pdbqt_string()
    with open(out_pdbqt, "w") as f:
        f.write(pdbqt_str)
    assert_file_ok(out_pdbqt, 100, f"PDBQT write failed: {out_pdbqt}")

def prepare_ligand_vina_batch(smiles, lig_name, ph, workdir):
    if ACD_CORE_AVAILABLE:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        prefix = safe_name(lig_name)
        res = _ACD_CORE.prepare_ligand(
            smiles,
            prefix,
            ph,
            workdir,
            mode="pkanet",
            use_pubchem=True,
            max_tautomers=8,
            ph_window=1.0,
            pkanet_selection_mode="auto_recommended",
        )
        if not res.get("success"):
            raise RuntimeError(res.get("error", "Anyone Can Dock ligand preparation failed"))
        return {
            "success": True,
            "name": lig_name,
            "pdb": res.get("pdb"),
            "sdf": res.get("sdf"),
            "pdbqt": res.get("pdbqt"),
            "ph_smiles": res.get("prepared_smiles") or res.get("prot_smiles") or smiles,
            "prepared_smiles": res.get("prepared_smiles") or res.get("prot_smiles"),
            "charge": res.get("charge"),
            "pkanet_ranked_csv": res.get("pkanet_ranked_csv"),
            "pkanet_decision_log": res.get("pkanet_decision_log"),
            "log": ["✓ Anyone Can Dock ligand protocol"] + res.get("log", []),
        }

    workdir = Path(workdir)
    log = []
    prefix = safe_name(lig_name)
    out_pdb   = str(workdir / f"{prefix}_min.pdb")
    out_sdf   = str(workdir / f"{prefix}_min.sdf")
    out_pdbqt = str(workdir / f"{prefix}_min.pdbqt")
    
    ph_smiles = ph_adjust_smiles(smiles, ph)
    log.append(f"✓ SMILES @ pH {ph}: {ph_smiles[:60]}")
    mol = build_3d_mol(ph_smiles)
    log.append(f"✓ 3D conformer: {mol.GetNumAtoms()} atoms")
    Chem.MolToPDBFile(mol, out_pdb)
    with Chem.SDWriter(out_sdf) as w:
        w.write(mol)
    log.append(f"✓ PDB: {out_pdb}")
    log.append(f"✓ SDF: {out_sdf}")
    
    try:
        write_pdbqt_meeko(mol, out_pdbqt)
        log.append(f"✓ PDBQT (meeko): {out_pdbqt}")
    except Exception as e:
        log.append(f"⚠ Meeko failed ({e}) — trying obabel fallback")
        r = subprocess.run(f'obabel "{out_sdf}" -O "{out_pdbqt}" -xh 2>/dev/null', shell=True, capture_output=True, text=True)
        if not os.path.exists(out_pdbqt) or os.path.getsize(out_pdbqt) < 100:
            raise RuntimeError(f"PDBQT conversion failed for {lig_name}")
        log.append(f"✓ PDBQT (obabel): {out_pdbqt}")
    
    return {
        "success": True,
        "name": lig_name,
        "pdb": out_pdb,
        "sdf": out_sdf,
        "pdbqt": out_pdbqt,
        "ph_smiles": ph_smiles,
        "log": log
    }

# ═══════════════════════════════════════════════════════════════════════════
#  SIMPLE RMSD UTILITY FOR LIGAND VALIDATION (NEW SECTION)
# ═══════════════════════════════════════════════════════════════════════════

def validate_ligand_preparation_with_rmsd(smiles_string, prepared_sdf_path, reference_pdb=None):
    """
    Validate ligand preparation by calculating RMSD between the prepared structure
    and a reference structure (if available).
    """
    try:
        template = Chem.MolFromSmiles(smiles_string)
        prepared_mol = _first_sdf_mol(prepared_sdf_path)
        
        if template is None or prepared_mol is None:
            return None
        
        # Assign bond orders from template
        prepared_mol = _assign_template_bonds(template, prepared_mol)
        query = _heavy_mol(template) if template is not None else None
        
        # If reference PDB is provided, calculate RMSD against it
        if reference_pdb and os.path.exists(reference_pdb):
            ref_mol = Chem.MolFromPDBFile(reference_pdb, sanitize=False, removeHs=False, proximityBonding=True)
            if ref_mol:
                ref_mol = _assign_template_bonds(template, ref_mol)
                return _rmsd_against_reference(ref_mol, prepared_mol, query)
        
        # Otherwise, just return the heavy atom count for validation
        return {"heavy_atoms": template.GetNumHeavyAtoms(), "conformer_energy": None}
    
    except Exception as e:
        return None

def display_rmsd_validation_ui():
    """
    Display a standalone RMSD validation UI for ligand preparation.
    This function is called within the Ligand Preparation tab.
    """
    st.markdown("---")
    st.markdown("### 📐 RMSD Validation Tool for Ligand Preparation")
    st.markdown("""
    > **Purpose:** Validate your ligand preparation by comparing the generated 3D structure
    > with an experimental reference (if available) or by checking structural integrity.
    > This helps ensure that your ligands are correctly prepared before batch docking.
    """)
    
    # Two-column layout for validation options
    col_val1, col_val2 = st.columns(2)
    
    with col_val1:
        st.markdown("#### 🔬 Quick Validation")
        st.markdown("""
        - Checks if SMILES can be converted to 3D structure
        - Verifies atom count and connectivity
        - No reference structure required
        """)
        
        if st.button("🔍 Validate Current Ligands", key="validate_ligands_btn", use_container_width=True):
            smiles_list = st.session_state.get("selected_smiles_list", [])
            if not smiles_list:
                st.warning("No ligands selected for validation.")
            else:
                results = []
                for name, smiles in smiles_list[:10]:  # Limit to first 10 for performance
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            heavy_atoms = mol.GetNumHeavyAtoms()
                            # Try to generate 3D structure
                            mol_3d = build_3d_mol(smiles)
                            results.append({
                                "Name": name,
                                "SMILES": smiles[:50] + "..." if len(smiles) > 50 else smiles,
                                "Heavy Atoms": heavy_atoms,
                                "Valid": "✅",
                                "3D Generated": "✅" if mol_3d else "❌"
                            })
                        else:
                            results.append({
                                "Name": name,
                                "SMILES": smiles[:50] + "..." if len(smiles) > 50 else smiles,
                                "Heavy Atoms": "N/A",
                                "Valid": "❌",
                                "3D Generated": "❌"
                            })
                    except Exception as e:
                        results.append({
                            "Name": name,
                            "SMILES": smiles[:50] + "..." if len(smiles) > 50 else smiles,
                            "Heavy Atoms": "N/A",
                            "Valid": "❌",
                            "3D Generated": f"Error: {str(e)[:30]}"
                        })
                
                st.dataframe(pd.DataFrame(results), use_container_width=True)
                
                valid_count = sum(1 for r in results if r["Valid"] == "✅")
                st.success(f"✅ {valid_count}/{len(results)} ligands passed validation")
                
                if valid_count < len(results):
                    st.warning("Some ligands failed validation. Check SMILES strings for errors.")
    
    with col_val2:
        st.markdown("#### 📊 RMSD Calculator (Experimental)")
        st.markdown("""
        For redocking validation, use the **Redocking Validation** section above.
        This tool is for comparing prepared ligands against reference structures.
        """)
        
        # Upload reference structure for RMSD calculation
        ref_file = st.file_uploader(
            "Upload reference ligand structure (PDB or SDF)",
            type=["pdb", "sdf"],
            key="rmsd_ref_upload",
            help="Optional: Upload experimental reference to calculate RMSD"
        )
        
        if ref_file:
            ref_path = os.path.join(OUTPUT_DIR, f"rmsd_ref_{timestamp_tag()}.{ref_file.name.split('.')[-1]}")
            with open(ref_path, "wb") as f:
                f.write(ref_file.getvalue())
            
            st.success(f"✅ Reference uploaded: {ref_file.name}")
            
            # Select ligand to compare
            smiles_list = st.session_state.get("selected_smiles_list", [])
            if smiles_list:
                lig_options = [f"{name}" for name, _ in smiles_list[:20]]
                selected_lig = st.selectbox("Select ligand to compare", lig_options, key="rmsd_lig_select")
                
                if selected_lig and st.button("Calculate RMSD", key="calc_rmsd_btn"):
                    # Find the SMILES for selected ligand
                    lig_smiles = next((smi for name, smi in smiles_list if name == selected_lig), None)
                    if lig_smiles:
                        # Prepare ligand temporarily
                        temp_dir = unique_workdir(Path(OUTPUT_DIR) / "rmsd_temp")
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            prep_result = prepare_ligand_vina_batch(lig_smiles, "temp", 7.4, temp_dir)
                            if prep_result['success']:
                                rmsd_val = calculate_rmsd_between_ligands(
                                    ref_path, prep_result['sdf'], "LIG", lig_smiles
                                )
                                if rmsd_val is not None:
                                    message, status = get_rmsd_validation_message(rmsd_val)
                                    st.markdown(f"""
                                    <div class="rmsd-card">
                                        <strong>RMSD Calculation Result:</strong><br>
                                        <span class="rmsd-value">{rmsd_val:.2f} Å</span><br>
                                        {message}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.warning("Could not calculate RMSD. Check atom matching.")
                            else:
                                st.error("Failed to prepare ligand for comparison.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                        finally:
                            # Cleanup temp directory
                            shutil.rmtree(temp_dir, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
#  DOCKING EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def parse_vina_config_box(config_path):
    want = {k: None for k in ("center_x","center_y","center_z","size_x","size_y","size_z")}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for k in want:
                if line.lower().startswith(k):
                    parts = re.split(r"[=\s]+", line, maxsplit=1)
                    if len(parts) >= 2:
                        try:
                            want[k] = float(parts[1])
                        except ValueError:
                            pass
    missing = [k for k, v in want.items() if v is None]
    if missing:
        raise RuntimeError(f"Could not parse {missing} from config: {config_path}")
    return (want["center_x"], want["center_y"], want["center_z"], want["size_x"], want["size_y"], want["size_z"])

def convert_pdbqt_to_sdf(pdbqt_file, sdf_file=None):
    sdf_file = sdf_file or pdbqt_file.replace(".pdbqt", ".sdf")
    r = subprocess.run(["obabel", pdbqt_file, "-O", sdf_file], capture_output=True, text=True, timeout=60)
    if not os.path.exists(sdf_file) or os.path.getsize(sdf_file) < 10:
        raise RuntimeError(f"SDF conversion failed: {sdf_file}")
    return sdf_file

def _find_unidock_outputs(out_dir):
    if not os.path.isdir(out_dir):
        return {"pdbqt": None, "sdf": None, "all": []}
    all_files = sorted([p for p in glob.glob(os.path.join(out_dir,"*")) if os.path.isfile(p) and os.path.getsize(p)>0])
    sdf_hits = sorted(glob.glob(os.path.join(out_dir,"*_out*.sdf"))) + sorted(glob.glob(os.path.join(out_dir,"*.sdf")))
    pdbqt_hits = sorted(glob.glob(os.path.join(out_dir,"*_out*.pdbqt"))) + sorted(glob.glob(os.path.join(out_dir,"*.pdbqt")))
    def _uniq(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen and os.path.isfile(x) and os.path.getsize(x)>10:
                seen.add(x); out.append(x)
        return out
    sdf_hits = _uniq(sdf_hits); pdbqt_hits = _uniq(pdbqt_hits)
    return {"sdf": sdf_hits[0] if sdf_hits else None, "pdbqt": pdbqt_hits[0] if pdbqt_hits else None, "all": all_files}

def _update_live_log(log_placeholder, lines, header=""):
    if log_placeholder is None:
        return
    tail = "".join(lines[-80:]).strip()
    body = (header + "\n\n" if header else "") + (tail or "Running...")
    log_placeholder.code(body[-12000:], language="text")

def run_subprocess_docking(cmd_list, timeout=1800, log_placeholder=None):
    header = "$ " + " ".join(str(x) for x in cmd_list)
    if log_placeholder is None:
        try:
            r = subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return -1, f"Timed out after {timeout} seconds"
        except Exception as e:
            return -1, str(e)

    lines = []
    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        _update_live_log(log_placeholder, lines, header)
        while True:
            if proc.poll() is not None:
                rest = proc.stdout.read() if proc.stdout else ""
                if rest:
                    lines.append(rest)
                break
            if time.time() - start > timeout:
                proc.kill()
                lines.append(f"\nTimed out after {timeout} seconds\n")
                _update_live_log(log_placeholder, lines, header)
                return -1, "".join(lines)
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                lines.append(line)
                _update_live_log(log_placeholder, lines, header)
            else:
                time.sleep(0.2)
        _update_live_log(log_placeholder, lines, header)
        return proc.returncode, "".join(lines)
    except subprocess.TimeoutExpired:
        return -1, f"Timed out after {timeout} seconds"
    except Exception as e:
        return -1, str(e)

def dock_one_ligand(engine, dock_bin, receptor_pdbqt, ligand_input, config_file, out_prefix, params):
    log = ""
    timeout = int(params.get("timeout", 1800))
    log_placeholder = params.get("log_placeholder")
    
    if engine == "GNINA":
        cx, cy, cz, sx, sy, sz = parse_vina_config_box(config_file)
        out_sdf = f"{out_prefix}.sdf"
        gnina_args = [
            "-r", receptor_pdbqt, "-l", ligand_input,
            "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
            "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
            "--exhaustiveness", str(params.get("gnina_exhaustiveness", 8)),
            "--num_modes", str(params.get("gnina_num_modes", 9)),
            "--cnn_scoring", params.get("gnina_cnn_scoring", "rescore"),
            "-o", out_sdf
        ]
        if _is_docker_gnina(dock_bin):
            image = dock_bin.split(":", 1)[1] or "gnina/gnina"
            docker_args = [
                "-r", _docker_work_path(receptor_pdbqt), "-l", _docker_work_path(ligand_input),
                "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
                "--exhaustiveness", str(params.get("gnina_exhaustiveness", 8)),
                "--num_modes", str(params.get("gnina_num_modes", 9)),
                "--cnn_scoring", params.get("gnina_cnn_scoring", "rescore"),
                "-o", _docker_work_path(out_sdf)
            ]
            cmd = ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{os.getcwd()}:/work", "-w", "/work", image, "gnina"] + docker_args
        else:
            cmd = [dock_bin] + gnina_args
        rc, log = run_subprocess_docking(cmd, timeout=timeout, log_placeholder=log_placeholder)
        if rc != 0 or not os.path.exists(out_sdf) or os.path.getsize(out_sdf) < 10:
            raise RuntimeError(f"GNINA failed (exit {rc}): {log[:500]}")
        return None, out_sdf, log
    
    if engine == "UNIDOCK":
        cx, cy, cz, sx, sy, sz = parse_vina_config_box(config_file)
        out_dir = out_prefix + "_unidock_out"
        os.makedirs(out_dir, exist_ok=True)
        search_mode = params.get("unidock_search_mode", "balance")
        cmd = [
            dock_bin, "--receptor", receptor_pdbqt, "--gpu_batch", ligand_input,
            "--scoring", params.get("unidock_scoring", "vina"),
            "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
            "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
            "--num_modes", str(params.get("unidock_num_modes", 9)),
            "--dir", out_dir,
            "--seed", str(params.get("unidock_seed", 0))
        ]
        if search_mode != "custom":
            cmd += ["--search_mode", search_mode]
        else:
            cmd += [
                "--exhaustiveness", str(params.get("unidock_exhaustiveness", 512)),
                "--max_step", str(params.get("unidock_max_step", 40))
            ]
        rc, log = run_subprocess_docking(cmd, timeout=timeout, log_placeholder=log_placeholder)
        hits = _find_unidock_outputs(out_dir)
        docked_sdf, docked_pdbqt = hits["sdf"], hits["pdbqt"]
        if docked_sdf is None and docked_pdbqt is not None:
            docked_sdf = convert_pdbqt_to_sdf(docked_pdbqt)
        if docked_sdf is None and docked_pdbqt is None:
            raise RuntimeError(f"Uni-Dock produced no output in {out_dir}: {log[:500]}")
        return docked_pdbqt, docked_sdf, log
    
    # VINA / VINAXB
    out_pdbqt = f"{out_prefix}.pdbqt"
    cmd = [
        dock_bin, "--receptor", receptor_pdbqt, "--ligand", ligand_input,
        "--config", config_file,
        "--exhaustiveness", str(params.get("exhaustiveness", 16)),
        "--num_modes", str(params.get("num_modes", 9)),
        "--energy_range", str(params.get("energy_range", 3)),
        "--out", out_pdbqt
    ]
    rc, log = run_subprocess_docking(cmd, timeout=timeout, log_placeholder=log_placeholder)
    if not os.path.exists(out_pdbqt) or os.path.getsize(out_pdbqt) < 10:
        raise RuntimeError(f"{engine} failed or produced no output (exit {rc}): {log[:500]}")
    out_sdf = convert_pdbqt_to_sdf(out_pdbqt)
    return out_pdbqt, out_sdf, log

# ═══════════════════════════════════════════════════════════════════════════
#  SCORE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_top_score(pdbqt_path, sdf_path, engine):
    engine = engine.upper()
    if engine == "GNINA":
        if sdf_path and os.path.exists(sdf_path):
            supp = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
            mol = next((m for m in supp if m is not None), None)
            if mol:
                for prop in ("minimizedAffinity","affinity","docking_score"):
                    if mol.HasProp(prop):
                        return {
                            "score": float(mol.GetProp(prop)),
                            "CNNscore": float(mol.GetProp("CNNscore")) if mol.HasProp("CNNscore") else None,
                            "CNNaffinity": float(mol.GetProp("CNNaffinity")) if mol.HasProp("CNNaffinity") else None
                        }
        raise ValueError("No GNINA affinity property found")
    
    if engine == "UNIDOCK":
        if sdf_path and os.path.exists(sdf_path):
            with open(sdf_path) as f:
                for line in f:
                    if "ENERGY=" in line:
                        nums = re.findall(r"[-+]?\d*\.?\d+", line)
                        if nums:
                            return {
                                "score": float(nums[0]),
                                "rmsd_lb": float(nums[1]) if len(nums)>=2 else "",
                                "rmsd_ub": float(nums[2]) if len(nums)>=3 else ""
                            }
        raise ValueError("No UNIDOCK ENERGY= line found")
    
    if pdbqt_path and os.path.exists(pdbqt_path):
        with open(pdbqt_path) as f:
            for line in f:
                if re.search(r"VINA(?:XB)? RESULT", line):
                    parts = line.split()
                    try:
                        return {
                            "score": float(parts[3]),
                            "rmsd_lb": float(parts[4]),
                            "rmsd_ub": float(parts[5])
                        }
                    except:
                        pass
    raise ValueError(f"No VINA RESULT line found in {pdbqt_path}")

def parse_all_poses(pdbqt_path=None, sdf_path=None, engine="VINA"):
    engine = engine.upper()
    
    if engine == "GNINA" and sdf_path and os.path.exists(sdf_path):
        supp = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
        poses = []
        for i, mol in enumerate(supp):
            if mol is None: continue
            aff = None
            for prop in ("minimizedAffinity","affinity"):
                if mol.HasProp(prop):
                    try: aff = float(mol.GetProp(prop)); break
                    except: pass
            poses.append({
                "pose": i+1,
                "affinity": aff,
                "CNNscore": float(mol.GetProp("CNNscore")) if mol.HasProp("CNNscore") else None,
                "CNNaffinity": float(mol.GetProp("CNNaffinity")) if mol.HasProp("CNNaffinity") else None
            })
        return poses
    
    if engine == "UNIDOCK" and sdf_path and os.path.exists(sdf_path):
        with open(sdf_path) as f:
            content = f.read()
        blocks = content.split("$$$$")
        poses = []
        for i, block in enumerate(blocks):
            if not block.strip(): continue
            nums = []
            for line in block.splitlines():
                if "ENERGY=" in line:
                    nums = re.findall(r"[-+]?\d*\.?\d+", line)
                    break
            aff = float(nums[0]) if nums else None
            poses.append({
                "pose": i+1,
                "affinity": aff,
                "rmsd_lb": float(nums[1]) if len(nums)>=2 else None,
                "rmsd_ub": float(nums[2]) if len(nums)>=3 else None
            })
        return poses
    
    poses = []
    if pdbqt_path and os.path.exists(pdbqt_path):
        current_mode = None
        with open(pdbqt_path) as f:
            for line in f:
                if line.startswith("MODEL"):
                    try: current_mode = int(line.split()[1])
                    except: pass
                elif re.search(r"VINA(?:XB)? RESULT", line):
                    parts = line.split()
                    try:
                        poses.append({
                            "pose": current_mode,
                            "affinity": float(parts[3]),
                            "rmsd_lb": float(parts[4]),
                            "rmsd_ub": float(parts[5])
                        })
                    except: pass
    return poses

# ═══════════════════════════════════════════════════════════════════════════
#  LINKED FILE CREATION
# ═══════════════════════════════════════════════════════════════════════════

def create_docking_session_files(workdir, session_name, receptor_result, ligand_results, dock_records, engine, redock_result=None):
    """Create a well-structured ZIP with subfolders + summary CSV + session report."""
    workdir = Path(workdir)
    ts = timestamp_tag()

    # ── Build summary CSV (all ligands, sorted best→worst) ──────────
    summary_rows = []
    for d in dock_records:
        summary_rows.append({
            "Compound": d["name"],
            "Best_Affinity_kcal_mol": round(d["top_score"], 2) if d["top_score"] is not None else "",
            "Status": "success" if d["top_score"] is not None else "failed",
        })
    summary_rows.sort(key=lambda r: (r["Status"] != "success", float(r["Best_Affinity_kcal_mol"]) if r["Best_Affinity_kcal_mol"] != "" else 999))
    summary_csv_path = str(workdir / "docking_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    # ── Build human-readable session report ─────────────────────────
    report_path = str(workdir / "session_report.txt")
    with open(report_path, "w") as f:
        f.write(f"IMPPATez Docking Session Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Engine    : {engine}\n")
        f.write(f"Timestamp : {ts}\n\n")
        if redock_result and redock_result.get("success"):
            rmsd_v = redock_result.get("rmsd")
            f.write(f"Redocking Validation (RMSD, Pose 1)\n")
            f.write(f"  Ligand   : {redock_result.get('original_resname','')}\n")
            f.write(f"  RMSD     : {rmsd_v:.2f} Å\n" if rmsd_v is not None else "  RMSD     : N/A\n")
            f.write("\n")
        f.write(f"Docking Results (sorted by affinity)\n")
        f.write(f"{'-'*40}\n")
        for row in summary_rows:
            score_str = f"{row['Best_Affinity_kcal_mol']} kcal/mol" if row["Best_Affinity_kcal_mol"] != "" else "failed"
            f.write(f"  {row['Compound']:<35} {score_str}\n")

    # ── Assemble ZIP with organised subfolders ───────────────────────
    # Structure:
    #   docking_results_<ts>.zip
    #   ├── docking_summary.csv
    #   ├── session_report.txt
    #   ├── receptor/
    #   │   ├── rec.pdb
    #   │   ├── rec.pdbqt
    #   │   └── rec.box.txt
    #   ├── docked_poses/
    #   │   ├── <ligand>_docked.sdf
    #   │   └── <ligand>_docked.pdbqt
    #   ├── ligands/
    #   │   ├── <ligand>.sdf
    #   │   └── <ligand>.pdbqt
    #   └── redocking/          (only if redocking was performed)
    #       ├── original_<resname>.pdb
    #       └── redocked.sdf

    zip_path = str(workdir / f"docking_results_{ts}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Top-level summary files
        zf.write(summary_csv_path, arcname="docking_summary.csv")
        zf.write(report_path,      arcname="session_report.txt")

        # receptor/
        for key, label in [
            ("receptor_pdb",   "rec.pdb"),
            ("receptor_pdbqt", "rec.pdbqt"),
            ("config_file",    "rec.box.txt"),
            ("box_pdb",        "rec.box.pdb"),
        ]:
            fp = receptor_result.get(key)
            if fp and os.path.exists(fp):
                zf.write(fp, arcname=f"receptor/{label}")

        # ligands/  (prepared input files)
        for lr in ligand_results:
            safe = safe_name(lr["name"])
            for key, ext in [("sdf", ".sdf"), ("pdbqt", ".pdbqt"), ("pdb", ".pdb")]:
                fp = lr.get(key)
                if fp and os.path.exists(fp):
                    zf.write(fp, arcname=f"ligands/{safe}{ext}")

        # docked_poses/
        for d in dock_records:
            if d.get("top_score") is None:
                continue
            safe = safe_name(d["name"])
            if d.get("docked_sdf") and os.path.exists(d["docked_sdf"]):
                zf.write(d["docked_sdf"],   arcname=f"docked_poses/{safe}_docked.sdf")
            if d.get("docked_pdbqt") and os.path.exists(d["docked_pdbqt"]):
                zf.write(d["docked_pdbqt"], arcname=f"docked_poses/{safe}_docked.pdbqt")

        # redocking/
        if redock_result and redock_result.get("success"):
            for key, label in [
                ("original_ligand_pdb", f"original_{redock_result.get('original_resname','LIG')}.pdb"),
                ("docked_sdf",          "redocked.sdf"),
                ("docked_pdbqt",        "redocked.pdbqt"),
            ]:
                fp = redock_result.get(key)
                if fp and os.path.exists(fp):
                    zf.write(fp, arcname=f"redocking/{label}")

    session_json = str(workdir / f"session_{ts}.json")
    session_data = {
        "session_name": session_name, "engine": engine, "timestamp": ts,
        "receptor": {
            "pdb": receptor_result.get("receptor_pdb"),
            "pdbqt": receptor_result.get("receptor_pdbqt"),
            "config": receptor_result.get("config_file"),
        },
        "docked_poses": [
            {"name": d["name"], "top_score": d.get("top_score"), "docked_sdf": d.get("docked_sdf")}
            for d in dock_records
        ],
    }
    with open(session_json, "w") as f:
        _json.dump(session_data, f, indent=2)

    return {"session_json": session_json, "links_txt": report_path, "zip": zip_path}

# ═══════════════════════════════════════════════════════════════════════════
#  IMPPAT SCRAPER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_page(plant):
    url = f"{BASE}/phytochemical/{urllib.parse.quote(plant)}"
    r = get_session().get(url, timeout=25)
    r.raise_for_status()
    return r.text

def fetch_detail_page(imphy_id):
    url = f"{BASE}/phytochemical-detailedpage/{imphy_id}"
    r = get_session().get(url, timeout=25)
    r.raise_for_status()
    return r.text

def extract_entries(html):
    entries = []
    try:
        for table in pd.read_html(html):
            cols = [str(c).strip() for c in table.columns]
            if "IMPPAT Phytochemical identifier" in cols and "Phytochemical name" in cols:
                sub = table[["IMPPAT Phytochemical identifier", "Phytochemical name"]].copy()
                sub.columns = ["IMPHY_ID", "Phytochemical_Name"]
                for _, row in sub.iterrows():
                    imphy_id = str(row["IMPHY_ID"]).strip()
                    name = str(row["Phytochemical_Name"]).strip()
                    if re.fullmatch(r"IMPHY\d+", imphy_id):
                        entries.append({"IMPHY_ID": imphy_id, "Phytochemical_Name": name})
                break
    except Exception:
        pass
    if entries:
        seen, deduped = set(), []
        for item in entries:
            if item["IMPHY_ID"] not in seen:
                seen.add(item["IMPHY_ID"]); deduped.append(item)
        return deduped
    pairs = re.findall(r"(IMPHY\d+)\s*<\/td>\s*<td[^>]*>\s*([^<]+)", html, flags=re.IGNORECASE)
    seen = set()
    for imphy_id, name in pairs:
        if imphy_id not in seen:
            seen.add(imphy_id)
            entries.append({"IMPHY_ID": imphy_id.strip(), "Phytochemical_Name": name.strip()})
    return entries

def extract_smiles(detail_html):
    text = re.sub(r"<[^>]+>", " ", detail_html)
    text = re.sub(r"\s+", " ", text).strip()
    for pattern in [r"SMILES:\s*(.*?)\s*InChI:", r"SMILES:\s*(.*?)\s*InChIKey:", r"SMILES:\s*(.*?)\s*Functional groups:"]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m: return m.group(1).strip()
    return ""

def extract_cid(detail_html):
    for pattern in [r'https?://pubchem\.ncbi\.nlm\.nih\.gov/compound/(\d+)', r'https?://pubchem\.ncbi\.nlm\.nih\.gov/summary/summary\.cgi\?cid=(\d+)']:
        matches = re.findall(pattern, detail_html, re.IGNORECASE)
        if matches: return matches[0]
    text = re.sub(r"<[^>]+>", " ", detail_html)
    text = re.sub(r"\s+", " ", text).strip()
    for pattern in [r"PubChem\s+CID[:\s]+(\d+)", r"CID[:\s]+(\d+)"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m: return m.group(1)
    return ""

def ro5_from_smiles(smiles):
    if not smiles:
        return "No SMILES"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return "Invalid SMILES"
        violations = sum([
            Descriptors.MolWt(mol)>500,
            Descriptors.MolLogP(mol)>5,
            Lipinski.NumHDonors(mol)>5,
            Lipinski.NumHAcceptors(mol)>10
        ])
        return "Passed" if violations <= 1 else f"Failed ({violations} violations)"
    except: return "RDKit Error"

def process_single_entry(item, max_retries=2):
    for attempt in range(max_retries+1):
        try:
            if attempt > 0: time.sleep(attempt)
            html = fetch_detail_page(item["IMPHY_ID"])
            smiles = extract_smiles(html)
            cid = extract_cid(html)
            return {
                "IMPHY_ID": item["IMPHY_ID"],
                "CID": cid,
                "Phytochemical_Name": item["Phytochemical_Name"],
                "SMILES_IMPPAT": smiles,
                "LIPINSKI": ro5_from_smiles(smiles) if smiles else "No SMILES"
            }
        except Exception:
            if attempt == max_retries:
                return {
                    "IMPHY_ID": item["IMPHY_ID"],
                    "CID": "",
                    "Phytochemical_Name": item["Phytochemical_Name"],
                    "SMILES_IMPPAT": "",
                    "LIPINSKI": "No SMILES"
                }
    return None

def get_bangla_name_from_wikipedia(plant_name):
    """
    Fetch Bangla common/vernacular name using a 3-step approach:
    1. Wikidata entity label in Bengali (bn) — most reliable
    2. English Wikipedia -> pageprops -> wikibase_item -> Wikidata bn label
    3. Bangla Wikipedia REST summary title as last resort
    """
    if not plant_name:
        return ""
    try:
        formatted = plant_name.strip().replace(" ", "_")

        # ── Step 1: Get Wikidata entity ID from English Wikipedia ────────
        wp_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&titles={urllib.parse.quote(formatted)}"
            "&prop=pageprops&ppprop=wikibase_item&format=json"
        )
        r = requests.get(wp_url, timeout=10, headers=HEADERS)
        qid = None
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                qid = page.get("pageprops", {}).get("wikibase_item")
                if qid:
                    break

        # ── Step 2: Fetch Bengali label from Wikidata ────────────────────
        if qid:
            wd_url = (
                f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
            )
            wd_r = requests.get(wd_url, timeout=10, headers=HEADERS)
            if wd_r.status_code == 200:
                entity = wd_r.json().get("entities", {}).get(qid, {})
                # Try Bengali label first
                bn_label = entity.get("labels", {}).get("bn", {}).get("value", "")
                if bn_label:
                    return bn_label
                # Try Bengali aliases
                bn_aliases = entity.get("aliases", {}).get("bn", [])
                if bn_aliases:
                    return bn_aliases[0].get("value", "")

        # ── Step 3: Bangla Wikipedia REST summary title ──────────────────
        bn_url = (
            f"https://bn.wikipedia.org/api/rest_v1/page/summary/"
            f"{urllib.parse.quote(formatted)}"
        )
        br = requests.get(bn_url, timeout=10, headers=HEADERS)
        if br.status_code == 200:
            data = br.json()
            # description contains common name, title is usually scientific
            desc = data.get("description", "")
            title = data.get("title", "")
            # prefer title if it contains Bangla Unicode characters
            if title and any("\u0980" <= c <= "\u09FF" for c in title):
                return title
            if desc and any("\u0980" <= c <= "\u09FF" for c in desc):
                return desc

    except Exception:
        pass
    return ""

def extract_plant_info(html):
    info = {}
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    # Stop token: each field ends where the next label (or System of medicine or More Information) begins
    _stop = r"(?=Kingdom|Family|Group|Common\s+[Nn]ame|Synonym|System\s+of\s+medicine|More\s+Information|Total|$)"
    for key, pat in [
        ("Kingdom",
         r"Kingdom\s*:?\s*([A-Za-z][A-Za-z ]*?)" + _stop),
        ("Family",
         r"Family\s*:?\s*([A-Za-z][A-Za-z ]*?)" + _stop),
        ("Group",
         r"Group\s*:?\s*([A-Za-z][A-Za-z ]*?)" + _stop),
        ("Common name",
         r"Common\s+[Nn]ames?\s*:?\s*(.+?)" + _stop),
        ("Synonymous names",
         r"Synonymous\s+[Nn]ames?\s*:?\s*(.+?)"
         r"(?=System\s+of\s+medicine|More\s+Information|Kingdom|Family|Group|Common\s+[Nn]ame|Total|$)"),
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip(".")
            if val:
                info[key] = val
    return info

def run_step1(plant_name):
    plant_name = str(plant_name or "").strip()
    if not plant_name: return None, None, "Please enter a plant name."
    try:
        html = fetch_page(plant_name)
    except Exception as exc:
        return None, None, f"Failed to fetch: {exc}"
    plant_info = extract_plant_info(html)
    entries = extract_entries(html)
    if not entries: return None, None, "No phytochemicals found."
    rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_entry, item): item for item in entries}
        for i, future in enumerate(as_completed(futures)):
            status_text.text(f"Processing compounds ({i+1}/{len(entries)})...")
            progress_bar.progress((i+1)/len(entries))
            try:
                r = future.result()
                if r: rows.append(r)
            except: pass
    progress_bar.empty(); status_text.empty()
    df = pd.DataFrame(rows)[["IMPHY_ID","CID","Phytochemical_Name","SMILES_IMPPAT","LIPINSKI"]]
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", plant_name).strip("_") or "plant"
    csv_path = os.path.join(OUTPUT_DIR, f"{safe}_phytochemicals_{timestamp_tag()}.csv")
    df.to_csv(csv_path, index=False)
    total = len(df); passed = int((df["LIPINSKI"]=="Passed").sum()); failed = total - passed
    return df, csv_path, plant_info, total, passed, failed

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — SDF DOWNLOADER
# ═══════════════════════════════════════════════════════════════════════════

def download_sdf(imphy_id, folder):
    url = f"{BASE}/images/3D/SDF/{imphy_id}_3D.sdf"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=25, headers=HEADERS)
            if r.status_code == 200 and len(r.text.strip()) > 50:
                path = os.path.join(folder, f"{imphy_id}.sdf")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                return path
        except: pass
        if attempt < 2: time.sleep(1+attempt)
    return None

def run_step2(csv_path, sdf_mode):
    if not csv_path or not os.path.exists(csv_path):
        return None, None, "No CSV file found. Run Step 1 first."
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        return None, None, f"Could not read CSV: {exc}"
    if "IMPHY_ID" not in df.columns:
        return None, None, "CSV must contain IMPHY_ID column."
    work_df = df.copy()
    if sdf_mode in ("RO5 Passed", "Lipinski Passed Only"):
        lipinski = work_df["LIPINSKI"] if "LIPINSKI" in work_df.columns else pd.Series("", index=work_df.index)
        work_df = work_df[lipinski.fillna("").str.strip()=="Passed"]
    folder = os.path.join(OUTPUT_DIR, f"sdf_files_{timestamp_tag()}")
    os.makedirs(folder, exist_ok=True)
    saved, missing = [], []
    ids = work_df["IMPHY_ID"].dropna().astype(str).str.strip().tolist()
    ids = [x for x in ids if x]
    progress_bar = st.progress(0); status_text = st.empty()
    for i, imphy in enumerate(ids):
        status_text.text(f"Downloading SDF ({i+1}/{len(ids)})...")
        progress_bar.progress((i+1)/len(ids))
        if re.fullmatch(r"IMPHY\d+", imphy):
            path = download_sdf(imphy, folder)
            if path: saved.append(path)
            else: missing.append(imphy)
        else: missing.append(imphy)
    progress_bar.empty(); status_text.empty()
    zip_name = None
    if saved:
        zip_name = os.path.join(OUTPUT_DIR, f"IMPPAT_3D_SDF_{timestamp_tag()}.zip")
        with zipfile.ZipFile(zip_name, "w") as z:
            for fp in saved: z.write(fp, arcname=os.path.basename(fp))
    return zip_name, missing, len(saved), len(missing)  # Changed None to missing

# ═══════════════════════════════════════════════════════════════════════════
#  SCORE CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_score_chart(dock_records, engine, ref_score=None, ref_label="Reference"):
    df = pd.DataFrame([{
        "name": d["name"],
        "score": d.get("top_score") or 0
    } for d in dock_records if d.get("top_score") is not None]).sort_values("score", ascending=True).reset_index(drop=True)
    if df.empty:
        return None
    width = max(7, 0.7*len(df)+2)
    fig, ax = plt.subplots(figsize=(width, 5.0))
    ax.scatter(range(len(df)), df["score"], s=80, color="steelblue", zorder=3, label="Docked compounds")
    for i, v in enumerate(df["score"]):
        ax.text(i, v+0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color="dimgray")
    handles = [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor="steelblue",markersize=8,label="Docked compounds")]
    if ref_score is not None:
        ax.axhline(ref_score, color="crimson", linewidth=1.5, linestyle="--", zorder=2)
        handles.append(plt.Line2D([0],[0],color="crimson",linewidth=1.5,linestyle="--",label=f"{ref_label} ({ref_score:.1f} kcal/mol)"))
    ax.legend(handles=handles, fontsize=9, framealpha=0.9, loc="lower left", bbox_to_anchor=(0,1.12), ncol=len(handles), borderaxespad=0)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["name"], rotation=55, ha="right", fontsize=9)
    ax.set_xlabel("Compound", fontsize=11)
    ax.set_ylabel("Binding energy (kcal/mol)", fontsize=11)
    ax.set_title(f"Top-pose {engine} score — best pose per compound", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.subplots_adjust(top=0.82)
    return fig

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN UI - TABS (REORGANIZED)
# ═══════════════════════════════════════════════════════════════════════════

# Session state init
for key, default in [
    ("csv_path", None), ("df", None),
    ("receptor_result", None), ("receptor_pdb_path", None), ("cocrystal_info", None),
    ("dock_records", []), ("ligand_results", []), ("redock_result", None),
    ("session_files", None), ("dock_bin_path", None), ("current_engine", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌿 Phytochemical Extraction",
    "⚙️ Docking Engine",
    "🔬 Receptor Preparation",
    "🔬 Ligand Preparation",
    "⚗️ Batch Docking (Results & Download)",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — IMPPAT Phytochemical Extraction (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">Section 1.1</div>
        <div class="step-heading">🍀 Phytochemical Library </div>
    </div>
    """, unsafe_allow_html=True)

    # Form enables Enter key submission
    with st.form("imppat_form"):

        col1, col2 = st.columns([5, 1])

        with col1:
            plant_name = st.text_input(
                "Scientific Name:",
                placeholder="e.g., Azadirachta indica",
                key="plant_input",
                label_visibility="visible"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Align button vertically
            search_btn = st.form_submit_button(
                "🔍 Search IMPPAT",
                use_container_width=True
            )

    if search_btn and plant_name:
        with st.spinner(f"Searching IMPPAT for {plant_name}..."):
            result = run_step1(plant_name)
        if result[0] is not None:
            df, csv_path, plant_info, total, passed, failed = result
            # Save everything needed for re-runs to session_state
            st.session_state.df = df
            st.session_state.csv_path = csv_path
            st.session_state["phyto_total"] = total
            st.session_state["phyto_passed"] = passed
            st.session_state["phyto_failed"] = failed
            st.session_state["phyto_plant_info"] = plant_info
            st.session_state["phyto_plant_name"] = plant_name
            st.session_state["phyto_bangla"] = get_bangla_name_from_wikipedia(plant_name)
            st.session_state["phyto_chart_filter"] = None  # reset chart on new search
        else:
            st.error(result[2] if len(result)>2 else "No phytochemicals found.")

    # ── Display results (runs on every re-run if data exists) ────────────
    if st.session_state.get("df") is not None and st.session_state.get("phyto_total") is not None:
        df       = st.session_state.df
        csv_path = st.session_state.csv_path
        total    = st.session_state["phyto_total"]
        passed   = st.session_state["phyto_passed"]
        failed   = st.session_state["phyto_failed"]
        plant_info  = st.session_state["phyto_plant_info"]
        _plant_name = st.session_state["phyto_plant_name"]
        _bangla     = st.session_state.get("phyto_bangla", "")

        _parts    = _plant_name.strip().split()
        _fmt_name = (_parts[0].capitalize() + " " + " ".join(p.lower() for p in _parts[1:])) if len(_parts) >= 2 else _plant_name.strip().capitalize()
        _kingdom  = plant_info.get("Kingdom", "") or "—"
        _family   = plant_info.get("Family", "") or "—"
        _group    = plant_info.get("Group", "") or "—"
        _common   = plant_info.get("Common name", "") or "—"
        _synonyms = plant_info.get("Synonymous names", "") or "—"
        _bangla_d = _bangla or "—"

        _card_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:transparent; font-family:'IBM Plex Sans',sans-serif; padding:2px 0; }}
.card {{
  background:linear-gradient(135deg,#f0fbf0 0%,#e4f5e5 100%);
  border:1.5px solid #a5d6a7; border-radius:16px;
  padding:18px 22px 16px 22px; box-shadow:0 4px 20px rgba(46,125,50,0.12);
}}
.header {{ display:flex; align-items:center; gap:12px; margin-bottom:14px; }}
.icon {{ background:linear-gradient(135deg,#2e7d32,#43a047); border-radius:10px; padding:8px 12px; font-size:1.5rem; line-height:1; }}
.plant-name {{ font-size:1.15rem; font-weight:800; color:#1a3d20; font-style:italic; }}
.sub-label {{ font-size:0.74rem; color:#4a7a52; font-weight:500; margin-top:2px; }}
.info-table {{ background:rgba(255,255,255,0.6); border-radius:10px; padding:6px 14px; }}
.row {{ display:flex; align-items:baseline; gap:10px; padding:5px 0; border-bottom:1px solid rgba(46,125,50,0.1); }}
.row:last-child {{ border-bottom:none; }}
.key {{ font-size:0.75rem; color:#4a7a52; font-weight:700; min-width:155px; flex-shrink:0; }}
.val {{ font-size:0.86rem; color:#1f3d25; }}
</style></head><body>
<div class="card">
  <div class="header">
    <div class="icon">🌿</div>
    <div>
      <div class="plant-name">{_fmt_name}</div>
      <div class="sub-label">IMPPATez Phytochemical Profile</div>
    </div>
  </div>
  <div class="info-table">
    <div class="row"><span class="key">✰ Kingdom</span><span class="val">{_kingdom}</span></div>
    <div class="row"><span class="key">✰ Family</span><span class="val">{_family}</span></div>
    <div class="row"><span class="key">✰ Group</span><span class="val">{_group}</span></div>
    <div class="row"><span class="key">✰ Common name</span><span class="val">{_common}</span></div>
    <div class="row">
        <span class="key">✰ Synonymous names</span>
        <span class="val" style="font-style: italic;">{_synonyms}</span>
    </div>
    <div class="row"><span class="key">✰ বাংলা নাম</span><span class="val">{_bangla_d}</span></div>
  </div>
</div>
</body></html>"""
        components.html(_card_html, height=300, scrolling=False)

        # ── Clickable stat cards ──────────────────────────────────────────
        _sc1, _sc2, _sc3 = st.columns(3)
        with _sc1:
            st.markdown(f"""<div style="background:#e8f5e9;color:#1b5e20;border-radius:12px;
                padding:14px;text-align:center;margin-bottom:6px;">
                <div style="font-size:1.6rem;font-weight:800;line-height:1.1;">{total}</div>
                <div style="font-size:0.70rem;font-weight:600;text-transform:uppercase;margin-top:4px;">Total Phytochemicals</div>
            </div>""", unsafe_allow_html=True)
            if st.button("🌿 View Table", key="btn_chart_all", use_container_width=True):
                st.session_state["phyto_chart_filter"] = "all"
        with _sc2:
            st.markdown(f"""<div style="background:#d4edda;color:#155724;border-radius:12px;
                padding:14px;text-align:center;margin-bottom:6px;">
                <div style="font-size:1.6rem;font-weight:800;line-height:1.1;">{passed}</div>
                <div style="font-size:0.70rem;font-weight:600;text-transform:uppercase;margin-top:4px;">RO5 Passed</div>
            </div>""", unsafe_allow_html=True)
            if st.button("✅ View Table", key="btn_chart_passed", use_container_width=True):
                st.session_state["phyto_chart_filter"] = "passed"
        with _sc3:
            st.markdown(f"""<div style="background:#fdecea;color:#7f1d1d;border-radius:12px;
                padding:14px;text-align:center;margin-bottom:6px;">
                <div style="font-size:1.6rem;font-weight:800;line-height:1.1;">{failed}</div>
                <div style="font-size:0.70rem;font-weight:600;text-transform:uppercase;margin-top:4px;">RO5 Failed</div>
            </div>""", unsafe_allow_html=True)
            if st.button("❌ View Table", key="btn_chart_failed", use_container_width=True):
                st.session_state["phyto_chart_filter"] = "failed"

        # ── Filtered table based on which button was clicked ─────────────
        _chart_filter = st.session_state.get("phyto_chart_filter", None)
        if _chart_filter is not None:
            _plot_df = df.copy()
            if _chart_filter == "passed":
                _plot_df = _plot_df[_plot_df["LIPINSKI"] == "Passed"]
                _table_title = f"✅ RO5 Passed Compounds : {len(_plot_df)} compounds"
            elif _chart_filter == "failed":
                _plot_df = _plot_df[_plot_df["LIPINSKI"] != "Passed"]
                _table_title = f"❌ RO5 Failed Compounds : {len(_plot_df)} compounds"
            else:
                _table_title = f"🌿 All Phytochemicals : {len(_plot_df)} compounds"

            st.markdown(f"**{_table_title}**")
            if len(_plot_df) > 0:
                st.dataframe(_plot_df, use_container_width=True, height=350, hide_index=True)
            else:
                st.info("No compounds to display for this filter.")

        with open(csv_path, "rb") as f:
            st.download_button("📥  Download All Phytochemicals (.csv)", data=f,
                               file_name=os.path.basename(csv_path), mime="text/csv",
                               use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Section 1.2</div>
        <div class="step-heading">📂 3D SDF Downloader</div>
    </div>
    """, unsafe_allow_html=True)

    sdf_mode = st.radio("Download Filter", ["All Phytochemicals","RO5 Passed"], horizontal=True)
    if st.button("📦 Extract 3D SDF", use_container_width=True):
        if st.session_state.csv_path:
            with st.spinner("Downloading SDF files from IMPPAT..."):
                zip_path, missing_ids, saved_count, missing_count = run_step2(st.session_state.csv_path, sdf_mode)
            st.session_state["sdf_zip_path"]      = zip_path
            st.session_state["sdf_missing_ids"]   = missing_ids or []
            st.session_state["sdf_saved_count"]   = saved_count
            st.session_state["sdf_missing_count"] = missing_count
        else:
            st.warning("Run Step 1 first.")

    # ── Persistent results (survive re-runs) ─────────────────────────────
    if st.session_state.get("sdf_zip_path"):
        _zip      = st.session_state["sdf_zip_path"]
        _saved    = st.session_state["sdf_saved_count"]
        _miss_ids = st.session_state["sdf_missing_ids"]
        _miss_n   = st.session_state["sdf_missing_count"]
        
        # Green success box for downloaded files
        if _saved > 0:
            st.markdown(
                f'<div style="background:#d4edda;border:1px solid #c3e6cb;border-radius:8px;'
                f'padding:8px 14px;font-size:0.88rem;color:#155724;margin-bottom:12px;">'
                f'✅ &nbsp;<b>Successfully extracted {_saved} SDF files</b>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # Red warning box for missing files
        if _miss_n > 0 and _miss_ids:
            _links_html = " &nbsp;|&nbsp; ".join(
                f'<a href="https://cb.imsc.res.in/imppat/phytochemical-detailedpage/{mid}" '
                f'target="_blank" style="color:#c0392b;font-weight:600;text-decoration:underline;">'
                f'{mid}</a>'
                for mid in _miss_ids
            )
            st.markdown(
                f'<div style="background:#fdecea;border:1px solid #f5c6cb;border-radius:8px;'
                f'padding:8px 14px;font-size:0.88rem;color:#7f1d1d;margin-bottom:12px;">'
                f'⚠️ &nbsp;<b>Missing {_miss_n} SDF file(s):</b> &nbsp;{_links_html}'
                f'</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
        if os.path.exists(_zip):
            with open(_zip, "rb") as f:
                st.download_button("📥 Download ZIP", data=f, file_name=os.path.basename(_zip),
                                   mime="application/zip", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCKING ENGINE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Section 2.1</div>
        <div class="step-heading">⚙️ Docking Engine & Binary</div>
    </div>
    """, unsafe_allow_html=True)

    col_engine, col_binary = st.columns([1, 1])

    with col_engine:
        engine = st.selectbox(
            "Docking engine",
            ["VINA", "VINAXB", "GNINA", "UNIDOCK"],
            format_func=lambda x: {
                "VINA": "⚡ AutoDock Vina",
                "VINAXB": "🧲 VinaXB",
                "GNINA": "🧠 GNINA",
                "UNIDOCK": "🚀 Uni-Dock"
            }[x],
            key="main_engine_select"
        )

        st.session_state.current_engine = engine

    with col_binary:
        custom_bin = st.text_input(
            "Custom binary path",
            key="custom_bin_input",
            help="Example: /usr/local/bin/vina or docker:gnina/gnina"
        )

    if st.button("🔍 Find Binary", key="find_binary_btn", use_container_width=True):

        with st.spinner(f"Looking for {engine} binary..."):

            dock_bin, source = find_docking_binary(engine, custom_bin)

            if dock_bin:

                st.session_state.dock_bin_path = dock_bin

                if engine == "GNINA" and _is_docker_gnina(dock_bin):
                    st.success("✅ GNINA will run through Docker image `gnina/gnina`")
                    st.caption("First run can take a while because Docker may need to download the image.")
                else:
                    st.success(f"✅ Found {engine} at: {dock_bin}")
                    if source == "custom":
                        st.caption("✓ Using custom binary path")
                    else:
                        st.caption("✓ Using auto-detected binary")

            else:

                st.session_state.dock_bin_path = None
                st.error(f"❌ {engine} binary not found!")

    elif engine == "GNINA":

        ok, msg = docker_is_ready()

        if ok:
            st.info("🍎 macOS GNINA option: click **Find Binary** to use Docker image `gnina/gnina`.")
        else:
            st.warning(f"GNINA on macOS needs Docker or a Linux GNINA binary. Docker status: {msg}")


    st.markdown("""
    <div class="step-card">
        <div class="step-title">Section 2.2</div>
        <div class="step-heading">🎛️ Docking Paramters</div>
    </div>
    """, unsafe_allow_html=True)
    # Engine-specific parameters
    if engine in ("VINA","VINAXB"):
        exhaustiveness = st.slider("Exhaustiveness", 4, 128, 32, 2, key="exhaust")
        num_modes = st.slider("Number of binding modes", 1, 40, 20, 1, key="modes")
        energy_range = st.slider("Energy range (kcal/mol)", 1, 10, 5, 1, key="energy")
        dock_params = {"exhaustiveness": exhaustiveness, "num_modes": num_modes, "energy_range": energy_range}
    elif engine == "GNINA":
        gnina_preset = st.selectbox(
            "GNINA protocol",
            ["Quick test", "Balanced", "CNN rescore"],
            help="Quick test is best on Mac/Docker for checking the pipeline. Balanced/CNN rescore are slower.",
            key="gnina_preset"
        )
        default_exhaust, default_modes, default_cnn = {
            "Quick test": (4, 1, "none"),
            "Balanced": (16, 9, "none"),
            "CNN rescore": (16, 20, "rescore"),
        }[gnina_preset]
        gnina_exhaust = st.slider("Exhaustiveness", 1, 128, default_exhaust, 1, key="gnina_exhaust")
        gnina_modes = st.slider("Number of binding modes", 1, 40, default_modes, 1, key="gnina_modes")
        gnina_cnn = st.selectbox("CNN scoring mode", ["none", "rescore"], index=0 if default_cnn == "none" else 1, key="gnina_cnn")
        dock_params = {"gnina_exhaustiveness": gnina_exhaust, "gnina_num_modes": gnina_modes, "gnina_cnn_scoring": gnina_cnn}
    else:
        ud_search = st.selectbox("Search mode", ["fast", "balance", "detail", "custom"], key="ud_search")
        ud_modes = st.slider("Number of binding modes", 1, 40, 20, 1, key="ud_modes")
        ud_scoring = st.selectbox("Scoring function", ["vina", "vinardo"], key="ud_scoring")
        dock_params = {"unidock_search_mode": ud_search, "unidock_num_modes": ud_modes, "unidock_scoring": ud_scoring}

    ph_val = st.number_input("Target pH for protonation", 0.0, 14.0, 7.4, 0.1, key="ph_val")
    
    st.markdown("---")

    # Store parameters
    st.session_state.dock_params = dock_params

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — RECEPTOR PREPARATION (formerly Tab 2, moved after engine)
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    if not PRODY_AVAILABLE:
        st.warning("⚠️ ProDy not installed. Install with `pip install prody` for co-crystal ligand auto-detection and RMSD calculation.")
    if not check_obabel():
        st.error("❌ OpenBabel (obabel) not found. Install with `conda install -c conda-forge openbabel`")

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Section 3.1: Receptor</div>
        <div class="step-heading">📦 Receptor Preparation</div>
    """, unsafe_allow_html=True)

    rec_source = st.selectbox(
        "Receptor source",
        ["Search by PDB ID", "Search by protein name", "Upload PDB file"],
        key="receptor_source_select"
    )

    receptor_pdb_path = None

    if rec_source == "Search by PDB ID":
        col1, col2 = st.columns([3, 1])
        with col1:
            pdb_id_input = st.text_input("PDB ID", placeholder="Enter 4-character PDB ID (e.g. 4EY7)")
        with col2:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            direct_download = st.button("⬇ Direct Download", use_container_width=True)
        
        if direct_download and pdb_id_input:
            pdb_id = pdb_id_input.strip().upper()
            if re.match(r'^[0-9A-Z]{4}$', pdb_id):
                with st.spinner(f"Downloading PDB structure {pdb_id} from RCSB..."):
                    out = os.path.join(OUTPUT_DIR, f"{pdb_id}.pdb")
                    if download_pdb_direct(pdb_id, out):
                        receptor_pdb_path = out
                        st.session_state.receptor_pdb_path = out
                        
                        info = get_pdb_info(pdb_id)
                        if info.get("title"):
                                st.info(f"📄 PDB Title: {info['title'][:1000]}")
                    else:
                        st.error(f"Failed to download {pdb_id}")
            else:
                st.error("Invalid PDB ID format")

    elif rec_source == "Search by protein name":
        search_q = st.text_input("Search by protein name or gene", 
                                 placeholder="e.g., EGFR, COVID-19 main protease, BRD4")
        
        if search_q:
            with st.spinner(f"Searching RCSB for '{search_q}'..."):
                hits = search_rcsb_pdb(search_q, top_n=15)
            
            if hits:
                
                result_options = []
                for h in hits:
                    display = f"{h['pdb_id']}"
                    if h.get('resolution') and h['resolution'] != 'N/A':
                        try:
                            res_val = float(h['resolution']) if isinstance(h['resolution'], (int, float)) else float(h['resolution'])
                            display += f" | {res_val:.1f}Å"
                        except:
                            display += f" | {h['resolution']}"
                    if h.get('method') and h['method'] != 'N/A':
                        display += f" | {h['method']}"
                    if h.get('title'):
                        display += f" | {h['title']}"
                    result_options.append((h['pdb_id'], display))
                
                selected_pdb = st.selectbox("Select structure to download",
                                            options=[r[0] for r in result_options],
                                            format_func=lambda x: next((r[1] for r in result_options if r[0]==x), x))
                
                if st.button("⬇ Download Selected Structure", type="secondary", use_container_width=True):
                    out = os.path.join(OUTPUT_DIR, f"{selected_pdb}.pdb")
                    with st.spinner(f"Downloading {selected_pdb}..."):
                        if download_pdb_direct(selected_pdb, out):
                            receptor_pdb_path = out
                            st.session_state.receptor_pdb_path = out
                        else:
                            st.error(f"Failed to download {selected_pdb}")

    elif rec_source == "Upload PDB file":
        uploaded = st.file_uploader("Upload PDB/mmCIF file", type=["pdb","cif","mmcif","ent"])
        if uploaded:
            suffix = Path(uploaded.name).suffix or ".pdb"
            out = os.path.join(OUTPUT_DIR, f"receptor_{timestamp_tag()}{suffix}")
            with open(out, "wb") as f: 
                f.write(uploaded.getvalue())
            receptor_pdb_path = out
            st.session_state.receptor_pdb_path = out
            st.success(f"✅ Uploaded: {uploaded.name}")

    if st.session_state.receptor_pdb_path and os.path.exists(st.session_state.receptor_pdb_path):
        receptor_pdb_path = st.session_state.receptor_pdb_path
        st.markdown(
            f"""
            <div class="current-receptor-card">
                <div class="current-receptor-label">Current receptor</div>
                <div class="current-receptor-path">📄 {receptor_pdb_path}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        file_size = os.path.getsize(receptor_pdb_path) / 1024
        st.caption(f"File size: {file_size:.1f} KB")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Section 3.2: Grid Box Generator</div>
        <div class="step-heading">🎯 Grid Box Generator</div>
    """, unsafe_allow_html=True)

    if receptor_pdb_path and os.path.exists(receptor_pdb_path):
        if st.button("🔍 Detect Co-crystal Ligand", type="secondary"):
            if PRODY_AVAILABLE:
                with st.spinner("Analyzing PDB file for co-crystal ligands..."):
                    info = detect_cocrystal_ligand(receptor_pdb_path)
                if info["found"]:
                    st.session_state.cocrystal_info = info
                else:
                    st.warning(f"⚠️ {info.get('error', 'No ligand detected')}")
            else:
                st.error("ProDy not installed")

    if st.session_state.get("cocrystal_info") and st.session_state.cocrystal_info.get("found"):
        info = st.session_state.cocrystal_info
        candidates = info.get("all_candidates") or []
        if len(candidates) > 1:
            def _lig_fmt(key):
                row = next((c for c in candidates if c.get("key") == key), None)
                if not row:
                    return key
                return (
                    f"{row.get('resname')} | chain {row.get('chain') or '-'} | "
                    f"resid {row.get('resid')} | {row.get('n_atoms')} atoms"
                )

            selected_key = st.selectbox(
                "Reference ligand for docking grid",
                options=[c["key"] for c in candidates],
                format_func=_lig_fmt,
                index=next((i for i, c in enumerate(candidates) if c.get("key") == info.get("key")), 0),
                key="acd_reference_ligand_key",
                help="This ligand is removed from the receptor and its centroid defines the docking box.",
            )
            selected = next((c for c in candidates if c.get("key") == selected_key), None)
            if selected and selected.get("key") != info.get("key"):
                info = dict(info)
                info.update({
                    "resname": selected.get("resname"),
                    "chain": selected.get("chain"),
                    "resid": selected.get("resid"),
                    "n_atoms": selected.get("n_atoms"),
                    "center": (selected.get("cx"), selected.get("cy"), selected.get("cz")),
                    "sel_str": selected.get("sel_str"),
                    "key": selected.get("key"),
                    "message": (
                        f"{selected.get('resname')} (chain {selected.get('chain') or '-'}, "
                        f"resid {selected.get('resid')}, {selected.get('n_atoms')} atoms)"
                    ),
                })
                st.session_state.cocrystal_info = info
        chain = info.get("chain") or "-"
        center = info.get("center") or (0.0, 0.0, 0.0)
        st.markdown(
            f"""
            <div class="ligand-detect-card">
                <div class="ligand-detect-item">
                    <div class="ligand-detect-label">Detected ligand</div>
                    <div class="ligand-detect-value">✅ {info.get('resname', 'LIG')} (Chain {chain})</div>
                </div>
                <div class="ligand-detect-item">
                    <div class="ligand-detect-label">Center Coordinates</div>
                    <div class="ligand-detect-value">📍 ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    center_mode = st.radio("Box Center",
                           ["Use co-crystal ligand center", "Enter XYZ manually"],
                           horizontal=True,
                           key="box_center_mode")

    auto_cx = auto_cy = auto_cz = None
    if center_mode == "Use co-crystal ligand center" and st.session_state.cocrystal_info:
        cx, cy, cz = st.session_state.cocrystal_info["center"]
        auto_cx, auto_cy, auto_cz = cx, cy, cz
        st.info(f"📍 Using co-crystal center: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: auto_cx = st.number_input("Center X (Å)", value=0.0, step=5.0, format="%.2f")
        with c2: auto_cy = st.number_input("Center Y (Å)", value=0.0, step=5.0, format="%.2f")
        with c3: auto_cz = st.number_input("Center Z (Å)", value=0.0, step=5.0, format="%.2f")

    st.markdown("#### 📦 Box Size (Å)")
    bs1, bs2, bs3 = st.columns(3)
    with bs1: box_sx = st.slider("Size X (Å)", 10, 40, 20, 2, key="box_sx")
    with bs2: box_sy = st.slider("Size Y (Å)", 10, 40, 20, 2, key="box_sy")
    with bs3: box_sz = st.slider("Size Z (Å)", 10, 40, 20, 2, key="box_sz")
    
    volume = box_sx * box_sy * box_sz
    st.info(f"📐 Box volume: **{volume:,} Å³**")

    if st.button("⚙️ Prepare Receptor", type="primary", use_container_width=True):
        if not receptor_pdb_path or not os.path.exists(receptor_pdb_path):
            st.error("❌ Please load a receptor PDB file first.")
        elif not check_obabel():
            st.error("❌ OpenBabel not found.")
        else:
            work = Path(OUTPUT_DIR) / f"receptor_{timestamp_tag()}"
            work.mkdir(parents=True, exist_ok=True)
            with st.spinner("Preparing receptor..."):
                try:
                    rec_result = prepare_receptor_vina_batch(
                        receptor_pdb_path, work,
                        auto_cx, auto_cy, auto_cz,
                        box_sx, box_sy, box_sz
                    )
                    st.session_state.receptor_result = rec_result
                    st.success("✅ Receptor preparation completed!")
                    
                    with st.expander("📋 Preparation Log", expanded=True):
                        st.text(format_receptor_preparation_log(rec_result))
                except Exception as e:
                    st.error(f"❌ Receptor preparation failed: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — LIGAND PREPARATION (with RMSD validation added)
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">Section 4.1: RMSD Calculation</div>
        <div class="step-heading">⚗️ Docking Protocol Validation</div>
    """, unsafe_allow_html=True)

    rec_result = st.session_state.get("receptor_result", None)

    if not rec_result:
        st.warning("⚠️ No receptor prepared yet. Go to the **Receptor Preparation** tab first.")

    # ========== REDOCKING VALIDATION SECTION
    if rec_result and st.session_state.receptor_pdb_path and st.session_state.dock_bin_path:
            
            # Use the already-detected co-crystal ligand from Receptor Preparation
            cocrystal_info = st.session_state.get('cocrystal_info')
            
            if cocrystal_info and cocrystal_info.get('found'):
                # Display the already-detected ligand info
                detected_resname = cocrystal_info['resname']
                detected_chain = cocrystal_info.get('chain') or '-'
                detected_resid = cocrystal_info.get('resid') or ''
                center = cocrystal_info.get('center') or (0.0, 0.0, 0.0)
                
                st.markdown(
                    f"""
                    <div class="ligand-detect-card">
                        <div class="ligand-detect-item">
                            <div class="ligand-detect-label">Co-crystal Ligand (from Receptor Prep)</div>
                            <div class="ligand-detect-value">✅ {detected_resname} (Chain {detected_chain}, Resid {detected_resid})</div>
                        </div>
                        <div class="ligand-detect-item">
                            <div class="ligand-detect-label">Center Coordinates</div>
                            <div class="ligand-detect-value">📍 ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Only ask for SMILES
                cocrystal_smiles = st.text_input(
                    "🔬 SMILES of Co-crystallized Ligand",
                    placeholder="e.g., CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
                    help="Enter the SMILES of the ligand that is co-crystallized with your receptor for RMSD validation.",
                    key="cocrystal_smiles_input"
                )
                
                # Show protocol summary
                if st.session_state.current_engine in ("VINA", "VINAXB"):
                    protocol_summary = (
                        f"{st.session_state.current_engine}: exhaustiveness {st.session_state.dock_params.get('exhaustiveness')}, "
                        f"modes {st.session_state.dock_params.get('num_modes')}, energy range {st.session_state.dock_params.get('energy_range')} kcal/mol"
                    )
                elif st.session_state.current_engine == "GNINA":
                    protocol_summary = (
                        f"GNINA: exhaustiveness {st.session_state.dock_params.get('gnina_exhaustiveness')}, "
                        f"modes {st.session_state.dock_params.get('gnina_num_modes')}, CNN scoring {st.session_state.dock_params.get('gnina_cnn_scoring')}"
                    )
                else:
                    protocol_summary = (
                        f"Uni-Dock: search mode {st.session_state.dock_params.get('unidock_search_mode')}, "
                        f"modes {st.session_state.dock_params.get('unidock_num_modes')}, scoring {st.session_state.dock_params.get('unidock_scoring')}"
                    )
                st.info(f"🔁 Redocking uses the same protocol as batch docking: {protocol_summary}")

                with st.expander("⚙️ Redocking accuracy options", expanded=True):
                    use_crystal_box = st.checkbox(
                        "Use compact box around the crystal ligand for validation",
                        value=True,
                        help="Keeps redocking focused on the known binding site.",
                        key="use_crystal_box"
                    )
                    rb1, rb2, rb3 = st.columns(3)
                    with rb1:
                        redock_padding = st.slider("Box padding (Å)", 4.0, 12.0, 8.0, 0.5, key="redock_padding")
                    with rb2:
                        redock_min_box = st.slider("Minimum box side (Å)", 8.0, 18.0, 12.0, 0.5, key="redock_min_box")
                    with rb3:
                        redock_max_box = st.slider("Maximum box side (Å)", 16.0, 32.0, 24.0, 0.5, key="redock_max_box")
                
                if st.button("🔄 Perform Redocking & Calculate RMSD", type="secondary", use_container_width=True, key="redock_button"):
                    if not cocrystal_smiles.strip():
                        st.error("❌ Please enter the SMILES string of the co-crystal ligand.")
                    elif not st.session_state.dock_bin_path:
                        st.error("❌ Please find/configure the docking binary first using the 'Find Binary' button above.")
                    else:
                        redock_params = dict(st.session_state.dock_params)
                        redock_params["log_placeholder"] = None
                        redock_params["use_crystal_box"] = use_crystal_box
                        redock_params["redock_box_padding"] = redock_padding
                        redock_params["redock_min_box_size"] = redock_min_box
                        redock_params["redock_max_box_size"] = redock_max_box
                        work = unique_workdir(Path(OUTPUT_DIR) / f"redock_validation_{timestamp_tag()}")
                        work.mkdir(parents=True, exist_ok=True)
                        
                        with st.spinner(f"Running redocking validation with {st.session_state.current_engine} (this may take a few minutes)..."):
                            redock_result = redock_cocrystal_ligand_from_smiles(
                                rec_result, 
                                cocrystal_smiles.strip(), 
                                detected_resname,
                                st.session_state.current_engine, 
                                st.session_state.dock_bin_path, 
                                redock_params, 
                                work,
                                st.session_state.receptor_pdb_path,
                                cocrystal_info.get('chain'),
                                cocrystal_info.get('resid')
                            )
                        
                        if redock_result and redock_result['success']:
                            st.session_state.redock_result = redock_result
                            rmsd_val = redock_result['rmsd']
                            message, status = get_rmsd_validation_message(rmsd_val)
                            
                            if status == "excellent":
                                st.success(f"🎉 {message}")
                            elif status == "good":
                                st.success(f"✅ {message}")
                            elif status == "acceptable":
                                st.warning(f"⚠️ {message}")
                            else:
                                st.error(f"❌ {message}")
                            
                            binding_affinity = redock_result.get("binding_affinity")
                            affinity_text = fmt2(binding_affinity)

                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a: st.metric("RMSD (Pose 1)", fmt2(rmsd_val, " Å"))
                            with col_b: st.metric("Binding affinity (kcal/mol)", affinity_text)
                            with col_c: st.metric("Original Atoms", redock_result.get('n_atoms_original', '?'))
                            with col_d: st.metric("Docked Heavy Atoms", redock_result.get('n_atoms_docked', '?'))

                            rmsd_details = redock_result.get("rmsd_details") or {}
                            if rmsd_details:
                                pose_rows = rmsd_details.get("pose_rmsds") or []
                                if pose_rows:
                                    with st.expander("Pose RMSD table (after Kabsch alignment)"):
                                        st.dataframe(pd.DataFrame(round_numeric_results(pose_rows)), use_container_width=True)

                            st.markdown("#### Pose Overlay")
                            render_redocking_pose_viewer(
                                redock_result.get('original_ligand_pdb'),
                                redock_result.get('docked_sdf'),
                                receptor_pdb=rec_result.get("receptor_pdb") if rec_result else None,
                                redock_result=redock_result,
                                key_prefix="redock_validation",
                            )
                            
                            col_d, col_e = st.columns(2)
                            with col_d:
                                if redock_result.get('docked_sdf'):
                                    with open(redock_result['docked_sdf'], "rb") as f:
                                        st.download_button("📥 Redocked SDF", data=f, 
                                                           file_name=os.path.basename(redock_result['docked_sdf']), 
                                                           mime="chemical/x-mdl-sdfile")
                            with col_e:
                                if redock_result.get('docked_pdbqt'):
                                    with open(redock_result['docked_pdbqt'], "rb") as f:
                                        st.download_button("📥 Redocked PDBQT", data=f, 
                                                           file_name=os.path.basename(redock_result['docked_pdbqt']), 
                                                           mime="chemical/x-pdbqt")
                            
                            if rmsd_val is None:
                                st.warning("💡 **Recommendation:** RMSD could not be calculated. Check that the SMILES matches the co-crystal ligand residue exactly.")
                            elif rmsd_val < 2.0:
                                st.info("💡 **Recommendation:** Your docking protocol is validated! Proceed with batch docking.")
                            elif rmsd_val < 3.0:
                                st.info("💡 **Recommendation:** Consider increasing exhaustiveness or adjusting box size.")
                            else:
                                st.warning("💡 **Recommendation:** Optimize your docking protocol before batch docking.")
                        else:
                            st.error("Redocking validation failed. Check SMILES and residue name.")
            else:
                st.warning("⚠️ No co-crystal ligand detected in the Receptor Preparation step. Please prepare a receptor with a detected ligand first.")
            
            st.markdown("---")
      
    # ========== LIGAND SOURCE SECTION ==========
    st.markdown("### ⚗️ Ligand Source")

    lig_source = st.radio("Ligand input",
                          ["From IMPPAT (CSV - ALL compounds)", "Manual SMILES", "Upload SDF/MOL2"],
                          horizontal=True, key="lig_source")

    selected_smiles_list = []

    if lig_source == "From IMPPAT (CSV - ALL compounds)":
        if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
            df_lig = pd.read_csv(st.session_state.csv_path)
            df_with_smi = df_lig[df_lig["SMILES_IMPPAT"].notna() & (df_lig["SMILES_IMPPAT"]!="")]
            st.info(f"📊 {len(df_with_smi)} compounds with SMILES available from IMPPAT.")

            filter_ro5 = st.checkbox("Filter by Lipinski's Rule of 5 (passed only)", value=False)
            if filter_ro5:
                df_with_smi = df_with_smi[df_with_smi["LIPINSKI"]=="Passed"]
                st.caption(f"📊 {len(df_with_smi)} compounds after Lipinski filter.")

            df_to_dock = df_with_smi

            if len(df_to_dock) > 0:
                st.dataframe(df_to_dock[["IMPHY_ID","CID","Phytochemical_Name","SMILES_IMPPAT","LIPINSKI"]], 
                           use_container_width=True, height=300)
                selected_smiles_list = list(zip(df_to_dock["Phytochemical_Name"], df_to_dock["SMILES_IMPPAT"]))
            else:
                st.warning("No compounds available.")
        else:
            st.warning("No IMPPAT CSV found. Run Phytochemical Extraction first (Tab 1).")

    elif lig_source == "Manual SMILES":
        manual_input = st.text_area(
            "SMILES input (one per line, optionally with name)",
            placeholder="CC(=O)Oc1ccccc1C(=O)O  Aspirin\nC1=CC(=CC=C1C2=CC(=O)C3=C(O2)C=C(C=C3O)O)O  Apigenin",
            height=150
        )
        if manual_input.strip():
            for line in manual_input.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts:
                    smi = parts[0]
                    if len(parts) > 1:
                        name = " ".join(parts[1:])
                    else:
                        name = f"lig_{len(selected_smiles_list)+1}"
                    selected_smiles_list.append((name, smi))
            st.success(f"✅ Parsed {len(selected_smiles_list)} SMILES entries.")

    else:
        uploaded_lig = st.file_uploader("Upload ligand file", type=["sdf","mol2","smi"])
        if uploaded_lig:
            tmp = os.path.join(OUTPUT_DIR, f"uploaded_lig_{timestamp_tag()}{Path(uploaded_lig.name).suffix}")
            with open(tmp, "wb") as f: 
                f.write(uploaded_lig.getvalue())
            ext = Path(uploaded_lig.name).suffix.lower()
            
            with st.spinner(f"Processing {uploaded_lig.name}..."):
                try:
                    if ext == ".smi":
                        content = open(tmp).read()
                        for line in content.strip().splitlines():
                            if line.strip():
                                parts = line.strip().split()
                                smi = parts[0]
                                name = " ".join(parts[1:]) if len(parts)>1 else f"mol_{len(selected_smiles_list)+1}"
                                selected_smiles_list.append((name, smi))
                    else:
                        if ext == ".sdf":
                            supp = Chem.SDMolSupplier(tmp, sanitize=False, removeHs=False)
                        elif ext == ".mol2":
                            supp = Chem.MolFromMol2File(tmp, sanitize=False, removeHs=False)
                            supp = [supp] if supp else []
                        else:
                            st.error(f"Unsupported file format: {ext}")
                            supp = []
                        
                        for i, mol in enumerate(supp):
                            if mol is None:
                                continue
                            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                            smi = Chem.MolToSmiles(mol, canonical=True)
                            if smi:
                                selected_smiles_list.append((name.strip() or f"mol_{i+1}", smi))
                    st.success(f"✅ Loaded {len(selected_smiles_list)} molecules")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

    # ── Prepare Ligands button ──────────────────────────────────────
    if selected_smiles_list:
        st.markdown("---")
        col_prep1, col_prep2 = st.columns([3, 1])
        with col_prep1:
            st.markdown(
                f"<div style='padding:10px 0;font-size:0.93rem;color:#2e7d32;font-weight:600;'>"
                f"🧪 {len(selected_smiles_list)} ligand(s) ready for preparation</div>",
                unsafe_allow_html=True
            )
        with col_prep2:
            prep_btn = st.button("🧪 Prepare Ligands", type="primary", use_container_width=True, key="prep_ligands_btn")

        if prep_btn:
            ph_val_prep = st.session_state.get("ph_val", 7.4)
            work_prep = unique_workdir(Path(OUTPUT_DIR) / f"ligprep_{timestamp_tag()}")
            work_prep.mkdir(parents=True, exist_ok=True)
            prep_results = []
            prog_p = st.progress(0)
            stat_p = st.empty()
            for i, (lig_name, smiles) in enumerate(selected_smiles_list):
                stat_p.text(f"Preparing {i+1}/{len(selected_smiles_list)}: {lig_name}")
                prog_p.progress((i + 1) / len(selected_smiles_list))
                lig_dir = work_prep / f"lig_{safe_name(lig_name)}"
                lig_dir.mkdir(exist_ok=True)
                try:
                    lr = prepare_ligand_vina_batch(smiles, lig_name, ph_val_prep, lig_dir)
                    prep_results.append(lr)
                except Exception as e:
                    prep_results.append({"success": False, "name": lig_name, "error": str(e)})
            prog_p.empty()
            stat_p.empty()
            ok = [r for r in prep_results if r.get("success")]
            st.success(f"✅ Prepared {len(ok)}/{len(prep_results)} ligands successfully.")
            if len(ok) < len(prep_results):
                failed_names = [r["name"] for r in prep_results if not r.get("success")]
                st.warning(f"⚠️ Failed: {', '.join(failed_names)}")
            st.session_state["ligand_prep_results"] = prep_results
            st.session_state["ligand_prep_signature"] = {
                "ph": ph_val_prep,
                "ligands": [(str(n), str(s)) for n, s in selected_smiles_list],
            }

    # Store the selected ligand list for the batch docking tab
    st.session_state.selected_smiles_list = selected_smiles_list

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — BATCH DOCKING (Results & Download)
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    rec_result = st.session_state.receptor_result
    dock_bin_path = st.session_state.dock_bin_path
    dock_params = st.session_state.dock_params
    # Get pH value from the widget's session state if available, otherwise use default
    ph_val = st.session_state.get("ph_val", 7.4)
    selected_smiles_list = st.session_state.get("selected_smiles_list", [])
    current_engine = st.session_state.current_engine

    st.markdown("""
    <div class="step-card">
        <div class="step-title">Batch Docking</div>
        <div class="step-heading">⚗️ Run Batch Docking & Download Results</div>
    """, unsafe_allow_html=True)

    # ── Pre-flight status ────────────────────────────────────────────
    _status_parts = []
    if not rec_result:
        st.warning("⚠️ No receptor prepared yet. Go to the **Receptor Preparation** tab first.")
    if not dock_bin_path:
        st.warning("⚠️ No docking binary configured. Go to the **Docking Engine** tab first.")
    if not selected_smiles_list:
        st.info("ℹ️ No ligands selected. Go to the **Ligand Preparation** tab to add ligands.")

    # Auto-generate internal session name (not shown to user)
    current_receptor_for_session = st.session_state.get("receptor_pdb_path")
    if (
        "session_name" not in st.session_state
        or st.session_state.get("session_receptor_path") != current_receptor_for_session
    ):
        st.session_state["session_name"] = default_docking_session_name(current_receptor_for_session)
        st.session_state["session_receptor_path"] = current_receptor_for_session
    session_name = st.session_state["session_name"]

    run_btn = st.button("▶ RUN BATCH DOCKING", type="primary", use_container_width=True,
                        disabled=(len(selected_smiles_list)==0 or rec_result is None or dock_bin_path is None))

    if run_btn:
        if not rec_result:
            st.error("❌ No receptor prepared.")
        elif not selected_smiles_list:
            st.error("❌ No ligands selected.")
        elif not dock_bin_path:
            st.error("❌ No docking binary found. Please configure the engine and find binary first.")
        else:
            work = unique_workdir(Path(OUTPUT_DIR) / safe_name(session_name))
            work.mkdir(parents=True, exist_ok=True)

            ligand_results = []
            dock_records = []

            prep_signature = {
                "ph": ph_val,
                "ligands": [(str(n), str(s)) for n, s in selected_smiles_list],
            }
            cached_prep = st.session_state.get("ligand_prep_results") or []
            cached_signature = st.session_state.get("ligand_prep_signature")
            required_key = "sdf" if current_engine == "UNIDOCK" else "pdbqt"

            cache_is_usable = (
                cached_signature == prep_signature
                and len(cached_prep) == len(selected_smiles_list)
                and all(
                    r.get("success")
                    and r.get(required_key)
                    and os.path.exists(r.get(required_key))
                    for r in cached_prep
                )
            )

            if cache_is_usable:
                ligand_results = cached_prep
                st.info(f"✅ Using {len(ligand_results)} ligand(s) already prepared in the Ligand Preparation tab.")
            else:
                st.subheader("📦 Ligand Preparation")
                st.caption("Preparing ligands now because no matching prepared ligand set was found.")
                prog_lig = st.progress(0)
                lig_status = st.empty()

                for i, (lig_name, smiles) in enumerate(selected_smiles_list):
                    lig_status.text(f"Preparing {i+1}/{len(selected_smiles_list)}: {lig_name}")
                    prog_lig.progress((i+1)/len(selected_smiles_list))
                    lig_dir = work / f"lig_{safe_name(lig_name)}"
                    lig_dir.mkdir(exist_ok=True)
                    try:
                        lr = prepare_ligand_vina_batch(smiles, lig_name, ph_val, lig_dir)
                        ligand_results.append(lr)
                    except Exception as e:
                        st.warning(f"⚠️ Failed for {lig_name}: {e}")
                prog_lig.empty()
                lig_status.empty()
                st.session_state["ligand_prep_results"] = ligand_results
                st.session_state["ligand_prep_signature"] = prep_signature

            if ligand_results:
                pass  # silently continue
            else:
                st.error("❌ No ligands prepared.")
                st.stop()

            st.subheader("🔬 Running Docking Calculations")
            prog_dock = st.progress(0)
            dock_status = st.empty()

            for i, lr in enumerate(ligand_results):
                dock_status.text(f"Docking {i+1}/{len(ligand_results)}: {lr['name']}…")
                prog_dock.progress((i+1)/len(ligand_results))

                lig_input = lr["sdf"] if current_engine == "UNIDOCK" else lr["pdbqt"]
                out_prefix = str(work / f"dock_{safe_name(lr['name'])}")
                run_params = dict(dock_params)
                run_params["log_placeholder"] = None

                try:
                    docked_pdbqt, docked_sdf, dock_log = dock_one_ligand(
                        current_engine, dock_bin_path,
                        rec_result["receptor_pdbqt"],
                        lig_input, rec_result["config_file"],
                        out_prefix, run_params
                    )
                    try:
                        score_info = extract_top_score(docked_pdbqt, docked_sdf, current_engine)
                        top_score = score_info["score"]
                        dock_records.append({
                            "name": lr["name"],
                            "docked_pdbqt": docked_pdbqt,
                            "docked_sdf": docked_sdf,
                            "top_score": top_score,
                            "log": dock_log,
                        })
                    except Exception as e:
                        dock_records.append({
                            "name": lr["name"],
                            "docked_pdbqt": docked_pdbqt,
                            "docked_sdf": docked_sdf,
                            "top_score": None,
                            "log": dock_log,
                        })
                except Exception as e:
                    dock_records.append({
                        "name": lr["name"],
                        "docked_pdbqt": None,
                        "docked_sdf": None,
                        "top_score": None,
                        "log": str(e),
                    })

            prog_dock.empty()
            dock_status.empty()

            st.session_state.ligand_results = ligand_results
            st.session_state.dock_records = dock_records

            with st.spinner("📦 Packaging results…"):
                session_files = create_docking_session_files(
                    work, session_name, rec_result, ligand_results, dock_records, current_engine,
                    st.session_state.get('redock_result')
                )
            st.session_state.session_files = session_files
            st.success("🎉 Docking complete! Results are shown below.")

    # ========== DISPLAY RESULTS ==========
    dock_records = st.session_state.dock_records
    session_files = st.session_state.session_files
    redock_result = st.session_state.redock_result
    current_engine = st.session_state.current_engine

    if dock_records:
        successful = [d for d in dock_records if d["top_score"] is not None]
        if successful:
            best = min(successful, key=lambda x: x["top_score"])
            scores = [d["top_score"] for d in successful]
            avg_score = sum(scores) / len(scores) if scores else 0

            st.markdown("### 📊 Docking Summary")
            
            if redock_result and redock_result.get('success'):
                rmsd_val = redock_result['rmsd']
                if rmsd_val is None:
                    rmsd_class = "rmsd-poor"
                    rmsd_text = "RMSD could not be calculated"
                elif rmsd_val < 1.5:
                    rmsd_class = "rmsd-excellent"
                    rmsd_text = f"RMSD = {rmsd_val:.2f} Å"
                elif rmsd_val < 2.0:
                    rmsd_class = "rmsd-good"
                    rmsd_text = f"RMSD = {rmsd_val:.2f} Å"
                else:
                    rmsd_class = "rmsd-poor"
                    rmsd_text = f"RMSD = {rmsd_val:.2f} Å"
                
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
                    <strong>🔄 Redocking Validation</strong><br>
                    <span class="{rmsd_class}">{rmsd_text}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if redock_result.get('original_ligand_pdb') and redock_result.get('docked_sdf'):
                    st.markdown("#### Redocking Pose Overlay")
                    render_redocking_pose_viewer(
                        redock_result.get('original_ligand_pdb'),
                        redock_result.get('docked_sdf'),
                        height=460,
                        receptor_pdb=rec_result.get("receptor_pdb") if rec_result else None,
                        redock_result=redock_result,
                        key_prefix="redock_results",
                    )
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("Total Ligands", len(dock_records))
            with col2: st.metric("Successful", len(successful))
            with col3: st.metric("Best Score", f"{best['top_score']:.2f} kcal/mol")
            with col4: st.metric("Average Score", f"{avg_score:.2f} kcal/mol")
            with col5: st.metric("Best Compound", best["name"][:15] + "...", help=best["name"])

            st.markdown("---")
            st.markdown("### 📈 Score Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_score_chart(dock_records, engine=current_engine or "VINA")
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                if len(successful) > 1:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.hist(scores, bins=min(10, len(scores)), color="steelblue", alpha=0.7, edgecolor="black")
                    ax2.axvline(avg_score, color="red", linestyle="--", label=f"Average: {avg_score:.2f}")
                    ax2.axvline(best['top_score'], color="green", linestyle="--", label=f"Best: {best['top_score']:.2f}")
                    ax2.set_xlabel("Binding Energy (kcal/mol)")
                    ax2.set_ylabel("Frequency")
                    ax2.set_title("Score Distribution")
                    ax2.legend()
                    st.pyplot(fig2)
                    plt.close(fig2)

            st.markdown("### 📋 Detailed Results")
            score_rows = []
            for d in dock_records:
                row = {
                    "Compound": d["name"],
                    "Score (kcal/mol)": f"{d['top_score']:.2f}" if d["top_score"] is not None else "—",
                    "Status": "✅" if d["top_score"] is not None else "❌",
                }
                score_rows.append(row)
            score_df = pd.DataFrame(score_rows)
            st.dataframe(score_df.sort_values("Score (kcal/mol)", na_position="last"), 
                        use_container_width=True, hide_index=True)

            st.markdown("### 🔍 Per-Ligand Results")
            for d in dock_records:
                score_label = fmt2(d["top_score"]) if d["top_score"] is not None else "failed"
                with st.expander(f"{'✅' if d['top_score'] is not None else '❌'} {d['name']} — Score: {score_label}"):
                    if d["top_score"] is not None:
                        try:
                            poses = parse_all_poses(d.get("docked_pdbqt"), d.get("docked_sdf"),
                                                    engine=current_engine or "VINA")
                            if poses:
                                st.markdown("**All binding poses:**")
                                st.dataframe(pd.DataFrame(round_numeric_results(poses)), use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.caption(f"⚠️ Could not parse poses: {e}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if d.get("docked_pdbqt") and os.path.exists(d["docked_pdbqt"]):
                                with open(d["docked_pdbqt"],"rb") as f:
                                    st.download_button(f"⬇ PDBQT", data=f,
                                                       file_name=os.path.basename(d["docked_pdbqt"]),
                                                       mime="chemical/x-pdbqt", key=f"pdbqt_{d['name']}")
                        with col2:
                            if d.get("docked_sdf") and os.path.exists(d["docked_sdf"]):
                                with open(d["docked_sdf"],"rb") as f:
                                    st.download_button(f"⬇ SDF", data=f,
                                                       file_name=os.path.basename(d["docked_sdf"]),
                                                       mime="chemical/x-mdl-sdfile", key=f"sdf_{d['name']}")

        st.markdown("---")
        st.markdown("### 📦 Download Results")

        if session_files and os.path.exists(session_files["zip"]):
            zip_size = os.path.getsize(session_files["zip"])
            with open(session_files["zip"], "rb") as f:
                st.download_button(
                    f"📦 Download ZIP ({zip_size/1024/1024:.1f} MB)",
                    data=f,
                    file_name=os.path.basename(session_files["zip"]),
                    mime="application/zip",
                    use_container_width=True,
                    type="primary",
                )
            st.caption("ZIP contains: receptor PDBQT · ligand SDFs & PDBQTs · docked poses · summary CSV · session report")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.85rem; padding:1rem;">
    🌿 IMPPATez · Natural Product Informatics Pipeline<br>
    <b>Features:</b> IMPPAT Database | RCSB PDB | RMSD Calculation | Co-crystal Redocking with SMILES | Unlimited Compound Docking<br>
    <b>Powered by:</b> RDKit | AutoDock Vina | GNINA | Uni-Dock | OpenBabel | ProDy | Meeko
</div>
""", unsafe_allow_html=True)
