"""
Script Maître — Projet Hekzam
==============================
Auteur : W. Wassim

Nouvelle logique en 4 phases :
  PHASE 1 — OCR sur TOUS les PDFs (correction + détection + pipeline)
  PHASE 2 — Génération des labels.json pour chaque PDF
  PHASE 3 — Reconnaissance sur TOUS les PDFs (après OCR complet)
  PHASE 4 — Matrice de confusion globale + accuracy + temps total

Usage :
    python run_pipeline.py --pdfs PDF_INPUT/ --json atomic-boxes.json
    python run_pipeline.py --pdfs PDF_INPUT/ --json atomic-boxes.json --debug
"""

import os
import sys
import json
import re
import argparse
import subprocess
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PYTHON = sys.executable

SCRIPTS = {
    "correction"    : "correction_distorsion.py",
    "detection"     : "detection_coins.py",
    "conversion"    : "convertir_json.py",
    "pipeline"      : "pipeline_ocr.py",
    "reconnaissance": "reconnaissance.py",
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def log(msg, niveau="INFO"):
    symboles = {"INFO": "─", "OK": "✓", "WARN": "⚠", "ERR": "✗"}
    print(f"  {symboles.get(niveau, ' ')}  {msg}")


def run(cmd, description):
    log(f"{description}...", "INFO")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    duree = time.time() - t0
    if result.returncode == 0:
        log(f"{description} -> OK  ({duree:.1f}s)", "OK")
        return True, duree
    else:
        log(f"{description} -> ECHEC (code {result.returncode})  ({duree:.1f}s)", "ERR")
        return False, duree


def verifier_scripts():
    manquants = [s for s in SCRIPTS.values() if not os.path.exists(s)]
    if manquants:
        print("\n[ERREUR] Scripts manquants :")
        for s in manquants:
            print(f"  x {s}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION LABELS.JSON
# ─────────────────────────────────────────────────────────────────────────────

def generer_labels(chemin_atomic_boxes, cases_json, dossier_cases):
    try:
        with open(chemin_atomic_boxes, "r", encoding="utf-8") as f:
            boxes = json.load(f)
        with open(cases_json, "r", encoding="utf-8") as f:
            cases = json.load(f)
    except Exception as e:
        log(f"Erreur lecture pour labels : {e}", "WARN")
        return None

    atomic_cases = []
    for key, val in boxes.items():
        parts = key.split("-")
        if parts[0] == "id" and len(parts) == 4:
            atomic_cases.append({
                "label": int(parts[3]),
                "cx": val["x"] + val["width"]  / 2,
                "cy": val["y"] + val["height"] / 2,
                "page": val["page"]
            })

    candidate_map = defaultdict(list)
    for case in cases:
        coins = case.get("coins", {})
        tl = coins.get("tl")
        bl = coins.get("bl")
        if tl is None:
            continue
        cx = tl[0] + 3.0
        cy = (tl[1] + bl[1]) / 2 if bl else tl[1] + 4.0
        page = case.get("page", 1)
        for i, ac in enumerate(atomic_cases):
            if ac["page"] != page:
                continue
            dy = abs(ac["cy"] - cy)
            dx = abs(ac["cx"] - cx)
            if dy < 5.0 and dx < 4.0:
                dist = (dx**2 + dy**2) ** 0.5
                candidate_map[i].append((dist, case["id"], ac["label"]))

    labels = {}
    for i, cands in candidate_map.items():
        cands.sort()
        _, case_id, label = cands[0]
        labels[case_id] = label

    chemin_labels = os.path.join(dossier_cases, "labels.json")
    os.makedirs(dossier_cases, exist_ok=True)
    with open(chemin_labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    log(f"labels.json genere : {len(labels)} cases -> {chemin_labels}", "OK")
    return chemin_labels


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — OCR D'UN PDF (sans reconnaissance)
# ─────────────────────────────────────────────────────────────────────────────

def ocr_pdf(chemin_pdf, chemin_json, debug=False):
    """Étapes 1-4 pour un PDF. Retourne les infos du PDF."""
    pdf_name = os.path.splitext(os.path.basename(chemin_pdf))[0]
    debut    = time.time()

    dossier_corriges = os.path.join("scans_corriges", pdf_name)
    dossier_coins    = os.path.join("results_coins",  pdf_name)
    cases_json       = os.path.join("results", pdf_name, "cases.json")
    dossier_cases    = os.path.join("results", pdf_name, "cases_hekzam")
    results_json     = os.path.join("results", pdf_name, "results.json")

    os.makedirs(dossier_corriges, exist_ok=True)
    os.makedirs(dossier_coins,    exist_ok=True)
    os.makedirs(os.path.dirname(cases_json), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  OCR : {pdf_name}")
    print(f"{'='*60}")

    etapes = {}
    durees = {}

    # Etape 1 : Correction
    cmd = [PYTHON, SCRIPTS["correction"],
           "--pdf", chemin_pdf, "--json", chemin_json, "--output", dossier_corriges]
    if debug: cmd.append("--debug")
    ok, d = run(cmd, "Etape 1 - Correction distorsion")
    etapes["correction"] = ok; durees["correction"] = d
    if not ok:
        return {"pdf": pdf_name, "statut": "echec_correction", "etapes": etapes,
                "durees": durees, "duree_ocr": round(time.time()-debut, 1),
                "dossier_cases": dossier_cases, "cases_json": cases_json,
                "results_json": results_json}

    # Etape 2 : Detection
    cmd = [PYTHON, SCRIPTS["detection"],
           "--input", dossier_corriges, "--output", dossier_coins]
    if debug: cmd.append("--debug")
    ok, d = run(cmd, "Etape 2 - Detection des coins")
    etapes["detection"] = ok; durees["detection"] = d
    if not ok:
        return {"pdf": pdf_name, "statut": "echec_detection", "etapes": etapes,
                "durees": durees, "duree_ocr": round(time.time()-debut, 1),
                "dossier_cases": dossier_cases, "cases_json": cases_json,
                "results_json": results_json}

    # Etape 3 : Conversion JSON
    pages_json = sorted([
        f for f in os.listdir(dossier_coins)
        if f.endswith(".json") and not f.startswith("converti_")
    ])
    if not pages_json:
        log("Aucun JSON trouve.", "ERR")
        return {"pdf": pdf_name, "statut": "echec_conversion", "etapes": etapes,
                "durees": durees, "duree_ocr": round(time.time()-debut, 1),
                "dossier_cases": dossier_cases, "cases_json": cases_json,
                "results_json": results_json}

    toutes_cases = []
    t_conv = time.time()
    for page_json_file in pages_json:
        page_json_path = os.path.join(dossier_coins, page_json_file)
        match = re.search(r'(\d+)', page_json_file)
        page_num = int(match.group(1)) if match else 1
        tmp_output = os.path.join(dossier_coins, f"converti_{page_json_file}")
        cmd = [PYTHON, SCRIPTS["conversion"],
               "--input", page_json_path, "--output", tmp_output, "--page", str(page_num)]
        ok, _ = run(cmd, f"Etape 3 - Conversion page {page_num}")
        etapes[f"conv_page{page_num}"] = ok
        if ok and os.path.exists(tmp_output):
            with open(tmp_output, 'r', encoding='utf-8') as f:
                toutes_cases.extend(json.load(f))
    durees["conversion"] = time.time() - t_conv

    with open(cases_json, 'w', encoding='utf-8') as f:
        json.dump(toutes_cases, f, indent=2, ensure_ascii=False)
    log(f"JSON fusionne : {len(toutes_cases)} cases -> {cases_json}", "OK")

    # Etape 4 : Pipeline OCR
    cmd = [PYTHON, SCRIPTS["pipeline"],
           "--images", dossier_corriges, "--json", cases_json]
    if debug: cmd.append("--debug")
    ok, d = run(cmd, "Etape 4 - Pipeline OCR")
    etapes["pipeline"] = ok; durees["pipeline"] = d
    if not ok:
        return {"pdf": pdf_name, "statut": "echec_pipeline", "etapes": etapes,
                "durees": durees, "duree_ocr": round(time.time()-debut, 1),
                "dossier_cases": dossier_cases, "cases_json": cases_json,
                "results_json": results_json}

    duree_ocr = time.time() - debut
    statut = "ok" if all(v for k, v in etapes.items()
                         if not k.startswith("conv_page")) else "partiel"
    return {
        "pdf": pdf_name, "statut": statut,
        "duree_ocr": round(duree_ocr, 1),
        "durees": {k: round(v, 1) for k, v in durees.items()},
        "dossier_cases": dossier_cases,
        "cases_json": cases_json,
        "results_json": results_json,
        "etapes": etapes,
        "accuracy": None,
        "duree_reco": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — RECONNAISSANCE D'UN PDF
# ─────────────────────────────────────────────────────────────────────────────

def reconnaitre_pdf(r):
    """Lance la reconnaissance sur un PDF déjà traité par OCR."""
    if r["statut"].startswith("echec"):
        log(f"{r['pdf']} ignore (echec OCR)", "WARN")
        return

    cmd = [PYTHON, SCRIPTS["reconnaissance"], "--dossier", r["dossier_cases"]]
    ok, d = run(cmd, f"Reconnaissance {r['pdf']}")
    r["duree_reco"] = round(d, 1)
    r["reconnaissance_ok"] = ok

    # Lire accuracy depuis results.json
    try:
        with open(r["results_json"], "r", encoding="utf-8") as f:
            res = json.load(f)
        if res and "accuracy" in res[0]:
            r["accuracy"] = res[0]["accuracy"]
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — RÉSUMÉ GLOBAL + MATRICE DE CONFUSION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

def afficher_resume_global(resultats, debut_global):
    duree_totale = time.time() - debut_global

    # Collecter toutes les prédictions + labels réels pour matrice globale
    y_vrai_global = []
    y_pred_global = []

    for r in resultats:
        if not os.path.exists(r.get("results_json", "")):
            continue
        try:
            with open(r["results_json"], "r", encoding="utf-8") as f:
                res = json.load(f)
            for entry in res:
                if (entry.get("chiffre_predit") is not None
                        and entry.get("label_reel") is not None):
                    y_vrai_global.append(entry["label_reel"])
                    y_pred_global.append(entry["chiffre_predit"])
        except Exception:
            pass

    # Matrice de confusion globale
    if y_vrai_global:
        n = 10
        matrice = np.zeros((n, n), dtype=int)
        for vrai, pred in zip(y_vrai_global, y_pred_global):
            if 0 <= vrai < n and 0 <= pred < n:
                matrice[vrai][pred] += 1
        accuracy_globale = (np.trace(matrice) / matrice.sum() * 100
                            if matrice.sum() > 0 else 0)

        print(f"\n{'='*60}")
        print(f"  MATRICE DE CONFUSION GLOBALE (tous PDFs)")
        print(f"{'='*60}")
        header = "       " + "  ".join(f"P{c}" for c in range(n))
        print(header)
        print("       " + "───" * n)
        for i in range(n):
            ligne = f"  R{i} | " + "  ".join(
                f"\033[92m{matrice[i][j]:2d}\033[0m" if i == j
                else f"{matrice[i][j]:2d}"
                for j in range(n)
            )
            print(ligne)
        print(f"{'='*60}")
        print(f"  Accuracy globale : {accuracy_globale:.2f}%  "
              f"({int(np.trace(matrice))}/{int(matrice.sum())} bien classes)")
        print(f"{'='*60}")
    else:
        accuracy_globale = None
        print("\n  Aucune donnee pour la matrice de confusion globale.")

    # Tableau par PDF
    print(f"\n{'='*60}")
    print(f"  RESUME PAR PDF")
    print(f"{'='*60}")
    print(f"  {'PDF':<25}  {'Statut':<12}  {'Accuracy':>10}  {'OCR':>7}  {'Reco':>7}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*10}  {'─'*7}  {'─'*7}")

    for r in resultats:
        sym      = "OK" if r["statut"] == "ok" else "~" if r["statut"] == "partiel" else "X"
        accuracy = f"{r['accuracy']:.2f}%" if r.get("accuracy") is not None else "N/A"
        d_ocr    = f"{r.get('duree_ocr', '?')}s"
        d_reco   = f"{r.get('duree_reco', '?')}s"
        print(f"  [{sym}] {r['pdf']:<23}  {r['statut']:<12}  "
              f"{accuracy:>10}  {d_ocr:>7}  {d_reco:>7}")

    ok    = sum(1 for r in resultats if r["statut"] == "ok")
    echec = sum(1 for r in resultats if r["statut"].startswith("echec"))

    print(f"\n  OK={ok}  echec={echec}")
    print(f"  Temps total : {duree_totale:.1f}s")
    if accuracy_globale is not None:
        print(f"  Accuracy globale : {accuracy_globale:.2f}%")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def traiter_dossier(dossier_pdfs, chemin_json, debug=False):
    verifier_scripts()

    pdfs = sorted([
        os.path.join(dossier_pdfs, f)
        for f in os.listdir(dossier_pdfs)
        if f.lower().endswith(".pdf")
    ])

    if not pdfs:
        print(f"\n[ERREUR] Aucun PDF trouve dans : {dossier_pdfs}")
        sys.exit(1)

    debut_global = time.time()

    print(f"\n{'='*60}")
    print(f"  PIPELINE HEKZAM - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"  {len(pdfs)} PDF(s) a traiter")
    print(f"  JSON : {chemin_json}")
    print(f"{'='*60}")

    # ── PHASE 1 : OCR sur tous les PDFs ──
    print(f"\n{'='*60}")
    print(f"  PHASE 1 - OCR (extraction des cases)")
    print(f"{'='*60}")

    resultats = []
    for chemin_pdf in pdfs:
        r = ocr_pdf(chemin_pdf, chemin_json, debug=debug)
        resultats.append(r)

    # ── PHASE 2 : Génération des labels.json ──
    print(f"\n{'='*60}")
    print(f"  PHASE 2 - GENERATION DES LABELS")
    print(f"{'='*60}")

    for r in resultats:
        if not r["statut"].startswith("echec"):
            generer_labels(chemin_json, r["cases_json"], r["dossier_cases"])

    # ── PHASE 3 : Reconnaissance sur tous les PDFs ──
    print(f"\n{'='*60}")
    print(f"  PHASE 3 - RECONNAISSANCE (tous les PDFs)")
    print(f"{'='*60}")

    for r in resultats:
        reconnaitre_pdf(r)

    # ── PHASE 4 : Résumé global + matrice de confusion ──
    afficher_resume_global(resultats, debut_global)

    with open("run_summary.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    print(f"  Resume complet -> run_summary.json\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script maitre - Pipeline complet Hekzam"
    )
    parser.add_argument("--pdfs",  required=True, help="Dossier des PDFs scannes")
    parser.add_argument("--json",  required=True, help="Fichier atomic-boxes.json")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    args = parser.parse_args()

    traiter_dossier(args.pdfs, args.json, debug=args.debug)
