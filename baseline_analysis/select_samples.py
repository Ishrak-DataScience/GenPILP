# -*- coding: utf-8 -*-
"""
select_samples.py
=================
Step 1: choose the interesting complexes the whole baseline runs on.

Metadata filters first, RDKit only on the survivors (RDKit on 109k rows is the
slow way to get the same answer).

WHAT "INTERESTING" MEANS HERE (decided with the user)
------------------------------------------------------
* FDA status 4 (approved) first, topped up from 1/2/3 (in trials -- we do not
  know whether they are ongoing or why they failed, so they are still of interest).
* Targets ranked by how many independent measurements metadata.tsv holds for
  them: more measurements = more general interest in the target.
* NOT interesting, and excluded: ions, buffers, cryo-protectants, detergents,
  sugars, cofactors and vitamins (the "flavin problem" -- FMN/FAD/NAD appear in
  thousands of structures as crystallisation aids, not as drugs), free
  nucleotides, free amino acids and lipids.
* Metal-coordinating complexes (metal_count > 0) are excluded outright, which
  also drops carbonic anhydrase II (P00918) despite it being the single
  best-measured target: its sulfonamide inhibitors bind by coordinating the
  catalytic Zn, and GNINA scores metal coordination poorly without explicit
  parameterisation, so those docking deltas would measure the scoring function's
  blind spot rather than the model.
* One complex per ligand and at most two per target, so the set cannot collapse
  onto a single over-represented drug or protein.

A NOTE ON THE TARGET RANKING
-----------------------------
The top-20 target list circulated by e-mail cannot be reproduced from this
metadata.tsv: D0VWR1, Q8DIQ1, Q8DIF8, P02945, P0A405, Q7NDN8, P0A407 and D0VWR7
have ZERO rows here, and shared IDs disagree (P00918: 897 here vs 1112 there;
Unknown: 2885 vs 14459). The local file is byte-identical to the Drive copy
(5,343,497 bytes), so that list came from a larger table -- most likely the full
PLIP corpus before the FDA/UniProt join. Per the user's decision, ranking uses
THIS file's counts; overview.txt prints both rankings side by side so the
difference stays visible.

HOW TO RUN
-----------
    python baseline_analysis/select_samples.py                # 24 complexes
    python baseline_analysis/select_samples.py --n 40         # more
    python baseline_analysis/select_samples.py --out-dir <bundle>
"""
import argparse, os, sys, time
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):   # repo root (config) + this dir
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROOT = os.environ.get("GENPLIP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
META = os.path.join(ROOT, "metadata.tsv")
S1B  = os.path.join(ROOT, "stage1b_large_scale_plip_mask_summary.csv")
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# ---- blocklist of uninteresting PDB chemical-component IDs -------------------
IONS = set("""NA K MG CA MN ZN FE FE2 CO 3CO NI CU CU1 CU3 CD HG AU AU3 AG PT PT4 PD RU RH IR MO 4MO 6MO W V CR PB
BA SR CS RB LI TL GA IN SN SB BI AL BE VO4 MO4 WO4 MOO 2MO PER OS YT3 YB YB2 GD GD3 EU EU3 TB TB9 SM SM3 LA CE PR
ND HO HO3 ER3 TM LU DY CL BR IOD IDO I F FLO BRO OH OHX O O2 OXY HOH DOD H2S SO4 SO3 SO2 SUL S PO4 PO3 2HP PI IPS
NO3 NO2 NO N2O CO3 CO2 CN CNN SCN N3 AZI NH4 NH2 NH3 FES SF4 F3S FS4 CFM CFN CLF ICS NFS CUA CUB CUZ HDD F4S OEC
OEX SF3 CN1 BF2 BO4 BO3 B B4O BEF ALF AF3 ALF4 BEF3 MGF 3PO 2PO PPV DPO 6WO""".split())
BUFFERS = set("""EDO PEG PGE PG4 PG0 PG5 PG6 1PE 2PE 7PE 12P 15P P33 XPE N8E C8E JEF PGO PGR DEG P4G P6G 2PO DIO DOX
MPD MRD MPO BU3 BU1 BU2 BOM IPA IPH 2PN PDO MOH EOH EGL PGA OHE ACT ACY ACN ACE EEE DMS DMF DME DTT DTV DTU DTD TCE
BME MB2 MBO BEZ BEN PHN NHE CAC POP PPI TRS TAM TAU BTB EPE HEZ MES PIN PIP CIT FLC TLA TAR MLA MLI MAE SIN GAI SRT
FMT FOR GOL BCN ETX ETA EDT EGO PE3 PE4 PE5 PE8 OGA OLC LDA SDS LMT LMN DDQ BOG BNG SOG C10 D10 MYS F09 KEN HP6 CXE
C14 BAM D12 UMQ TWT TRT HTG HTO HEX OCT DKA MC3 3PE PX4 PEF PEE PEK PGV PGW PCW LHG PT5 DGA HXA CE9 P15 PL9 SPD SPM
SPK PUT 1BO B3P B7N 144 IMD IMI 2IM CCN THJ PYR GLV AKG OAA MLT FUM PGF DR6 GLC""".split())
SUGARS = set("""NAG NDG NGA A2G BGC GLC BMA MAN GAL GLA FUC FUL FCA FCB XYS XYP XYL RIP RIB ARA ARB AHR LAT LBT MAL
MLR SUC TRE CBI GLP G6P G1P F6P FBP BGP G6D G4D 16G SGN IDS BDP GCU ADA SIA SLB KDO KDN MMA GCS PA1 RAM RM4 TYV ABE
DDA MFU MFB MBG GYP GUP""".split())
COFACTORS = set("""FMN FAD FDA FMA RBF FNS 6FA FCG MGD 2MD MSS MOS MTE MTV PCD NAD NAI NAJ NAH NAP NDP NAX NDO CND
NDC ODP TAP TXD TXP SND DND SAM SAH SFG SMM 5AD MTA COA ACO COO CMC MLC HMG CAA COS COZ COF SCA DCC MCA BCO BYC NMN
NMD BTN BTI DTB BIO TPP TDP TD6 TD8 TD9 TPW THW TZD 2TP PLP PMP PLR LLP PDP P5P NPL PXG PXL HEM HEA HEB HEC HEO HDD
HAS HIF VER DHE 1FH 2FH MH0 SRM HNI HEV HE5 HDM HP5 COH HES HFM HEG 1CP CP3 MP1 HCO B12 B1M CNC COB CO5 CBY BLA BLV
BPB BPH BPD PEB PUB CYC DBV F43 COM MQ7 MQ8 MQ9 MQE UQ1 UQ2 UQ5 UQ6 UQ7 UQ8 UQ9 U10 UQ10 PQN PQQ TPQ TRQ CLA CL0
CHL BCL BCR LUT XAT NEX ZEX DGD GSH GDS GTS GBX GSF GSO GDN GTB GSN ATP ADP AMP ANP ACP APC AGS ADX APR A2P AP5 ADN
GTP GDP GNP GSP GCP 5GP G2P GTN GNH 2GP 3GP GMP CTP CDP C5P CMP C2P UTP UDP UMP U5P UPG UFP TTP DTP TYD TMP THM 5MU
DUT DUD DUP DGT DGI DCT DCP DAT DA DC DG DT DU DI A C G U I N PSU 5MC 1MA 7MG OMG OMC H2U 4SU THP IMP XMP ITP GSU
5BU 2MG M2G AKG NADH FDX""".split())
AMINO = set("""ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE MLY M3L SEP TPO
PTR CSO CSD CSS CSX OCS CME CSW KCX PCA ABU ORN DAL DAR DSN DTH DPR DTR DTY DVA DLE DIL DGL DAS MED AIB NLE HYP SAR
GGL CGU TYS SNN DHA NME MAA HIC OMT ACE NH2""".split())
LIPIDS = set("""PLM MYR OLA STE PEE PGV PCW LHG PT5 DGA D10 HXA CE9 LOP LPP LP3 1PG PX2 17F 2DP 3PH DLP DPG DPP 6PL
PA0 POV PSF SPH CLR CHD CHS Y01 HC3 HCD OLC OLB LI1 UND EPH PEV PCF SGM""".split())
BLOCK = IONS | BUFFERS | SUGARS | COFACTORS | AMINO | LIPIDS

EMAILED_TOP20 = {
    "Unknown": 14459, "P00918": 1112, "P29476": 1095, "D0VWR1": 949, "Q8DIQ1": 831,
    "P0DTD1": 799, "Q8DIF8": 751, "P00415": 750, "P02945": 743, "P0A405": 724,
    "P56817": 718, "Q7NDN8": 707, "P0A407": 675, "D0VWR7": 627, "P31224": 569,
    "P00396": 539, "P03372": 532, "P29274": 526, "P19491": 494, "P02768": 466,
}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Select interesting complexes for the baseline.")
    p.add_argument("--n", type=int, default=None,
                   help="Complexes to select (default config.BASELINE_N_COMPLEXES).")
    p.add_argument("--per-ligand", type=int, default=1, help="Max complexes per ligand.")
    p.add_argument("--per-target", type=int, default=2, help="Max complexes per UniProt target.")
    p.add_argument("--out-dir", default=None,
                   help="Also copy selection.csv/candidates_all.csv/overview.txt into this bundle.")
    return p.parse_args(argv)


if __name__ == "__main__":
    ARGS = _parse_args()
    log("read metadata")
    md = pd.read_csv(META, sep="\t", dtype=str)
    for c in md.columns:
        if c.endswith("_count") or c in ("fda_status", "total_complex_interactions"):
            md[c] = pd.to_numeric(md[c], errors="coerce")
    p = md["id"].str.split(":", expand=True)
    md["pdb_id"], md["resname"], md["chain"], md["resseq"] = p[0].str.lower(), p[1], p[2], p[3]
    md["lig_global_count"] = md["resname"].map(md["resname"].value_counts())
    md["target_measurements"] = md["UniProt ID"].map(md["UniProt ID"].value_counts())

    # ---- corpus overview ----
    with open(os.path.join(OUT, "overview.txt"), "w", encoding="utf-8") as f:
        f.write(f"metadata binding sites : {len(md)}\n")
        f.write(f"unique PDB / lig / uniprot : {md.pdb_id.nunique()} / {md.resname.nunique()} / {md['UniProt ID'].nunique()}\n\n")
        f.write("fda_status counts:\n" + md.fda_status.value_counts(dropna=False).sort_index().to_string() + "\n\n")
        f.write("top 20 ligands (global freq):\n" + md.resname.value_counts().head(20).to_string() + "\n\n")
        f.write("top 20 targets (measurement count on THIS metadata.tsv):\n"
                + md["UniProt ID"].value_counts().head(20).to_string() + "\n\n")
        # Both rankings side by side: the emailed list cannot be reproduced from
        # this file (see the module docstring), so the difference is printed
        # rather than quietly resolved in favour of one of them.
        vc = md["UniProt ID"].value_counts()
        f.write("emailed top-20 vs THIS file (rows here, 0 = absent):\n")
        f.write(f"  {'UniProt':<10}{'emailed':>9}{'here':>8}\n")
        for k, v in EMAILED_TOP20.items():
            f.write(f"  {k:<10}{v:>9}{int(vc.get(k, 0)):>8}\n")
        missing = [k for k in EMAILED_TOP20 if k not in vc.index]
        f.write(f"  -> {len(missing)} of the emailed top-20 are absent from this file: "
                f"{', '.join(missing) if missing else 'none'}\n")

    # ---- metadata-only exclusion funnel ----
    F = {
        "known_target"        : md["UniProt ID"].notna() & (md["UniProt ID"].str.lower() != "unknown"),
        "not_blocklisted"     : ~md.resname.isin(BLOCK),
        "no_metal_interaction": md.metal_count.fillna(0) == 0,
        "interactions_ge5"    : md.total_complex_interactions.fillna(0) >= 5,
        "real_pose"           : (md.hbonds_count.fillna(0) >= 1) & (md.hydrophobics_count.fillna(0) >= 3),
        "not_ubiquitous_le120": md.lig_global_count <= 120,
        "fda_1to4"            : md.fda_status.isin([1, 2, 3, 4]),
    }
    keep = pd.Series(True, index=md.index); funnel = {}
    for k, m in F.items():
        m = m.fillna(False); funnel[k] = int((keep & ~m).sum()); keep &= m
    cand = md[keep].copy()
    log(f"after metadata filters: {len(cand)} rows, {cand['UniProt ID'].nunique()} targets, {cand.resname.nunique()} ligs")

    # ---- join SMILES from stage1b (PLIP-- corpus) for survivors only ----
    log("read stage1b summary (usecols)")
    want = set(zip(cand.pdb_id, cand.resname, cand.chain, cand.resseq))
    it = pd.read_csv(S1B, dtype=str, usecols=["pdb_id","resname","chain","resseq","status","smiles",
                     "masked_smiles","bpe_mask_count","masked_atom_indices"], chunksize=200_000)
    hits = []
    for ch in it:
        ch = ch[ch.status == "ok"]
        ch["pdb_id"] = ch.pdb_id.str.lower()
        for c in ("resname","chain","resseq"): ch[c] = ch[c].astype(str)
        key = list(zip(ch.pdb_id, ch.resname, ch.chain, ch.resseq))
        ch = ch[[k in want for k in key]]
        if len(ch): hits.append(ch)
    s1b = pd.concat(hits).drop_duplicates(["pdb_id","resname","chain","resseq"]) if hits else pd.DataFrame()
    log(f"stage1b matched rows: {len(s1b)}")
    s1b = s1b.set_index(["pdb_id","resname","chain","resseq"])
    cand = cand.set_index(["pdb_id","resname","chain","resseq"])
    cand["smiles"] = s1b["smiles"]
    cand["plip_neg_masked_smiles"] = s1b["masked_smiles"]
    cand["plip_neg_pool_bpe"] = pd.to_numeric(s1b["bpe_mask_count"], errors="coerce")
    cand["plip_neg_pool_atoms"] = s1b["masked_atom_indices"]
    cand = cand.reset_index()
    cand = cand[cand.smiles.notna()].copy()
    log(f"with stage1b SMILES: {len(cand)}")

    # ---- RDKit props (survivors only) ----
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Crippen, rdMolDescriptors
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    ORG = {"C","N","O","S","P","F","Cl","Br","I","H"}
    def props(smi):
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is None: return {}
        try: q = round(QED.qed(m), 3)
        except Exception: q = None
        return dict(heavy=m.GetNumHeavyAtoms(), mw=round(Descriptors.MolWt(m), 1),
            logp=round(Crippen.MolLogP(m), 2), hbd=rdMolDescriptors.CalcNumHBD(m),
            hba=rdMolDescriptors.CalcNumHBA(m), rotb=rdMolDescriptors.CalcNumRotatableBonds(m),
            rings=rdMolDescriptors.CalcNumRings(m), arom=rdMolDescriptors.CalcNumAromaticRings(m),
            qed=q, fsp3=round(rdMolDescriptors.CalcFractionCSP3(m), 3),
            nonorg=any(a.GetSymbol() not in ORG for a in m.GetAtoms()),
            charge=Chem.GetFormalCharge(m))
    log("rdkit props")
    pr = cand.smiles.map(props)
    for k in ["heavy","mw","logp","hbd","hba","rotb","rings","arom","qed","fsp3","nonorg","charge"]:
        cand[k] = pr.map(lambda d, k=k: d.get(k) if isinstance(d, dict) else None)

    G = {
        "parsed"       : cand.heavy.notna(),
        "druglike_size": cand.heavy.between(14, 55),
        "mw_window"    : cand.mw.between(180, 700),
        "organic_only" : cand.nonorg != True,
        "low_charge"   : cand.charge.abs() <= 1,
        "has_ring"     : cand.rings.fillna(0) >= 1,
    }
    keep2 = pd.Series(True, index=cand.index)
    for k, m in G.items():
        m = m.fillna(False); funnel[k] = int((keep2 & ~m).sum()); keep2 &= m
    cand = cand[keep2].copy()
    log(f"final candidate pool: {len(cand)}")

    cand = cand.sort_values(["fda_status","target_measurements","total_complex_interactions"],
                            ascending=[False, False, False])
    keepcols = ["id","pdb_id","resname","chain","resseq","UniProt ID","fda_status","target_measurements",
        "lig_global_count","total_complex_interactions","hbonds_count","hydrophobics_count","pistacks_count",
        "pications_count","sbridges_count","halogens_count","wbridge_count","heavy","mw","logp","hbd","hba",
        "rotb","rings","arom","qed","fsp3","charge","plip_neg_pool_bpe","smiles"]
    cand[keepcols].to_csv(os.path.join(OUT, "candidates_all.csv"), index=False)

    with open(os.path.join(OUT, "overview.txt"), "a", encoding="utf-8") as f:
        f.write("\n\nEXCLUSION FUNNEL (rows dropped by each filter, in order):\n")
        for k, v in funnel.items(): f.write(f"  {k:<22} -{v}\n")
        f.write(f"\nfinal candidate pool: {len(cand)}\n")
        f.write(f"  unique targets: {cand['UniProt ID'].nunique()}\n  unique ligands: {cand.resname.nunique()}\n")
        f.write(f"  FDA==4: {(cand.fda_status==4).sum()}   FDA 1/2/3: {cand.fda_status.isin([1,2,3]).sum()}\n\n")
        f.write("candidate pool - targets by #candidate complexes (top 20):\n")
        f.write(cand['UniProt ID'].value_counts().head(20).to_string() + "\n\n")
        f.write("candidate pool - ligands by #candidate complexes (top 20):\n")
        f.write(cand.resname.value_counts().head(20).to_string() + "\n")

    UNIPROT_NAME = {
        "P03372":"Estrogen receptor alpha","P10275":"Androgen receptor","P37231":"PPAR-gamma",
        "P19793":"Retinoid X receptor alpha","P04150":"Glucocorticoid receptor","P06401":"Progesterone receptor",
        "Q05769":"COX-2 (PTGS2)","P05979":"COX-1 (PTGS1)","P00519":"ABL1 kinase","P00533":"EGFR kinase",
        "P24941":"CDK2","Q16539":"p38-alpha MAPK","P15056":"BRAF kinase","P11362":"FGFR1 kinase",
        "O60885":"BRD4 bromodomain","P56817":"BACE-1","P00918":"Carbonic anhydrase II",
        "P04058":"Acetylcholinesterase (Torpedo)","P22303":"Acetylcholinesterase (human)",
        "P42330":"Aldo-keto reductase 1C3","Q9UNA1":"AKR1C","P08235":"Mineralocorticoid receptor",
        "P11838":"T. brucei ... (check)","P08246":"Neutrophil elastase","P00734":"Thrombin",
        "P00742":"Factor Xa","P07711":"Cathepsin L","P25774":"Cathepsin S","P43235":"Cathepsin K",
        "Q9Y233":"PDE10A","O76074":"PDE5A","P27487":"DPP-4","P00520":"c-Src / Abl (murine)",
        "P00523":"c-Src (avian)","P43405":"SYK kinase","Q05397":"FAK",
    }
    LIG_NAME = {
        "RAL":"Raloxifene","DES":"Diethylstilbestrol","OHT":"4-Hydroxytamoxifen","EST":"Estradiol",
        "DHT":"Dihydrotestosterone","TES":"Testosterone","STR":"Progesterone-ish steroid","9CR":"9-cis retinoic acid",
        "REA":"Retinoic acid","STI":"Imatinib","IRE":"Gefitinib (Iressa)","FMM":"Lapatinib","BAX":"BIRB-796 / Doramapimod",
        "CEL":"Celecoxib","LUR":"Lumiracoxib","IMN":"Indomethacin","DIF":"Diclofenac","FLF":"Flufenamic acid",
        "TLS":"Telmisartan","ROC":"Ritonavir","TCL":"Triclosan","KAI":"Kainic acid","ZMR":"Zanamivir",
        "QUE":"Quercetin","STL":"Resveratrol / stilbene","GNT":"Galanthamine-like","NLB":"Nilotinib",
        "VIB":"Vitamin/flavin-like (check)","0LI":"Vismodegib-ish (check)","P06":"BRAF inhibitor (check)",
        "032":"BRAF inhibitor (check)","08J":"BRD4 fragment (check)","LOC":"BRD4 ligand (check)",
        "B49":"CDK2 inhibitor","ID8":"AKR1C3 inhibitor","CP0":"AChE inhibitor","9RA":"retinoid",
    }
    cand["target_name"] = cand["UniProt ID"].map(UNIPROT_NAME).fillna("")
    cand["lig_name"] = cand["resname"].map(LIG_NAME).fillna("")
    keepcols += ["target_name", "lig_name"]
    cand[keepcols].to_csv(os.path.join(OUT, "candidates_all.csv"), index=False)

    # ---- curated pick: <=1 per ligand, <=2 per target, FDA4 first ----
    import config as _cfg
    TARGET_N = ARGS.n or getattr(_cfg, "BASELINE_N_COMPLEXES", 24)
    PER_LIG, PER_TGT = ARGS.per_ligand, ARGS.per_target
    def pick(df, n, sl, st):
        df = df.copy()
        df["pref"] = (df.target_measurements.rank(pct=True) * .45
                      + df.total_complex_interactions.rank(pct=True) * .25
                      + df.qed.fillna(0).rank(pct=True) * .15
                      + df.arom.fillna(0).rank(pct=True) * .15)
        df = df.sort_values(["fda_status","pref"], ascending=[False, False])
        out = []
        for _, r in df.iterrows():
            if len(out) >= n: break
            if sl.get(r.resname, 0) >= PER_LIG or st.get(r["UniProt ID"], 0) >= PER_TGT: continue
            out.append(r); sl[r.resname] = sl.get(r.resname, 0)+1; st[r["UniProt ID"]] = st.get(r["UniProt ID"], 0)+1
        return out
    sl, st = {}, {}
    rows = pick(cand[cand.fda_status == 4], TARGET_N, sl, st)
    if len(rows) < TARGET_N:
        rows += pick(cand[cand.fda_status.isin([1,2,3])], TARGET_N - len(rows), sl, st)
    sel = pd.DataFrame(rows)[keepcols]
    sel.to_csv(os.path.join(OUT, "selection.csv"), index=False)
    log(f"selection: {len(sel)}")

    # The later steps read the bundle, not this folder, so mirror the three
    # artifacts there when --out-dir is given.
    if ARGS.out_dir:
        import shutil
        os.makedirs(ARGS.out_dir, exist_ok=True)
        for _name in ("selection.csv", "candidates_all.csv", "overview.txt"):
            shutil.copyfile(os.path.join(OUT, _name), os.path.join(ARGS.out_dir, _name))
        log(f"copied selection.csv, candidates_all.csv, overview.txt -> {ARGS.out_dir}")
    print("\n" + open(os.path.join(OUT, "overview.txt"), encoding="utf-8").read())
    print("\n=== SELECTION ===")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(sel[["id","UniProt ID","target_name","resname","lig_name","target_measurements",
                   "total_complex_interactions","heavy","mw","qed","plip_neg_pool_bpe"]].to_string(index=False))
