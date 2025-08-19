from pathlib import Path
import re

ROOT = Path(__file__).parent
TEXT_EXT = {".py", ".yaml", ".yml", ".toml", ".ini", ".md", ".txt"}

def is_text(p: Path) -> bool:
    return p.suffix.lower() in TEXT_EXT

def patch_text(s: str) -> str:
    # 1) CSV
    s = s.replace("Courses_Completes_a1.csv", "Courses_Completes_a1.csv")
    # 2) colonnes spécifiques
    s = re.sub(r"\bsi_a2_dans_pronos\b", "si_a1_dans_pronos", s)
    # 3) cible a1 -> a1 (mots entiers)
    s = re.sub(r"\ba2\b", "a1", s)
    s = re.sub(r"\bA2\b", "A1", s)
    # 4) si config.yaml contient target_col
    s = s.replace("target_col: a1", "target_col: a1")
    return s

changed = []
for p in ROOT.rglob("*"):
    if p.is_file() and is_text(p):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        new = patch_text(txt)
        if new != txt:
            p.write_text(new, encoding="utf-8")
            changed.append(str(p.relative_to(ROOT)))

print("Fichiers modifiés :")
for c in changed:
    print(" -", c)

cfg = ROOT / "config.yaml"
if cfg.exists():
    t = cfg.read_text(encoding="utf-8", errors="ignore")
    if "Courses_Completes_a1.csv" not in t:
        t = re.sub(r"data_path:\s*.*", "data_path: data/raw/Courses_Completes_a1.csv", t)
        cfg.write_text(t, encoding="utf-8")
        print("config.yaml: data_path -> data/raw/Courses_Completes_a1.csv")

print("\nOK. Pense à supprimer models/* et data/processed/* puis relance l'entraînement.")
