from pathlib import Path

# ====== A MODIFIER ======
DOSSIER = Path("/home/shakib/Desktop/S5/projet/subset50/images") 
DRY_RUN = False  

# Liste des fichiers à garder
KEEP = {
    "00017524_024.png",
    "00002457_003.png",
    "00021232_000.png",
    "00002569_000.png",
    "00018860_005.png",
    "00023696_001.png",
    "00012250_001.png",
    "00029552_000.png",
    "00018013_000.png",
    "00013917_025.png",
    "00020826_009.png",
    "00004360_023.png",
    "00014675_004.png",
    "00012620_005.png",
    "00007018_054.png",
    "00022716_000.png",
    "00017606_011.png",
    "00011702_075.png",
    "00011683_042.png",
    "00002294_000.png",
    "00006673_000.png",
    "00018260_004.png",
    "00019471_000.png",
    "00000569_004.png",
    "00018347_000.png",
    "00001386_000.png",
    "00020405_024.png",
    "00029713_001.png",
    "00015193_013.png",
    "00027264_000.png",
    "00011504_004.png",
    "00019475_000.png",
    "00007264_000.png",
    "00001555_002.png",
    "00015108_001.png",
    "00005061_000.png",
    "00020318_031.png",
    "00027066_011.png",
    "00009166_002.png",
    "00026366_001.png",
    "00011386_000.png",
    "00010384_035.png",
    "00013329_002.png",
    "00005140_007.png",
    "00023296_001.png",
    "00006416_000.png",
    "00015426_004.png",
    "00013128_021.png",
    "00009401_001.png",
    "00018284_000.png",
}

# Extensions d'images à considérer
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# ====== SCRIPT ======
if not DOSSIER.exists() or not DOSSIER.is_dir():
    raise SystemExit(f"Dossier introuvable: {DOSSIER}")

supprimes = 0
gardes = 0

for f in DOSSIER.iterdir():  # non-récursif
    if not f.is_file():
        continue
    if f.suffix.lower() not in IMG_EXT:
        continue

    if f.name in KEEP:
        gardes += 1
        continue

    # fichier image non présent dans KEEP => suppression
    if DRY_RUN:
        print(f"[DRY-RUN] Supprimerait: {f.name}")
    else:
        f.unlink()
        print(f"Supprimé: {f.name}")
    supprimes += 1

print(f"\nTerminé. Gardés: {gardes} | Supprimés: {supprimes} | Mode DRY_RUN={DRY_RUN}")
