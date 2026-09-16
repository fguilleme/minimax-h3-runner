# MiniMax H3 — Tête-à-tête headless (Français)

> **Versions** : [English README](README.md) — [README français](README_FR.md)

Runner local MiniMax H3 en trois processus indépendants, sans le serveur HTTP ComfyUI, ni Manager, ni interface graphique.

Il réutilise les chargeurs et noyaux Python de ComfyUI parce que le DiT W4A8 et ClipProj sont dans leurs formats optimisés. Les poids ne sont ni copiés ni téléchargés par ce projet — vous devez les récupérer à l'avance (étape 2 de la roadmap) .

## Évolution des segments audio longs

Un seul `.mp4` généré ; chaque **segment** d'une longue vidéo est écrit dans son propre dossier de l'exécution, avec exactement les mêmes données d'artefacts. Aucun fichier n'est réutilisé d'une exécution à l'autre : c'est intentionnel pour ne jamais mélanger des segments de vidéos différentes lors d'une concaténation ultérieure.

## Roadmap : du `git clone` à la génération vidéo

### 1. Prérequis

- Linux, `git`, `ffmpeg`, Python ≥ 3.10, environ 80 GiB de RAM, un GPU AMD ROCm (Radeon 8060S ou équivalent gfx11xx — ROCm gfx1151 validé).
- La source ComfyUI déjà installée (utilisée comme **chargement** ; aucune nouvelle instance ComfyUI n'est créée).
- ROCm ≥ 7.2 ; PyTorch ROCm ≥ 2.6.

### 2. Télécharger les modèles

Placez les six fichiers `.safetensors` dans `{comfy_root}/models/`. L'arborescence standard de ComfyUI attend `unet/`, `vae/`, `text_encoders/` et `lora/` :

| Fichier | Chemin estimé | Rôle |
|---|---|---|
| `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | `models/unet/…` | UNet + VAE vidéo (W4A8, pesage mixte) |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/…` | VAE de la vidéo (FP16) |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/…` | VAE de l'audio (FP32) |
| `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders/…` | Encodeur de texte |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `models/lora/…` | LoRA à 8 étapes |

Projection Clip :

- `mmh3-4b-ClipProj-v3.1.safetensors` dans `custom_nodes/ComfyUI-ClipProj/models/`.

> Vérifiez le nom réel du dépôt (source MiniMax ou le hub HF de référence) ; les noms de fichiers ci-dessus provient de `config.json`.

### 3. Installer ComfyUI et l'environnement virtuel

```bash
COMFY="${COMFY:-$HOME/comfy/ComfyUI}"
command -v git >/dev/null || (sudo pacman -S git)   # ou apt-get install -y git
git clone --depth 1 <URL_DU_REPO_COMFYUI> "$COMFY"
cd "$COMFY" && ./venv/bin/python -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
git submodule update --init --recursive custom_nodes/ComfyUI-ClipProj
```

### 4. Télécharger le runner et l'ouvrir

```bash
cd "$ROOT_PROJECT"
git clone <URL_DU_REPO_MINIMAX> .
source ./h3runner/venv/bin/activate && cd src && pip install -e .
```

### 5. Valider l'environnement (Run.sh)

Le script de validation (`--steps`, --fps` de la config). Il **ne lance pas** `ComfyUI/Manager`.

```bash
./run.sh          # depuis le dépôt du runner
git diff        # si rien ne ressort, la validation est stable
git fetch --all && git pull
python -m unittest test_*.py -v --force
```

### 6. Générer une courte vidéo (Run.sh)

```bash
./run.sh \
  --config "${RUN_PROJECT}/config.json" \
  --work-dir runs/mon-cerf \
  --output output/minimax-h3-mon-cerf-10fps.mp4
```

### 7. Générer une longue vidéo (Run-long.sh)

`--first-frame`, `--output` ; `--duration` en secondes :

```bash
./run-long.sh \
  --config config.json \
  --duration 30 \
  --output output/minimax-h3-30s.mp4
```

### 8. Suivi des problèmes

| Symptôme | Cause commune / correctif |
|---|---|
| `ComfyUI not found` | `comfy_root` mal nommé ; vérifier le path dans `config.json`. |
| Échec des import de `h3runner` | Renommer l'environnement virtuel `venv`. |
| `CUDA_OUT_OF_MEMORY`, VRAM saturée | `--lowvram` / `--novram` ; config GPU plus conservatrice. |
| VAE audio/vidéo manquant | Vérifier que les `.safetensors` sont où l'arborescence les attend. |
| `ClipProj` introuvable | Sous-modules : `git submodule --init init`. |

## Configuration

| Paramètre | Valeur par défaut | Rôle |
|---|---|---|
| `length` | 50 | Nombre d'images de base (≈ `length/2+1` après post-traitement). |
| `duration` / `output_fps` | — | Durée finale de la vidéo, fps. |
| `steps` / `sampler` | 8 / `res_multistep` | Qualité vs vitesse ; `steps=8`, `simple`. |
| `seed` | 13092029 | Graine (répétable). |
| `audio_mode` | `loop` | Conserve la piste native si `first_frame` non fourni. |
| `unet_name` / `video_vae_name` / `audio_vae_name` | — | Poids des VAE vidéo/audio. |

## Tests unitaires

```bash
source .venv/bin/activate && cd src && pip install -e .
```

Un seul `.mp4` est généré ; chaque segment d'une vidéo longue est écrit dans son propre dossier, avec exactement les mêmes artefacts. Il ne réutilise aucun fichier d'une exécution à l'autre — intentionnel pour éviter de mélanger des segments entre vidéos quand on concatène.

### Évolution des segments audio longs

Le runner génère une seule sortie `.mp4` ; chaque segment d'une longue vidéo est écrit dans son propre dossier de run avec les mêmes données d'artefacts. Il ne réutilise **jamais** un fichier d'une exécution à l'autre — intentionnel pour ne jamais mélanger des segments entre vidéos lors d'une concaténation.

## Utilisations courantes

| Usage | Commande | Exemple de sortie |
|---|---|---|
| Courte (≤ 8 s) | `h3runner.run --config …` | `output/*.mp4` |
| Longue (≥ 15 s) | `h3runner.longrun --steps 12 --fps 12 …` | `runs/*` + `output/*.mp4` |

---

**Retour vers le README anglais :** [English README](README.md)
