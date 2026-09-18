# MiniMax H3 Headless Runner — Français

[English README](README.md) — **README français**

Runner local MiniMax H3 en trois processus indépendants : encodage,
diffusion puis décodage/encapsulation MP4. Il réutilise les chargeurs et
noyaux Python de ComfyUI, mais ne lance ni l'application ComfyUI, ni son
serveur HTTP, ni Manager.

Les poids ne sont pas téléchargés par ce projet. Ils doivent déjà être
présents dans l'installation ComfyUI indiquée par `comfy_root`.

## Prérequis

- Linux, Python >= 3.10, `ffmpeg` et Git ;
- GPU AMD ROCm gfx11xx validé, notamment Radeon 8060S/gfx1151 ;
- PyTorch ROCm >= 2.6 et ROCm >= 7.2 ;
- ComfyUI installé, avec son environnement virtuel `.venv` ;
- les modèles MiniMax H3 et ClipProj aux emplacements décrits dans le
  [README anglais](README.md#reused-models).

Depuis le dépôt :

```bash
cd /home/francois/projects/minimax-h3-runner
PYTHONPATH=src /home/francois/comfy/ComfyUI/.venv/bin/python \
  -m unittest discover -s tests -v
```

## Génération directe

Vidéo courte avec le script fourni :

```bash
./run.sh
```

Image-to-video avec une première image :

```bash
./run.sh \
  --first-frame input/fox-first.png \
  --work-dir runs/fox-i2v \
  --output output/fox-i2v.mp4
```

Quand `--first-frame` ou `--last-frame` est fourni, le runner dérive
maintenant automatiquement `width` et `height` depuis le ratio de l’image,
en respectant les multiples de 32 et le plafond spatial H3. Le même canvas
est réutilisé pour tous les segments. Le fichier de configuration effectif
est conservé dans `effective-config.json` lorsque les dimensions changent.

Vidéo longue par segments chaînés :

```bash
./run-long.sh \
  --config config.json \
  --duration 30 \
  --work-dir runs/fox-long-30s \
  --output output/fox-long-30s.mp4 \
  --audio-policy first
```

Les segments sont générés avec continuité : la dernière image décodée d'un
segment devient la première image du suivant. Les exécutions sont
reprises automatiquement si leurs artefacts sont encore valides ; utiliser
`--force` pour tout recalculer.

## Interface Gradio

Le dépôt fournit une interface web Gradio qui appelle directement le runner
headless ; le serveur HTTP MiniMax n'est pas nécessaire :

```bash
./run-gradio.sh
```

Ouvrir `http://127.0.0.1:7860`. L'interface accepte une image initiale
facultative, un prompt simple ou une timeline comme
`2:the woman walks|3:the woman looks left and right slowly`, puis affiche
automatiquement la vidéo terminée.

Pour une écoute LAN, l'authentification est obligatoire :

```bash
H3_GRADIO_HOST=0.0.0.0 \
H3_GRADIO_USER=francois \
H3_GRADIO_PASSWORD='choisir-un-mot-de-passe' \
./run-gradio.sh
```


Le serveur accepte un JSON, place le travail en arrière-plan et renvoie
immédiatement un identifiant de job. Il lance `h3runner.longrun` dans le
même environnement Python que ComfyUI.

Démarrage depuis le répertoire du projet :

```bash
cd /home/francois/projects/minimax-h3-runner
PYTHONPATH=src /home/francois/comfy/ComfyUI/.venv/bin/python \
  -m h3runner.server \
  --host 127.0.0.1 \
  --port 8988 \
  --comfy-root /home/francois/comfy/ComfyUI \
  --runs-dir runs \
  --log-dir log \
  --pid runs/server.pid
```

Vérifier que le serveur répond :

```bash
curl -sS http://127.0.0.1:8988/healthz
# {"ok": true}
```

### Exemple 1 — prompt simple, 5 secondes, 10 fps

```bash
curl -sS -X POST http://127.0.0.1:8988/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Un renard roux marche dans une clairière ensoleillée",
    "duration": 5,
    "fps": 10,
    "width": 384,
    "height": 224
  }'
```

La réponse `201` contient notamment `job.id` :

```json
{"job":{"id":"abcd1234","state":"running"},"message":"job queued"}
```

### Exemple 2 — une image et un prompt, 5 secondes, 10 fps

`first_frame` accepte un chemin absolu ou un chemin relatif au répertoire
depuis lequel le serveur a été démarré :

```bash
curl -sS -X POST http://127.0.0.1:8988/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Le renard tourne lentement la tête pendant que les feuilles bougent",
    "first_frame": "input/fox-first.png",
    "duration": 5,
    "fps": 10,
    "width": 384,
    "height": 224
  }'
```

### Exemple 3 — longue durée

Cet exemple demande 30 secondes et utilise une image initiale. Le serveur
chaîne autant de segments H3 que nécessaire :

```bash
curl -sS -X POST http://127.0.0.1:8988/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Une danseuse exécute une chorégraphie énergique et élégante en studio",
    "first_frame": "input/dancer-source.jpg",
    "duration": 30,
    "fps": 10,
    "width": 256,
    "height": 384,
    "audio": "first"
  }'
```

### Suivre ou annuler un job

Remplacer `JOB_ID` par l'identifiant reçu lors du POST :

```bash
curl -sS 'http://127.0.0.1:8988/status?job=JOB_ID'
```

États possibles : `running`, `done` ou `failed`. Le champ `output` donne
le chemin du MP4 et `log_path` celui du journal du job.

### Récupérer la vidéo

Quand le job est terminé, télécharger directement le MP4 avec :

```bash
curl -fL -o minimax-h3-JOB_ID.mp4 \
  'http://127.0.0.1:8988/video?job=JOB_ID'
```

`GET /video?job=JOB_ID` renvoie `video/mp4` avec `Content-Disposition`. Il
renvoie `409` tant que le job n'est pas terminé et `404` si le job ou le
fichier n'existe pas.

Pour annuler un job en cours :

```bash
curl -sS -X DELETE 'http://127.0.0.1:8988/status?job=JOB_ID'
```

## Post-traitement vidéo IA : Real-ESRGAN et RIFE

Pour traiter une vidéo arbitraire, utiliser [Video2X](https://github.com/k4yt3x/video2x),
qui fournit des backends Vulkan pour [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
(super-résolution) et [RIFE](https://github.com/hzwer/ECCV2022-RIFE)
(interpolation d'images). Ce pipeline est indépendant de la génération MiniMax H3.
L'AppImage Linux a été validée sur une Radeon 8060S AMD avec RADV/Vulkan.

Exemple : agrandir une vidéo portrait 2×, puis passer de 10 à 30 fps :

```bash
mkdir -p tools/video2x
curl -fL -o tools/video2x/Video2X-x86_64.AppImage \\
  https://github.com/k4yt3x/video2x/releases/download/6.4.0/Video2X-x86_64.AppImage
chmod +x tools/video2x/Video2X-x86_64.AppImage

# Super-résolution 2×
./tools/video2x/Video2X-x86_64.AppImage --no-progress \\
  -i runs/input.mp4 -o runs/input-realesrgan-2x.mp4 \\
  -p realesrgan -s 2 --realesrgan-model realesr-animevideov3

# RIFE ×3 : 10 fps -> 30 fps
./tools/video2x/Video2X-x86_64.AppImage --no-progress \\
  -i runs/input-realesrgan-2x.mp4 -o runs/input-realesrgan-2x-rife-30fps.mp4 \\
  -p rife -m 3 --rife-model rife-v4.6
```

Les dimensions de génération H3 doivent de préférence être des multiples de 32.
Real-ESRGAN reconstruit des détails plausibles mais ne peut pas récupérer les
informations absentes de la source. RIFE synthétise des images intermédiaires et
peut produire des artefacts lors des mouvements rapides.

## Paramètres JSON

- `prompt` : texte de génération ;
- `first_frame` ou `firstframe` : image initiale facultative ;
- `duration` ou `secs` : durée finale en secondes ;
- `fps` ou `framerate` : fréquence de sortie ;
- `width`, `height` : dimensions, de préférence des multiples de 32 ;
- `steps`, `seed`, `sampler`, `scheduler` : paramètres de génération ;
- `audio` : `first` ou `segments` pour les longues vidéos.

Les prompts peuvent aussi être planifiés par segment avec la syntaxe
`durée:prompt|durée:prompt`. Chaque entrée produit un segment H3 séparé,
conserve la dernière image comme continuité et utilise son propre prompt. La
durée finale est la somme des entrées, puis la concaténation est ajustée à
cette durée :

```json
{
  "prompt": "2:the woman walks|3:the woman looks left and right slowly",
  "duration": 5,
  "fps": 10
}
```

Les durées sont exprimées en secondes. H3 aligne chaque segment sur sa grille
temporelle native ; la sortie finale est ensuite trimée à la somme demandée.

Les valeurs par défaut du serveur restent `704x480`, 10 fps, 6 secondes et
8 étapes lorsque le prompt n'utilise pas cette syntaxe.

## systemd

Le dépôt fournit l'unité prête à copier dans
`systemd/minimax-h3-server.service`. Installer et démarrer le service avec :

```bash
mkdir -p ~/.config/systemd/user
cp systemd/minimax-h3-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now minimax-h3-server.service
systemctl --user status minimax-h3-server.service
```

Le serveur ne doit pas être exposé sur le réseau sans authentification ou reverse
proxy sécurisé : l'interface actuelle est volontairement minimale et ne
fournit pas d'authentification HTTP.

## Limites connues

- Le serveur garde les jobs en mémoire ; après son redémarrage, les anciens
  identifiants ne sont plus interrogeables via `/status`.
- La continuité des longues vidéos est fondée sur la première/dernière image
  des segments ; elle ne garantit pas l'absence totale de dérive visuelle.
- Le son `audio: first` conserve la bande-son du premier segment et la boucle,
  plutôt que de composer une bande-son longue réellement continue.
