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

## Serveur HTTP et exemples `curl`

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

Pour annuler un job en cours :

```bash
curl -sS -X DELETE 'http://127.0.0.1:8988/status?job=JOB_ID'
```

## Paramètres JSON

- `prompt` : texte de génération ;
- `first_frame` ou `firstframe` : image initiale facultative ;
- `duration` ou `secs` : durée finale en secondes ;
- `fps` ou `framerate` : fréquence de sortie ;
- `width`, `height` : dimensions, de préférence des multiples de 32 ;
- `steps`, `seed`, `sampler`, `scheduler` : paramètres de génération ;
- `audio` : `first` ou `segments` pour les longues vidéos.

Les valeurs par défaut du serveur sont `704x480`, 10 fps, 6 secondes et
8 étapes. Pour obtenir exactement 5 secondes à 10 fps, les exemples
explicitent `duration: 5` et `fps: 10`.

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
