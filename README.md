# MiniMax H3 headless runner

Runner local MiniMax H3 en trois processus indépendants, sans serveur HTTP ComfyUI, sans Manager et sans exécution du graphe UI.

Il réutilise les chargeurs et noyaux Python de ComfyUI parce que le DiT W4A8 et ClipProj sont dans leurs formats optimisés. Les poids ne sont ni copiés ni téléchargés.

## Pipeline

1. `h3runner.encode` charge Qwen3-VL-4B + ClipProj, encode éventuellement une image initiale et/ou finale avec le VAE vidéo, écrit le conditioning et le latent AV vide, puis termine.
2. `h3runner.denoise` charge seulement le DiT W4A8 + LoRA, exécute les huit étapes, écrit le latent AV généré, puis termine.
3. `h3runner.decode` charge les VAE, décode par tuiles temporelles et encode le MP4 avec `ffmpeg`.

La fin de chaque processus force la libération des modèles avant la phase suivante.

## Environnement validé

- Radeon 8060S `gfx1151`, 32 Gio VRAM réservée + 32 Gio RAM
- ROCm système 7.2.4 dans `/opt/rocm`
- PyTorch `2.11.0+rocm7.13.0`, runtime HIP `7.13.99004`
- `comfy-kitchen 0.2.34`
- Python : `/home/francois/comfy/ComfyUI/.venv/bin/python`
- Docker laissé actif pendant tout le smoke test
- `comfyui.service` arrêté

## Modèles réutilisés

Les chemins sont résolus sous `/home/francois/comfy/ComfyUI/models` :

- `diffusion_models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`
- `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
- `clip_projections/mmh3-4b-ClipProj-v3.1.safetensors`
- `vae/minimax_h3_video_vae_fp16.safetensors`
- `vae/minimax_h3_audio_vae_fp32.safetensors`
- `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

## Utilisation

```bash
cd /home/francois/projects/minimax-h3-runner
./run.sh
```

Pour arrêter explicitement le service ComfyUI avant l’exécution :

```bash
./run.sh --stop-comfy-service
```

Le service n’est volontairement pas redémarré. Docker n’est jamais arrêté.

Pour tout recalculer malgré les artefacts existants :

```bash
./run.sh --force
```

### Image-to-Video

Image initiale :

```bash
./run.sh \
  --first-frame /chemin/vers/depart.png \
  --work-dir runs/mon-i2v \
  --output output/mon-i2v.mp4
```

Image initiale et image finale :

```bash
./run.sh \
  --first-frame /chemin/vers/depart.png \
  --last-frame /chemin/vers/arrivee.png \
  --work-dir runs/mon-i2v-bornes \
  --output output/mon-i2v-bornes.mp4
```

Les images RGB, RGBA ou niveaux de gris sont normalisées en RGB flottant. MiniMax les redimensionne à la résolution configurée ; l’image initiale est ancrée à la première trame et l’image finale à la dernière. Les mêmes options peuvent être placées dans `config.json` avec `first_frame` et `last_frame`.

## Reprise

Les fichiers intermédiaires sont sous `runs/fox-56f/` :

- `conditioning.{json,safetensors}`
- `empty-latent.{json,safetensors}`
- `sampled-latent.{json,safetensors}`

`run.sh` reprend à la première phase manquante. Une empreinte couvre le fichier de configuration et le contenu des keyframes : modifier une image, le prompt ou un paramètre invalide automatiquement les artefacts et relance les trois phases. Les tenseurs sont enregistrés avec Safetensors ; le manifeste JSON préserve la structure des listes, tuples, dictionnaires et scalaires sans pickle.

## Cadence et audio

H3 génère nativement à 24 FPS et aligne le nombre d’images sur `17k + 5`. Une demande de 50 images produit donc 56 images. À 10 FPS, la sortie dure 5,6 secondes.

Le décodage étire l’audio natif avec `ffmpeg atempo` pour conserver la hauteur et obtenir exactement la même durée que la vidéo.

## Résultat du smoke test

- conditioning : `(1, 67, 5120)` FP32
- latent vidéo : `(1, 24, 17, 22, 38)` FP32
- latent audio : `(1, 32, 2, 93)` FP32
- diffusion : 8/8 étapes en environ 55 secondes
- sortie : 56 images, 608×352, 10 FPS, H.264
- audio : AAC stéréo 32 kHz, 5,6 secondes

## Tests

```bash
PYTHONPATH=src /home/francois/comfy/ComfyUI/.venv/bin/python \
  -m unittest discover -s tests -v
```

## Limites

- Le moteur réutilise des modules Python internes de ComfyUI, mais ne lance pas l’application ComfyUI.
- Une mise à jour de ComfyUI peut modifier les signatures internes ; les tests et le bootstrap doivent être rejoués après mise à jour.
- L’attention traite toujours toute la séquence AV pendant la diffusion. Le découpage temporel est utilisé au décodage VAE, pas dans le DiT.
- `flash-attn` CUDA n’est pas utilisé. PyTorch indique qu’AOTriton AMD expérimental peut être activé, mais il reste désactivé tant qu’il n’est pas validé sur `gfx1151`.
