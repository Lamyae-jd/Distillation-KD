# Guide d'implémentation PI-KD (Physics-Informed Knowledge Distillation)
## Compression de PhysFormer pour déploiement FPGA

---

## 0. Vue d'ensemble du pipeline

```
PhysFormer (enseignant, F1≈0.85)
        │
        ▼
  ┌─────────────────────────────┐
  │  Étape 1: Préparer les     │
  │  données et le teacher      │
  └─────────────┬───────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │  Étape 2: Concevoir le     │
  │  StudentNet                 │
  └─────────────┬───────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │  Étape 3: Entraînement     │
  │  PI-KD (curriculum 6 sem.) │
  └─────────────┬───────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │  Étape 4: Fine-tuning QAT  │
  │  (quantification FPGA)     │
  └─────────────┬───────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │  Étape 5: Export hls4ml    │
  │  et déploiement Xilinx     │
  └─────────────────────────────┘
```

Critère de succès global : **Student F1 ≥ 0.83**, dégradation post-QAT < 1%.

---

## 1. Structure de projet recommandée

```
pi-kd/
├── CLAUDE.md                  # Contexte pour Claude Code
├── configs/
│   └── train_config.yaml      # Tous les hyperparamètres centralisés
├── data/
│   ├── datamodule.py          # PyTorch Lightning DataModule
│   └── preprocessing.py       # Normalisation, augmentation
├── models/
│   ├── teacher.py             # Wrapper du PhysFormer pré-entraîné
│   ├── student.py             # Architecture StudentNet
│   └── layers.py              # Blocs réutilisables (DepthwiseSep, etc.)
├── losses/
│   ├── focal_loss.py          # Focal loss (déséquilibre 1:24)
│   ├── distillation_loss.py   # KL divergence teacher/student
│   └── physics_loss.py        # Contrainte Klein-Nishina
├── training/
│   ├── trainer.py             # Boucle d'entraînement PI-KD
│   └── curriculum.py          # Scheduler des lambdas
├── quantization/
│   ├── qat.py                 # Fine-tuning QAT PyTorch natif
│   └── export_hls4ml.py       # Conversion vers HLS C++
├── evaluation/
│   ├── metrics.py             # F1, CNR, SNR, MTF
│   └── compare.py             # Teacher vs Student vs Student-QAT
└── scripts/
    ├── train.py               # Point d'entrée entraînement
    ├── evaluate.py            # Point d'entrée évaluation
    └── export.py              # Point d'entrée export FPGA
```

Pourquoi cette structure : chaque composant est isolé et testable indépendamment.
Le fichier `configs/train_config.yaml` centralise TOUS les hyperparamètres pour la
reproductibilité (essentiel pour la thèse).

---

## 2. CLAUDE.md (à placer à la racine du projet)

Ce fichier donne le contexte à Claude Code. Contenu suggéré :

```markdown
# PI-KD: Physics-Informed Knowledge Distillation pour PhysFormer

## Contexte
Compression du modèle PhysFormer (correction ICS en ToF-CT) vers un
StudentNet léger pour déploiement FPGA via hls4ml.

## Conventions
- Python 3.10+, PyTorch 2.x
- Pas de classes abstraites inutiles, garder le code simple et lisible
- Docstrings en français pour les modules, en anglais pour les API PyTorch
- Tous les hyperparamètres dans configs/train_config.yaml
- Logger : TensorBoard (pas wandb)
- Seed fixe (42) partout pour reproductibilité

## Architecture StudentNet
- Convolutions depthwise separable uniquement
- Zéro couche d'attention (contrainte FPGA)
- Compatible QAT PyTorch natif → hls4ml

## Contraintes
- Déséquilibre de classes ~1:24 (1 pixel primaire pour 24 négatifs)
- Le teacher a F1≈0.85, le student doit atteindre F1≥0.83
- Dégradation post-QAT tolérée : <1% de F1
```

---

## 3. Étape 1 — Préparer les données et le teacher

### 3.1 Ce que tu dois comprendre

**Le teacher (PhysFormer)** est déjà entraîné. Tu n'y touches plus. Tu en as
besoin uniquement en mode `eval()` pour deux choses :
1. Générer les **soft labels** (logits avant argmax) sur tes données
2. Servir de référence de performance (F1=0.85)

**Décision clé : pré-calculer vs calculer à la volée**

Option A — Pré-calcul : tu passes toutes tes données dans le teacher UNE fois,
tu sauvegardes les logits sur disque, et pendant l'entraînement du student tu
charges les logits pré-calculés. Avantage : entraînement plus rapide, pas besoin
de garder le teacher en mémoire GPU. Inconvénient : si tu fais de l'augmentation
de données, les logits pré-calculés ne correspondent plus aux données augmentées.

Option B — À la volée : le teacher tourne en parallèle du student à chaque batch.
Avantage : compatible avec l'augmentation. Inconvénient : ~2x la mémoire GPU.

**Recommandation pour ton cas** : Option B (à la volée), parce que tu auras
probablement besoin d'augmentation pour compenser le déséquilibre de classes.

### 3.2 Gestion du déséquilibre 1:24

C'est un point critique. Avec ~1 pixel primaire pour 24 négatifs, un modèle naïf
qui prédit toujours "négatif" atteint déjà ~96% d'accuracy. C'est pourquoi on
utilise F1 (pas accuracy) et focal loss (pas cross-entropy).

Stratégies complémentaires à considérer :
- **Focal loss** (déjà prévue) : réduit le poids des exemples faciles
- **Oversampling** des patches contenant des pixels primaires
- **pos_weight adaptatif** : calculé dynamiquement sur chaque batch
- **Mixup/CutMix** sur les patches positifs

### 3.3 Ce qu'il faut implémenter

- Un `DataModule` qui charge tes données ToF-CT et les sert en batches
- Un wrapper autour du teacher qui le charge, le met en `eval()`, et
  désactive le calcul des gradients (`torch.no_grad()`)
- Le calcul de pos_weight sur ton jeu d'entraînement complet

### 3.4 APIs PyTorch clés à connaître

```
torch.no_grad()           # Contexte pour le forward du teacher
model.eval()              # Met le teacher en mode évaluation
torch.utils.data.Dataset  # Classe de base pour tes données
torch.utils.data.DataLoader  # Itérateur par batch
```

---

## 4. Étape 2 — Concevoir le StudentNet

### 4.1 Philosophie architecturale

Le StudentNet n'est PAS une version réduite du PhysFormer. C'est une architecture
DIFFÉRENTE, conçue dès le départ autour de deux contraintes matérielles :

1. **Pas d'attention** : le softmax nécessite des exponentielles et des
   normalisations, opérations coûteuses sur FPGA (pas de FPU)
2. **Convolutions depthwise separable** : une convolution standard NxN sur C
   canaux coûte N²·C² multiplications. Une depthwise separable la décompose en
   une convolution spatiale par canal (N²·C) suivie d'une convolution 1x1
   (C²), soit environ N² fois moins d'opérations

### 4.2 Blocs de construction

**Bloc DepthwiseSeparable** :
```
Input (C_in canaux)
    │
    ├─ Conv2d(groups=C_in, kernel=3x3)   ← une convolution par canal
    ├─ BatchNorm2d
    ├─ ReLU (ou ReLU6 pour quantification)
    │
    ├─ Conv2d(kernel=1x1, C_in → C_out)  ← mélange les canaux
    ├─ BatchNorm2d
    └─ ReLU
```

Pourquoi ReLU6 plutôt que ReLU : ReLU6 borne les activations à [0, 6], ce qui
facilite la quantification (la plage dynamique est connue à l'avance). C'est un
choix classique pour les réseaux destinés à être quantifiés.

**Pourquoi BatchNorm est important pour le QAT** : pendant le QAT, BatchNorm sera
fusionné avec la convolution précédente (Conv+BN → ConvBN). PyTorch le fait
automatiquement avec `torch.quantization.fuse_modules`. Après fusion, il n'y a
plus de BN séparé, ce qui réduit la latence sur FPGA.

### 4.3 Architecture globale suggérée

```
Input (patch ToF-CT)
    │
    ├─ Conv2d initiale (3x3, stride=1, peu de filtres : 16 ou 32)
    ├─ BN + ReLU6
    │
    ├─ DepthwiseSep Block × 2-3  (expansion progressive : 32→64→128)
    │   (avec résiduel si dimensions compatibles)
    │
    ├─ Global Average Pooling
    ├─ Linear (128 → nombre de classes)
    └─ Output logits
```

Garde l'architecture PETITE au début. Tu pourras toujours l'agrandir si le F1
est trop bas. Il est beaucoup plus difficile de réduire un modèle qui dépasse
les ressources FPGA.

### 4.4 Dimensionnement

Pour estimer si ton modèle tient sur le FPGA :
- Compte le nombre total de paramètres : `sum(p.numel() for p in model.parameters())`
- Compte le nombre de MAC (multiply-accumulate) : utilise `torchinfo` ou `ptflops`
- Règle empirique pour hls4ml : reste sous ~100K paramètres pour un premier essai

### 4.5 APIs PyTorch clés

```
torch.nn.Conv2d(groups=C)     # groups=C pour depthwise
torch.nn.Conv2d(kernel_size=1) # pointwise (1x1)
torch.nn.BatchNorm2d
torch.nn.ReLU6
torch.nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
torchinfo.summary(model, input_size=...)  # Résumé de l'architecture
```

### 4.6 Erreurs courantes à éviter

- Ne mets PAS de Dropout : inutile avec la distillation et problématique pour QAT
- Ne mets PAS de couches d'attention "simplifiées" : même du linear attention
  pose des problèmes d'export hls4ml
- Évite MaxPool : préfère stride=2 dans une convolution (plus FPGA-friendly)
- Pas de GroupNorm ou LayerNorm : hls4ml ne les supporte pas bien, reste sur BN

---

## 5. Étape 3 — Entraînement PI-KD

### 5.1 Les trois composantes de la perte

**L_total = L_task + λ_kd · L_KD + λ_phys · L_phys**

#### 5.1.1 L_task — Focal Loss

La focal loss modifie la cross-entropy en réduisant le poids des exemples bien
classifiés. Formule :

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

Où :
- p_t est la probabilité prédite pour la bonne classe
- γ (gamma) contrôle la modulation (typiquement γ=2)
- α est le facteur de pondération de classe (lié au déséquilibre)

Avec γ=0, c'est une cross-entropy standard. Plus γ est grand, plus les exemples
faciles sont down-pondérés.

**Valeurs initiales recommandées** : γ=2.0, α calculé à partir du ratio de classes.

PyTorch ne fournit pas de focal loss native. Tu dois l'implémenter toi-même.
C'est un bon exercice : c'est essentiellement `F.binary_cross_entropy_with_logits`
multiplié par `(1 - p_t)^γ`.

#### 5.1.2 L_KD — Divergence KL (distillation)

C'est le cœur de la distillation. On compare les distributions de probabilités
(soft labels) du teacher et du student.

Concept clé : **la température T**

Avant d'appliquer softmax, on divise les logits par T :

    soft_teacher = softmax(logits_teacher / T)
    soft_student = log_softmax(logits_student / T)
    L_KD = KL_div(soft_student, soft_teacher) × T²

Pourquoi la température ?
- T=1 : distribution normale (peaked, peu d'information sur les classes proches)
- T>1 : distribution "adoucie" (spread out), révèle les relations inter-classes
  que le teacher a apprises (par ex. "ce pixel ressemble un peu à du scatter")
- Le facteur T² compense le fait que les gradients sont réduits par 1/T²

**Valeur initiale recommandée** : T=4.0 (classique en distillation)

API PyTorch : `torch.nn.KLDivLoss(reduction='batchmean')`
Attention : KLDivLoss attend des LOG-probabilités en entrée (student) et des
probabilités normales en target (teacher). Donc :
- Student : `F.log_softmax(logits_s / T, dim=1)`
- Teacher : `F.softmax(logits_t / T, dim=1)`

#### 5.1.3 L_phys — Contrainte Klein-Nishina

C'est la composante "Physics-Informed" qui distingue PI-KD d'une distillation
classique. L'idée : les prédictions du student doivent respecter la physique
de la diffusion Compton.

La section efficace différentielle de Klein-Nishina donne la probabilité qu'un
photon soit diffusé à un angle θ pour une énergie incidente donnée. Si le student
prédit qu'un pixel a reçu un photon diffusé, l'énergie déposée et l'angle de
diffusion impliqué doivent être cohérents avec Klein-Nishina.

**Comment l'implémenter concrètement :**

1. Pour chaque pixel classifié "ICS" par le student, calcule l'angle de
   diffusion implicite à partir de l'énergie déposée (via la cinématique Compton)
2. Évalue la section efficace de Klein-Nishina à cet angle
3. La perte pénalise les prédictions qui correspondent à des angles de diffusion
   physiquement improbables (section efficace très faible)

C'est essentiellement un prior physique sur les prédictions. Plutôt que de dire
au modèle "imite le teacher", on lui dit aussi "respecte les lois de la physique".

**Implémentation** : cette perte est un scalaire différentiable calculé à partir
des logits du student. Elle se backpropage normalement.

### 5.2 Calendrier d'entraînement (curriculum)

Le curriculum progressif est ESSENTIEL. Activer toutes les pertes dès le début
fait diverger l'entraînement parce que le student n'a pas encore appris les
bases.

```
Phase 1 (Semaines 1-2) : λ_kd=0, λ_phys=0
    Le student apprend la tâche de base avec L_task seule.
    Objectif : convergence initiale, F1 > 0.60

Phase 2 (Semaines 3-4) : λ_kd=1.0, λ_phys=0
    On ajoute la distillation. Le student imite le teacher.
    Objectif : F1 > 0.78

Phase 3 (Semaines 5-6) : λ_kd=1.0, λ_phys=0.1 → 0.5
    On ajoute progressivement la contrainte physique.
    λ_phys augmente linéairement sur les 2 semaines.
    Objectif : F1 ≥ 0.83
```

**Implémentation du curriculum** : un simple scheduler qui modifie λ_kd et λ_phys
en fonction de l'epoch courante. Pas besoin de framework compliqué.

### 5.3 Hyperparamètres initiaux recommandés

```yaml
# configs/train_config.yaml
seed: 42

# Architecture
student_channels: [32, 64, 128]
num_blocks: 3

# Optimisation
optimizer: AdamW
learning_rate: 1.0e-3
weight_decay: 1.0e-4
scheduler: CosineAnnealingLR
epochs_phase1: 50
epochs_phase2: 50
epochs_phase3: 50

# Focal Loss
focal_gamma: 2.0
focal_alpha: auto        # calculé depuis le ratio de classes

# Distillation
temperature: 4.0
lambda_kd_phase2: 1.0

# Physique
lambda_phys_start: 0.1
lambda_phys_end: 0.5

# Données
batch_size: 64
num_workers: 4
```

### 5.4 Conseils pratiques pour l'entraînement

- **Logger tout** : pour chaque epoch, log L_task, L_KD, L_phys séparément
  en plus de L_total. Ça te permettra de diagnostiquer si une composante domine
- **Gradient clipping** : utilise `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
  pour éviter les explosions de gradients, surtout quand L_phys s'active
- **Checkpoint** : sauvegarde le meilleur modèle selon F1 sur la validation,
  pas selon la perte totale (la perte peut baisser alors que F1 stagne)
- **Early stopping par phase** : si F1 ne progresse plus dans une phase,
  passe à la suivante plutôt que d'attendre la fin des epochs

---

## 6. Étape 4 — Fine-tuning QAT

### 6.1 Concept fondamental

QAT = Quantization-Aware Training. Le principe : pendant l'entraînement, on
simule la quantification (float32 → int8) en insérant des "fake quantizers"
après chaque couche. Le réseau apprend à être robuste aux erreurs d'arrondi.

```
Sans QAT :  Conv → BN → ReLU → (float32 partout)
Avec QAT :  Conv → BN → ReLU → [FakeQuant] → (simule int8, mais calcule en float32)
```

Les fake quantizers arrondissent les valeurs vers les niveaux int8 les plus
proches pendant le forward, mais laissent passer les gradients tels quels
pendant le backward (c'est le "Straight-Through Estimator").

### 6.2 Séquence d'opérations PyTorch

L'ordre est strict et important :

```
1. Charger le student float32 entraîné (meilleur checkpoint PI-KD)
2. Définir la config de quantification (qconfig)
3. Préparer le modèle : torch.quantization.prepare_qat(model)
   → Ceci insère les fake quantizers
4. Fine-tuner (~20 epochs, lr = lr_original / 10)
   → La loss physique reste active
5. Convertir : torch.quantization.convert(model)
   → Remplace les fake quantizers par de vraies opérations int8
6. Évaluer le F1 du modèle quantifié
```

### 6.3 Configuration de quantification

```python
# La qconfig définit comment quantifier poids et activations
# Pour compatibilité hls4ml, utilise la config par défaut PyTorch
torch.quantization.get_default_qat_qconfig('fbgemm')  # pour CPU x86
```

**Pourquoi 'fbgemm' et pas 'qnnpack'** : fbgemm est le backend de quantification
pour x86 (ton PC d'entraînement). Pour FPGA, c'est hls4ml qui se charge de la
conversion finale, pas le backend PyTorch. Le QAT sert juste à "endurcir" le
modèle aux erreurs d'arrondi.

### 6.4 Fusion de modules

AVANT de préparer le QAT, tu dois fusionner Conv+BN+ReLU :

```python
torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)
```

La fusion est obligatoire parce que :
- BN est absorbé dans les poids de Conv (pas de BN séparé en inférence)
- ReLU est appliqué in-place après la convolution fusionnée
- Ça réduit le nombre d'opérations sur FPGA

Tu dois spécifier exactement quels triplets fusionner, en utilisant les noms
d'attributs de ton module PyTorch.

### 6.5 Critère de validation

Après QAT + convert, évalue le F1 du modèle quantifié sur ton set de test.
La dégradation acceptée est < 1% par rapport au student float32.

Si la dégradation dépasse 1% :
- Augmente le nombre d'epochs de fine-tuning QAT (20 → 40 → 60)
- Réduis le learning rate
- Vérifie que la fusion de modules est correcte
- En dernier recours, essaie une précision plus élevée (int16 au lieu d'int8)

---

## 7. Étape 5 — Export hls4ml et déploiement Xilinx

### 7.1 Qu'est-ce que hls4ml

hls4ml est un outil développé au CERN/Fermilab (très pertinent pour ton domaine)
qui convertit un modèle ML entraîné en code HLS C++ synthétisable. Le code
généré est ensuite compilé par Vivado/Vitis HLS en un circuit FPGA.

### 7.2 Pipeline d'export

```
Student quantifié (PyTorch)
    │
    ├─ hls4ml.converters.convert_from_pytorch_model()
    │   → Génère un projet HLS C++
    │
    ├─ C Simulation (csim)
    │   → Vérifie la correction fonctionnelle (résultats identiques à PyTorch)
    │   → Donne une ESTIMATION de latence
    │
    ├─ Synthèse HLS (Vitis HLS)
    │   → Génère le RTL (Verilog/VHDL)
    │   → Rapport de ressources : LUT, FF, DSP, BRAM
    │
    ├─ Implémentation Vivado
    │   → Place & Route sur le FPGA cible
    │   → Rapport de timing (fréquence max)
    │
    └─ Génération du bitstream
        → Fichier .bit prêt à charger sur le FPGA
```

### 7.3 Ce que hls4ml supporte (et ne supporte pas)

**Supporte bien :**
- Conv2D, Dense/Linear, BatchNorm (fusionné), ReLU, ReLU6
- MaxPool, AveragePool, GlobalAveragePool
- Opérations élément-par-élément (Add pour skip connections)
- Quantification uniforme (int8, int16)

**Ne supporte PAS (ou mal) :**
- Softmax (exponentielles coûteuses)
- LayerNorm, GroupNorm
- Attention / self-attention
- Opérations dynamiques (boucles de taille variable)
- Certaines activations exotiques (GELU, Swish, Mish)

C'est exactement pourquoi le StudentNet n'a ni attention ni LayerNorm.

### 7.4 Paramètres hls4ml importants

Quand tu appelles le convertisseur, tu dois spécifier :

- **reuse_factor** : contrôle le parallélisme. reuse_factor=1 signifie que chaque
  multiplication a son propre multiplicateur (latence minimale, ressources maximales).
  reuse_factor=N signifie qu'un multiplicateur est partagé entre N opérations
  (latence ×N, ressources ÷N). Commence par un reuse_factor élevé et diminue.

- **precision** : 'ap_fixed<16,6>' signifie 16 bits total dont 6 pour la partie
  entière et 10 pour la partie fractionnaire. hls4ml utilise la notation
  Arbitrary Precision de Xilinx.

- **strategy** : 'Latency' (optimise pour la latence, plus de ressources) ou
  'Resource' (optimise pour les ressources, plus de latence)

### 7.5 Installation de hls4ml

```bash
pip install hls4ml[profiling]
```

Pour la synthèse complète, tu as besoin de Vivado/Vitis HLS installé sur ta
machine (ou sur un serveur du labo). La licence est académique via le programme
Xilinx University.

### 7.6 Itération architecture ↔ ressources

Le workflow réaliste est itératif :

```
Concevoir StudentNet → Export hls4ml → C simulation → Rapport ressources
    ↑                                                        │
    │                                                        │
    └── Trop gros? Réduire canaux / blocs ──────────────────┘
    └── Trop lent? Réduire reuse_factor (si ressources OK) ─┘
    └── F1 trop bas? Augmenter capacité ────────────────────┘
```

C'est normal de faire 3 à 5 itérations. Ne vise pas la perfection dès le premier
essai.

---

## 8. Évaluation et métriques

### 8.1 Métriques ML

- **F1 score** : métrique principale (harmonique de précision et rappel)
- **Précision** : parmi les pixels classés ICS, combien le sont vraiment
- **Rappel** : parmi les vrais pixels ICS, combien sont détectés
- **Matrice de confusion** : pour visualiser les erreurs

Utilise `torchmetrics.F1Score` ou `sklearn.metrics.f1_score`.

### 8.2 Métriques d'image (post-reconstruction)

Après correction ICS et reconstruction d'image :
- **CNR** (Contrast-to-Noise Ratio)
- **SNR** (Signal-to-Noise Ratio)
- **MTF** (Modulation Transfer Function)

Ces métriques sont celles que ton jury évaluera. Le F1 est un proxy pendant
l'entraînement, mais c'est la qualité d'image reconstruite qui compte in fine.

### 8.3 Métriques de compression

- **Ratio de compression** : paramètres teacher / paramètres student
- **Speedup** : latence teacher / latence student (sur FPGA)
- **Efficacité énergétique** : watts consommés sur FPGA vs GPU

### 8.4 Tableau comparatif à produire

Pour la thèse, tu voudras un tableau comme :

```
| Modèle           | Params | F1    | CNR   | Latence | Ressources FPGA |
|-------------------|--------|-------|-------|---------|-----------------|
| PhysFormer (GPU)  | XXX K  | 0.85  | XX.X  | XX ms   | N/A             |
| Student float32   | XX K   | 0.8X  | XX.X  | XX ms   | N/A             |
| Student QAT int8  | XX K   | 0.8X  | XX.X  | XX µs   | XX% LUT, ...    |
```

---

## 9. Dépendances et versions

```
# Entraînement
torch >= 2.0
torchvision
torchinfo            # résumé d'architecture
torchmetrics         # métriques standardisées

# Export FPGA
hls4ml[profiling]    # conversion vers HLS
onnx                 # format intermédiaire (optionnel)

# Visualisation
tensorboard          # logs d'entraînement
matplotlib           # figures

# Utilitaires
pyyaml               # config files
tqdm                 # barres de progression
```

---

## 10. Pièges courants et comment les éviter

1. **Le student ne converge pas en Phase 1**
   → Vérifie que la focal loss est correctement implémentée (signe, gamma)
   → Vérifie le pos_weight / alpha
   → Essaie un learning rate plus petit

2. **L_KD domine L_task après activation**
   → Réduis λ_kd (commencer à 0.5 au lieu de 1.0)
   → Augmente T (température plus haute = gradients KD plus faibles)

3. **L_phys fait diverger l'entraînement**
   → Active la plus progressivement (λ_phys de 0.01 à 0.1, pas 0.1 à 0.5)
   → Vérifie que la physique est correctement implémentée (unités!)
   → Utilise le gradient clipping

4. **F1 du student plafonne bien en dessous de 0.83**
   → Le student est peut-être trop petit : augmente les canaux
   → Ou le teacher n'est pas assez bon : vérifie son F1 réel
   → Ou les données sont bruitées : examine les échantillons d'erreur

5. **Le QAT dégrade le F1 de plus de 1%**
   → La fusion de modules est probablement incomplète
   → Le modèle utilise peut-être une opération non quantifiable
   → Essaie plus d'epochs de fine-tuning QAT

6. **hls4ml échoue à la conversion**
   → Couche non supportée dans le student (vérifier la liste 7.3)
   → Le modèle n'a pas été converti via torch.quantization.convert()
   → Version hls4ml incompatible avec ta version PyTorch

---

## 11. Ordre d'implémentation recommandé (pas à pas)

```
Semaine 0 : Setup projet, CLAUDE.md, configs, dépendances
            Vérifier que le teacher charge et produit des logits corrects

Semaine 1 : Implémenter StudentNet (models/student.py, models/layers.py)
            Tester que le forward pass fonctionne
            Vérifier le nombre de paramètres

Semaine 2 : Implémenter les 3 losses séparément, avec tests unitaires
            Tester chaque loss sur des tenseurs synthétiques

Semaine 3 : Implémenter le DataModule et la boucle d'entraînement
            Phase 1 : entraîner avec L_task seule

Semaine 4 : Phase 2 : activer L_KD
            Débugger si nécessaire

Semaine 5 : Phase 3 : activer L_phys progressivement
            Évaluer F1

Semaine 6 : QAT fine-tuning
            Export hls4ml (au moins la C simulation)
            Tableau comparatif final
```

---

## 12. Ressources de référence

- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) — papier
  fondateur de la distillation, explique la température et les soft labels
- Lin et al., "Focal Loss for Dense Object Detection" (2017) — focal loss originale
- Documentation hls4ml : https://fastmachinelearning.org/hls4ml/
- PyTorch Quantization : https://pytorch.org/docs/stable/quantization.html
- Howard et al., "MobileNets" (2017) — depthwise separable convolutions
- Brevitas (pour référence, même si on utilise PyTorch natif) :
  https://github.com/Xilinx/brevitas






1. Simulation-Augmented Distillation (exploite ton avantage unique)
Tu as quelque chose que presque personne dans le domaine KD n'a : une chaîne de simulation Monte Carlo (GATE) complète qui génère des données avec ground truth parfait. La plupart des travaux de distillation en imagerie médicale n'ont que des données cliniques bruitées. L'idée serait d'utiliser GATE pour générer des cas de scatter ciblés (angles rares, géométries extrêmes, matériaux variés) et de les injecter pendant la distillation comme exemples d'entraînement "difficiles". C'est un curriculum augmenté par la physique, pas juste par des transformations géométriques classiques. L'argument en conférence : le student voit des configurations que le teacher n'a jamais rencontrées dans les données réelles, mais qui sont physiquement valides. Ça renforce la robustesse et la généralisabilité. Et tu as déjà toute l'infrastructure pour le faire (tes 30 runs paramétriques, tes 4 axes de sweep).
2. Early Exit Network avec contrainte physique
Plutôt qu'un student monolithique, conçois un réseau à sorties anticipées. L'idée : les pixels "faciles" (clairement pas ICS, énergie hors de la fenêtre Compton) sortent après 1 ou 2 couches seulement. Seuls les pixels ambigus traversent le réseau complet. C'est extrêmement pertinent pour FPGA parce que ça s'aligne naturellement avec l'architecture dataflow (pipeline), et ça réduit la latence moyenne de façon drastique puisque ~96% de tes pixels sont négatifs et "faciles". La contrainte physique intervient dans le critère de sortie : un pixel ne peut sortir tôt que si son énergie déposée rend la diffusion Compton physiquement impossible selon Klein-Nishina. C'est un critère de confiance fondé sur la physique plutôt que sur un simple seuil de softmax. C'est récent dans la littérature (il y a des travaux 2025 sur les EENN pour FPGA), mais personne ne l'a fait avec une contrainte physique comme critère de sortie.
3. Certification de cohérence physique
Au delà de la perte physique pendant l'entraînement, tu pourrais ajouter une étape de vérification post-hoc : pour chaque prédiction du student quantifié, vérifie formellement que l'énergie et l'angle de scatter impliqués sont compatibles avec la section efficace de Klein-Nishina. Ça donne un taux de "violation physique" mesurable. L'argument pour le jury et la conférence : en imagerie médicale, la fiabilité est aussi importante que la performance. Un modèle qui garantit 0% de violations physiques a un avantage clair sur un modèle qui a un bon F1 mais fait parfois des prédictions physiquement impossibles. Ce type de vérification est très rare dans la littérature KD.
4. Latency-Aware Distillation Loss
Ajoute un terme de perte qui pénalise directement la latence FPGA estimée. L'idée : utilise les profils de latence par couche fournis par hls4ml pour construire une table de correspondance (opération → coût en cycles). Pendant la distillation, le student est pénalisé non seulement sur la tâche et la physique, mais aussi sur le coût matériel de ses activations. Concrètement, ça pousse le student à utiliser des chemins de calcul "légers" quand c'est suffisant. C'est du hardware-software co-design intégré dans la boucle d'entraînement. C'est plus ambitieux à implémenter, mais ça te positionne à l'intersection exacte de tes deux axes (physique + FPGA).
Ma recommandation
Si tu dois en choisir une seule, je te dirais l'idée 1 (Simulation-Augmented Distillation) parce que c'est la plus réaliste à implémenter dans ton calendrier, elle exploite un avantage concurrentiel que tu as déjà (le pipeline GATE), et elle a un narratif très fort en conférence : "on utilise la simulation Monte Carlo non seulement pour caractériser l'ICS, mais aussi pour entraîner le modèle qui le corrige". Ça ferme la boucle entre tes deux axes de thèse de façon élégante.
Si tu veux viser plus haut (et que le temps le permet), combine les idées 1 et 2 : un early exit network entraîné par distillation augmentée par simulation, avec un critère de sortie fondé sur la physique. Ça ferait une contribution vraiment distinctive.
Tu veux qu'on explore une de ces idées en détail ?

3. Hardware-Aware Distillation (Quantization-Friendly)
NSS MIC has a strong "RTSD" (Real-Time System Detector) component. A "nice" work would bridge the gap between software and FPGA/ASIC:

Bit-Width Sensitivity Distillation: Train the student network using the teacher’s guidance specifically to be robust against 4-bit or 8-bit quantization.

HLS-Ready Architectures: Design the student using layers that are specifically optimized for HLS4ML (e.g., separable convolutions or limited-depth transformers).