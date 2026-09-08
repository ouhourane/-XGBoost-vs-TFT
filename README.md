# ⚡ Baseline contrefactuelle 15 min — XGBoost vs Temporal Fusion Transformer (TFT)

> Estimation de la consommation électrique *« business-as-usual »* à la granularité **15 minutes** pour quatre segments de clientèle Hydro-Québec, afin de **mesurer l'effacement de charge** (énergie effacée, précharge, rebond) lors d'événements de gestion de la demande.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch--Forecasting-TFT-red)
![Databricks](https://img.shields.io/badge/Databricks-Spark%20%7C%20Unity%20Catalog-orange)
![GPU](https://img.shields.io/badge/GPU-CUDA%20auto--detect-yellow)

---

## 📌 Table des matières

- [Contexte métier](#-contexte-métier)
- [Objectif](#-objectif)
- [Architecture du pipeline](#-architecture-du-pipeline)
- [Les 4 segments modélisés](#-les-4-segments-modélisés)
- [Fonctionnalités clés](#-fonctionnalités-clés)
- [Structure du notebook](#-structure-du-notebook)
- [Prérequis & installation](#-prérequis--installation)
- [Configuration](#-configuration)
- [Exécution](#-exécution)
- [Résultats produits](#-résultats-produits)
- [Choix de conception importants](#-choix-de-conception-importants)
- [Dépannage (erreurs connues corrigées)](#-dépannage-erreurs-connues-corrigées)
- [Compétences démontrées](#-compétences-démontrées)
- [Limites & pistes d'amélioration](#-limites--pistes-damélioration)

---

## 🎯 Contexte métier

Lors d'un **événement de gestion de la demande** (effacement, appel de puissance), Hydro-Québec incite ses clients à réduire leur consommation. Pour quantifier l'effet réel de ces événements, il faut répondre à une question contrefactuelle :

> *« Quelle **aurait été** la consommation s'il n'y avait **pas eu** d'événement ? »*

Cette consommation hypothétique s'appelle la **baseline**. En comparant la baseline prédite à la consommation réelle les jours d'événement, on mesure trois grandeurs :

| Grandeur | Définition |
|----------|-----------|
| **Énergie effacée** | Déficit de consommation le jour J (baseline − réel) |
| **Précharge** | Surplus de consommation la veille (J-1), en anticipation |
| **Rebond** | Surplus de consommation le lendemain (J+1), par report |

L'**énergie évitée nette** = énergie effacée − précharge − rebond.

---

## 🎯 Objectif

Construire une baseline **fiable, non biaisée et sans fuite d'information**, à la granularité **15 minutes** (aucune agrégation horaire), puis **comparer deux familles de modèles** :

1. **XGBoost** — modèle d'arbres gradient-boostés (Partie 1)
2. **Temporal Fusion Transformer (TFT)** — modèle d'apprentissage profond pour séries temporelles (Partie 2)

La comparaison se fait **sur les jours propres** (hors événement), là où la baseline doit coller au réel.

---

## 🏗 Architecture du pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  DONNÉES (Spark SQL / Unity Catalog)                          │
│  4 segments de consommation + météo pondérée + événements     │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  PRÉPARATION 15 MIN                                           │
│  • Grille régulière 15 min (96 pas/jour)                     │
│  • Gestion des trous, exclusion des labels non finis         │
│  • Détection automatique maintenance / rebond                │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  INGÉNIERIE DE VARIABLES                                      │
│  • Calendrier + cyclicité (sin/cos)                          │
│  • Météo retardée (lag 24 h, delta 24 h), interactions temp  │
│  • Profils de référence par créneau (fallback anti-NaN)      │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌────────────────────┐                 ┌────────────────────┐
│  PARTIE 1 : XGBoost │                 │  PARTIE 2 : TFT     │
│  • Cible résiduelle │                 │  • Encoder 7 j réel │
│  • Monotonie HDD/CDD│                 │  • Decoder 96 pas   │
│  • Pseudo-Huber     │                 │    (calendrier +    │
│  • Débiaisage local │                 │     météo, NO leak) │
│  • Clipping         │                 │  • Glissant J par J │
└─────────┬──────────┘                 └─────────┬──────────┘
          └──────────────────┬────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  SYNTHÈSES & COMPARAISON                                      │
│  • Métriques par segment (RMSE, MAE, R², BIAS)               │
│  • Résumé journalier avec statut                             │
│  • Bilan de déplacement de charge                            │
│  • XGBoost vs TFT sur jours propres → gagnant par R²         │
└──────────────────────────────────────────────────────────────┘
```

---

## 🗂 Les 4 segments modélisés

| Segment | Description | Couleur de tracé |
|---------|-------------|------------------|
| **Pilote** | Clients pilotés (effacement actif) | Crimson |
| **Résidentiel** | Clientèle résidentielle | Royal blue |
| **Affaire** | Clientèle commerciale / industrielle | Dark orange |
| **Comportemental** | Clients **non** pilotés (réponse comportementale) | Sea green |

Le pipeline est **générique** : les fonctions lisent la configuration d'un segment (`SEGMENTS`) plutôt que des noms de colonnes codés en dur, ce qui permet une boucle unique sur les 4 segments.

---

## ✨ Fonctionnalités clés

### Partie 1 — XGBoost
- ✅ **Cible résiduelle** : le modèle prédit `énergie − profil_référence`, plus facile à apprendre.
- ✅ **Contraintes de monotonie** sur HDD/CDD → baseline physiquement cohérente (plus il fait froid, plus la consommation monte).
- ✅ **Objectif robuste** (`reg:pseudohubererror`) → moins sensible aux valeurs aberrantes.
- ✅ **Débiaisage local** : recalage du biais sur une fenêtre récente (30 j).
- ✅ **Clipping** des prédictions entre quantiles.
- ✅ **Détection automatique de maintenance** pour ne pas polluer l'apprentissage.

### Partie 2 — TFT
- ✅ **Anti-fuite strict** : le décodeur ne voit **que** le calendrier et la météo, **jamais** l'événement.
- ✅ **Baseline glissante jour par jour** : encoder = 7 jours réels, decoder = 96 créneaux futurs.
- ✅ **Persistance** : les modèles sont sauvegardés dans un **Volume Unity Catalog** pour réutilisation (pas de réentraînement inutile).
- ✅ **Détection GPU/CPU automatique**.
- ✅ **Reproductibilité** : graines fixées (`SEED`, `SEED_TFT`).

---

## 📓 Structure du notebook

### Partie 1 — Baseline XGBoost (cellules 1 à 20)

| Cellule | Rôle |
|---------|------|
| 1 | Imports et configuration (auto-installation des paquets) |
| 2 | Paramètres globaux (granularité, fenêtre de test, flags) |
| 3 | Détection GPU / CPU |
| 4 | Chargement fusionné Spark SQL (4 segments + météo pondérée) |
| 5 | Configuration des 4 segments |
| 6 | Alignement sur grille 15 min |
| 7 | Variables événementielles (phases pré/post) |
| 8 | Détection maintenance / rebond |
| 9 | Features communes (calendrier, cyclicité, météo retardée) |
| 10 | Masks train baseline + profils de référence (anti-NaN) |
| 11 | Listes de features |
| 12 | Modèle XGBoost (monotonie, objectif robuste) |
| 13 | Entraînement baseline (correctif label NaN) |
| 14 | Pipeline complet par segment |
| 15 | Synthèse des performances |
| 16 | Résumé journalier avec statut |
| 17 | Bilan énergétique du déplacement de charge |
| 18 | Graphique Réel vs Baseline |
| 19 | Importance des variables |
| 20 | Export des vues Spark |

### Partie 2 — Comparaison XGBoost vs TFT (cellules P2.1 à P2.9)

| Cellule | Rôle |
|---------|------|
| P2.1 | Imports TFT (torch, lightning, pytorch-forecasting) |
| P2.1bis | Dossier de checkpoints inscriptible (correctif PermissionError) |
| P2.1ter | Dossier persistant des modèles (Volume Unity Catalog) |
| P2.2 | Hyperparamètres TFT |
| P2.3 | Préparation d'un segment pour le TFT |
| P2.4 | Entraînement TFT (ou rechargement du modèle) |
| P2.5 | Baseline TFT glissante jour par jour |
| P2.6 | Exécution TFT sur les 4 segments |
| P2.7 | Synthèse comparative XGBoost vs TFT |
| P2.8 | Graphiques Réel vs XGBoost vs TFT |
| P2.9 | Export des vues Spark |

---

## 🔧 Prérequis & installation

### Environnement
- **Databricks** avec cluster ML (CPU ou **GPU** recommandé pour le TFT).
- Accès **Unity Catalog** aux tables sources et à un **Volume** persistant.
- Python 3.10+.

### Dépendances (auto-installées par le notebook)
```bash
xgboost>=2.0
scikit-learn
numpy
pandas
matplotlib
torch
lightning
pytorch-forecasting
```

> Le notebook contient une fonction `_assurer(pkg, pip_name)` qui installe automatiquement tout paquet manquant. Aucune installation manuelle n'est requise.

---

## ⚙️ Configuration

Les principaux paramètres sont centralisés dans la **cellule 2** :

```python
# Granularité
FREQ           = "15min"
PAS_PAR_JOUR   = 96          # 24 h × 4 créneaux

# Fenêtre de backtest
DATE_TEST       = "2026-03-03"
NB_JOURS_AVANT  = 8
NB_JOURS_APRES  = 8

# Améliorations activables (flags booléens)
MODE_CIBLE_RESIDU        = True
UTILISER_MONOTONE        = True
DEBIASAGE_LOCAL          = True
CLIP_PREDICTIONS         = True
DETECTER_MAINTENANCE     = True
OBJECTIF_XGB             = "reg:pseudohubererror"
```

Pour le TFT (cellules P2.2 et P2.1ter) :

```python
ENC_JOURS       = 7          # encoder = 7 jours réels (672 pas)
DEC_JOURS       = 1          # decoder = 1 jour (96 pas)
MAX_EPOCHS      = 30
HIDDEN_SIZE     = 32
ATTENTION_HEADS = 4

REENTRAINER_TFT = False       # True = force un réentraînement complet
MODELE_DIR      = "/Volumes/.../vehiculeelectrique"   # persistance
```

---

## ▶️ Exécution

1. Ouvrir le notebook dans **Databricks**.
2. Attacher un cluster **ML** (GPU conseillé pour la Partie 2).
3. Vérifier / ajuster les paramètres de la cellule 2 (surtout `DATE_TEST`).
4. **Exécuter tout** (`Run All`).

**Ordre logique** :
- Partie 1 (cellules 1→20) produit `resultats_segments` (données + baseline XGBoost).
- Partie 2 (P2.1→P2.9) **réutilise** `resultats_segments`, entraîne/recharge les TFT, puis compare.

> ⏱ Au premier lancement, les TFT s'entraînent et sont sauvegardés dans le Volume. Aux lancements suivants, ils sont **rechargés automatiquement** (`REENTRAINER_TFT = False`), ce qui accélère fortement l'exécution.

---

## 📊 Résultats produits

### Tableaux (exportés en vues temporaires Spark)
| Vue Spark | Contenu |
|-----------|---------|
| `perf_test_affaire_15min_par_segment` | Métriques par segment (tous jours vs jours propres) |
| `resume_journalier_test_affaire_15min` | Détail jour par jour avec statut |
| `bilan_deplacement_charge_affaire_15min` | Énergie effacée, précharge, rebond, énergie évitée nette |
| `comparaison_xgb_tft_15min` | Métriques XGBoost vs TFT |
| `resume_gagnant_xgb_tft_15min` | Gagnant par segment + gain de R² |
| `detail_baselines_xgb_tft_15min` | Détail créneau par créneau des deux baselines |

### Graphiques
- **Réel vs Baseline** 15 min avec bandes d'événement/maintenance (par segment).
- **Importance des variables** (top 30) pour chaque modèle XGBoost.
- **Réel vs XGBoost vs TFT** superposés, jours d'événement colorés.

### Métriques calculées
`RMSE` · `MAE` · `R²` · `BIAS` · `N`

---

## 🧠 Choix de conception importants

| Principe | Mise en œuvre |
|----------|---------------|
| **Anti-fuite** | Le décodeur du TFT ne reçoit que calendrier + météo, jamais l'événement |
| **Robustesse NaN** | Profil de référence garanti sans NaN (fallback global) ; exclusion du fit des lignes non finies |
| **Cohérence physique** | Contraintes de monotonie sur HDD/CDD |
| **Reproductibilité** | Graines fixées, device GPU/CPU auto-détecté |
| **Persistance** | Modèles TFT sauvegardés dans un Volume Unity Catalog |
| **Évaluation honnête** | Comparaison uniquement sur les jours propres (hors événement) |

---

## 🩹 Dépannage (erreurs connues corrigées)

### `XGBoostError: Label contains NaN`
**Cause** : le passage en 15 min crée des lignes où `énergie` est NaN (trous comblés), et `profil_ref_final` pouvait être NaN. Comme la cible résiduelle est `énergie − profil_ref_final`, un seul NaN faisait échouer `fit`.
**Correctif** : profil de référence **garanti sans NaN** (repli progressif + moyenne globale) + **exclusion du fit** de toute ligne dont le label ou l'énergie n'est pas finie (cellules 10 et 13).

### `PermissionError: [Errno 13] '/Workspace/.../checkpoints'`
**Cause** : sur Databricks, `/Workspace/...` est en lecture seule pour l'écriture de fichiers par Lightning.
**Correctif** : `_dossier_inscriptible()` teste plusieurs emplacements locaux (`/local_disk0`, `/dbfs/tmp`, `/tmp`) et redirige **toutes** les sorties Lightning (checkpoints + logs CSV) vers un disque inscriptible (cellule P2.1bis).

---

## 💼 Compétences démontrées

- **Séries temporelles** : baseline contrefactuelle, adstock/carryover thermique, cyclicité.
- **Machine Learning** : XGBoost avec contraintes de monotonie, objectif robuste, débiaisage.
- **Deep Learning** : Temporal Fusion Transformer (pytorch-forecasting), gestion encoder/decoder, anti-fuite.
- **Data Engineering** : Spark SQL, Unity Catalog, jointures multi-sources, météo pondérée.
- **MLOps** : persistance de modèles, détection GPU/CPU, reproductibilité, gestion d'erreurs d'environnement.
- **Rigueur analytique** : évaluation sur jours propres, métriques multiples, validation visuelle.

---

## 🚀 Limites & pistes d'amélioration

- ⏳ **Incertitude** : ajouter des **intervalles de prédiction** (le TFT via `QuantileLoss` le permet déjà — les exposer).
- 📅 **Validation croisée temporelle** sur plusieurs `DATE_TEST` plutôt qu'une date unique.
- 🌡 **Adstock thermique avancé** : effet retardé avec pic décalé plutôt que lag fixe 24 h.
- 🔁 **Backtest multi-événements** pour robustifier le bilan de déplacement de charge.
- 📈 **Suivi MLflow** des runs (métriques, artefacts, comparaison de versions).

---

## 📄 Licence & usage

Projet interne Hydro-Québec — données confidentielles non incluses.
Le code illustre une méthodologie de **mesure d'effacement de charge** à 15 min.

---

> *Développé pour la modélisation Résidentiel / Affaire / Pilote / Comportemental — granularité 15 minutes, sans agrégation horaire.*
