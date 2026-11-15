# Détection des comportements en classe avec YOLOv8

Projet de vision par ordinateur qui détecte automatiquement ce que font les élèves dans une salle de classe.

## Objectif

Entraîner un modèle YOLOv8 capable de reconnaître 4 comportements :

| Classe | Signification |
|--------|---------------|
| hand   | Main levée |
| read   | Lecture |
| sleep  | Endormissement |
| write  | Écriture |

## Technologies

- Python 3.10
- Ultralytics YOLOv8 (modèle nano `yolov8n`)
- Roboflow (dataset)
- Kaggle Notebook avec GPU Tesla T4

## Dataset

Dataset **Classroom** (version 24) hébergé sur Roboflow, au format YOLOv8.
Source : `mariotn689/classroom-coc8d`

## Méthode

1. Téléchargement du dataset avec l'API Roboflow
2. Entraînement de `yolov8n` pendant 15 époques, images en 704x704
3. Affichage des courbes (F1, précision, rappel, matrice de confusion)
4. Validation du meilleur modèle
5. Prédiction sur le jeu de test avec un seuil de confiance de 0.50

## Résultats

Résultats sur le jeu de validation (425 images, 1167 instances) :

| Classe | Précision | Rappel | mAP50 | mAP50 95 |
|--------|-----------|--------|-------|----------|
| Toutes | 0.914 | 0.960 | 0.970 | 0.716 |
| hand   | 0.924 | 0.949 | 0.965 | 0.702 |
| read   | 0.921 | 0.947 | 0.972 | 0.720 |
| sleep  | 0.928 | 0.985 | 0.974 | 0.683 |
| write  | 0.883 | 0.958 | 0.970 | 0.759 |

Vitesse d'inférence : environ 4.5 ms par image sur GPU T4.

## Installation

```bash
pip install ultralytics roboflow
```

## Utilisation

1. Créez un compte Roboflow et récupérez votre clé API
2. Définissez la clé dans une variable d'environnement :

```bash
export ROBOFLOW_API_KEY="votre_cle"
```

3. Ouvrez et exécutez `yolov8.ipynb` (Kaggle ou local avec GPU)

Pour lancer une prédiction avec le modèle entraîné :

```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.50 source=chemin/vers/images
```

## Structure du projet

```
.
├── yolov8_finetuning.ipynb   # Notebook complet commenté
└── README.md
```

## Pistes d'amélioration

- Entraîner plus d'époques ou tester `yolov8s` et `yolov8m`
- Ajouter de la détection en temps réel avec une webcam
- Compter les comportements par séance

## Auteur

Makhtar Faye
