# Cartes de Kohonen (SOM)

> Implémentation et étude expérimentale d'une carte auto-organisatrice de Kohonen sur une distribution 2D, avec analyse de l'impact des hyperparamètres.

**Stack** : Python · NumPy · Matplotlib

## Contexte

Projet réalisé dans le cadre du cours d'IA à **Polytech Lyon** (Pr. Mathieu Lefort), en binôme avec Lefèvre.

Le squelette du code (`kohonen.py`) est fourni par Pr. Lefort sous licence BSD. La contribution étudiante consiste à :

- compléter les méthodes `compute()` et `learn()` du neurone (règle de Kohonen avec voisinage gaussien),
- mener une étude expérimentale d'impact des hyperparamètres,
- rédiger le rapport d'analyse (`Rapport_LEFEVRE_LARZUL.docx`).

## L'algorithme

Une carte auto-organisatrice (SOM) est un réseau de neurones non-supervisé qui projette un espace d'entrée de grande dimension sur une grille 2D tout en préservant les relations topologiques. Chaque neurone porte un vecteur de poids dans l'espace d'entrée. À l'apprentissage, pour chaque exemple `x` :

1. on identifie le neurone gagnant (BMU), celui dont le vecteur de poids est le plus proche de `x`,
2. on met à jour les poids du gagnant et de ses voisins par la règle :

```
w ← w + eta · exp(- ||r - r*||² / 2σ²) · (x - w)
```

avec `eta` le taux d'apprentissage et `sigma` la largeur du voisinage gaussien.

## Étude expérimentale

Trois hyperparamètres clés sont étudiés séparément, puis combinés :

| Dossier | Variable étudiée | Description |
|---|---|---|
| `Result_eta/` | `eta` | Influence du taux d'apprentissage à itérations fixes |
| `Result_sigma/` | `sigma` | Influence de la largeur du voisinage |
| `Result_N/` | `N` | Influence du nombre d'itérations d'apprentissage |
| `Result_eta_sigma_var/` | `eta`, `sigma` décroissants | Décroissance simultanée des deux paramètres au cours du temps |

Chaque dossier contient les cartes finales de poids générées pour différentes valeurs, permettant la comparaison visuelle.

## Lancer une expérience

```bash
pip install numpy matplotlib
python kohonen.py
```

Les paramètres (`eta`, `sigma`, nombre d'itérations, taille de grille) sont définis dans le `main()` de `kohonen.py`.

## Fichiers

```
kohonen-som/
├── kohonen.py                       # Implémentation SOM (TODOs complétés)
├── tuto_numpy.py                    # Tutoriel NumPy fourni avec le sujet
├── donnees_apprentissage.png        # Visualisation des données
├── projet.pdf                       # Énoncé du sujet
├── Rapport_LEFEVRE_LARZUL.docx      # Rapport d'analyse rendu
├── Result_eta/                      # Résultats variant eta
├── Result_sigma/                    # Résultats variant sigma
├── Result_N/                        # Résultats variant le nombre d'itérations
└── Result_eta_sigma_var/            # Résultats avec décroissance
```
