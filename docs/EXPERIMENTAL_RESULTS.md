# Résultats expérimentaux

## 1. Sources des résultats

Les résultats sont lus depuis :

- `reports/results/experiments.csv` et `reports/results/runs/` ;
- `reports/results/phase_2x2_scaler_initialization/` ;
- `reports/results/final_audit/` ;
- `reports/results/final_model/`.

Les résultats bruts n'ont pas été modifiés par cette documentation.

## 2. Grille historique : 60 runs

La grille contient 5 stratégies × 4 fractions × 3 seeds. Les moyennes de
Macro F1 validation rapportées dans `statistical_summary_60.csv` sont :

| Stratégie | 1 % | 5 % | 10 % | 100 % |
|---|---:|---:|---:|---:|
| Scratch | 0.344803 | 0.428086 | 0.431560 | 0.468557 |
| Frozen | 0.051572 | 0.059451 | 0.074275 | 0.170935 |
| Partial | 0.060584 | 0.224912 | 0.297803 | 0.462761 |
| Fine-tuning | 0.073217 | 0.205871 | 0.299996 | 0.465927 |
| Progressive | 0.031457 | 0.101691 | 0.127293 | 0.376880 |

Dans cette grille historique, Scratch est le meilleur à chaque fraction
selon ces moyennes, mais les comparaisons sont confondues par les protocoles
de preprocessing source/cible. Cela ne constitue donc pas une preuve
universelle contre le Transfer Learning.

## 3. Expérience contrôlée 2×2 : 48 runs

Les quatre cellules sont `random_target`, `pretrained_target`,
`random_source` et `pretrained_source`, avec 4 fractions et 3 seeds.

| Cellule | 1 % | 5 % | 10 % | 100 % |
|---|---:|---:|---:|---:|
| random_target | 0.344803 | 0.428086 | 0.431560 | 0.468557 |
| pretrained_target | 0.298011 | 0.417416 | 0.437055 | 0.475289 |
| random_source | 0.360538 | 0.452514 | 0.474018 | 0.518541 |
| pretrained_source | 0.247298 | 0.454823 | 0.467568 | 0.515805 |

Cette expérience isole mieux les facteurs, car les cellules partagent les
hyperparamètres, seeds et sous-ensembles prévus. L'interaction varie selon
la fraction ; elle doit être interprétée fraction par fraction, sans effet
global simplificateur.

## 4. Audit et validation statistique

`final_audit_report.md` vérifie 60 JSON historiques, 48 JSON contrôlés,
l'absence de métriques manquantes/non finies et la présence des 18 classes.
La validation complémentaire corrige les IC historiques dans des fichiers
séparés (`statistical_validation/`) et calcule les contrastes appariés.

Les résultats restent descriptifs : trois seeds ne permettent pas une
conclusion confirmatoire robuste.

## 5. Modèle final

Le modèle final a été sélectionné par Macro F1 validation, sans utiliser le
test. Il s'agit de Scratch à 100 %, seed 0 :

| Métrique test | Valeur |
|---|---:|
| Accuracy | 0.753420 |
| Macro Precision | 0.476842 |
| Macro Recall | 0.578759 |
| Macro F1 | 0.468371 |
| Weighted F1 | 0.756415 |

La meilleure époque est 20 et le Macro F1 validation vaut 0.474940. Le
Weighted F1 élevé par rapport au Macro F1 indique que les classes majoritaires
dominent la performance globale. `UDPLag` et `WebDDoS` ont respectivement
F1 0.061538 et 0.029070, avec support 8 chacune.

## 6. Conclusion prudente

Scratch a été retenu parce qu'il fournit la meilleure configuration de
validation dans la sélection documentaire et un modèle exportable entraîné
selon le protocole final. Les expériences montrent néanmoins que les effets
du transfert dépendent de l'initialisation, du scaler, de la fraction et du
domain shift. Elles ne démontrent ni la supériorité générale de Scratch ni
l'inutilité générale du Transfer Learning.

