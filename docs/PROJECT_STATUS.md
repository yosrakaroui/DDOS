# État actuel du projet

## 1. Tableau de statut

| Fonctionnalité | État | Preuve | Travail restant |
|---|---|---|---|
| Préparation source/cible | Terminé | `data_report.json`, `data_report.md` | Vérifier la provenance amont |
| 76 features | Terminé | metadata préparées et `best_meta.json` | Tester davantage de formats CSV |
| 18 classes | Terminé | `best_meta.json`, évaluation finale | Surveiller les classes rares |
| 60 runs initiaux | Terminé | `experiments.csv`, `runs/` | Aucune modification |
| 48 runs contrôlés | Terminé | dossier `phase_2x2_scaler_initialization/` | Aucune modification |
| Audit 108 runs | Terminé | `final_audit_report.md` | Conclusions causales limitées |
| IC corrigés | Terminé | dossier `statistical_validation/` | Augmenter le nombre de seeds |
| Modèle final | Terminé | `models/best_model.pt`, `best_meta.json` | Revalidation périodique éventuelle |
| Tests unitaires/intégration | Terminé | `47 passed` | Ajouter tests UI navigateur |
| Streamlit | Implémenté | compilation et smoke HTTP `200 ok` | Tester un parcours utilisateur complet |
| Test réel CSV | À vérifier | aucune preuve persistée dédiée | Conserver un fixture anonymisé reproductible |
| Dashboard | Implémenté | `dashboard_validation.md` | Vérifier Plotly/browser sur machine de démo |
| Documentation projet | En cours | présents : docs historiques | Documents de cette série |
| Rapport PFE | Partiel | rapports scientifiques existants | Mise en forme finale et bibliographie |
| Présentation orale | Non vérifié | aucun artefact dans le repository | Préparer démonstration et limites |

## 2. Limites scientifiques

- Les miroirs locaux ne possèdent pas une provenance amont indépendante.
- Le split aléatoire par lignes peut favoriser des flux proches dans plusieurs
  partitions.
- Le faible support de `UDPLag` et `WebDDoS` rend leurs F1 instables.
- Le dataset source est binaire et limité au fichier DDoS du vendredi.
- Les trois seeds donnent des estimations incertaines.
- Le scaler et le domain shift influencent fortement la comparaison.

## 3. Intégrité

Cette documentation ne modifie aucun dataset, checkpoint, résultat historique
ou script. Les dix fichiers ajoutés sont exclusivement Markdown dans `docs/`.

