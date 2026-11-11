# Protocole expérimental

1. **Un seul split**, défini une fois (`seed` de la config) et partagé par toutes les stratégies.
2. **Test intouchable** : il n'est utilisé ni pour l'early stopping, ni pour Optuna, ni pour choisir une stratégie.
3. **Seeds multiples** (`experiments.seeds`) : toujours rapporter moyenne ± écart-type, jamais un run isolé.
4. **Métrique principale : Macro F1** ; accuracy seulement en indicateur secondaire ; analyse par classe et matrice de confusion.
5. **Étude low-data** : mêmes stratégies avec 10 %, 50 % et 100 % du train cible
   (échantillon stratifié, minimum 5 flux par classe).
6. **Baseline équitable** : `scratch` a exactement la même architecture, le même optimiseur et le même budget d'epochs.

## Hypothèses

- H1 : Transfer Learning > from scratch (Macro F1).
- H2 : l'écart augmente quand les données cibles diminuent.
- H3 : Fine-tuning partiel ou progressif > Freezing total.

Un résultat négatif (pas de gain, transfert négatif) reste un résultat valide s'il est analysé.

## Points de vigilance liés aux données

- Retirer les colonnes d'identification avant d'apprendre (IP, ports, identifiant de flux, horodatage).
- Si les fichiers correspondent à des jours différents, envisager `split.mode: by_file` et le justifier ; vérifier d'abord la répartition des classes par fichier (notebook 01).
- Une classe absente du train mais présente dans le test fait forcément chuter le Macro F1 : à documenter.
- Les rapports `data_report.json` et `data_report.md` documentent le nettoyage,
  l'alignement canonique, les exclusions et les contrôles de fuite avant entraînement.
## Prétraitement et scalers

Le scaler source est ajusté uniquement sur le train CIC-IDS2017. Les
stratégies de transfert (`frozen`, `partial`, `finetune`,
`progressive`) utilisent ce scaler pour les trois splits CIC-DDoS2019.
La baseline `scratch` utilise un scaler distinct, ajusté uniquement sur
le train cible. Les deux protocoles sont conservés et identifiés dans
les métadonnées (`scaler_fit_dataset`, `scaler_fit_split`,
`scaler_mean`, `scaler_scale` et `preprocessing_strategy`).
