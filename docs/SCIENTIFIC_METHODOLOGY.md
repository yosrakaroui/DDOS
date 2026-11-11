# Méthodologie scientifique

## 1. Problématique

Une attaque DDoS surcharge une ressource en générant un volume ou un rythme
de trafic qui empêche le service normal. Le projet ne détecte pas directement
les paquets : il classe des flux CICFlowMeter décrits par des variables
statistiques.

CIC-IDS2017 fournit une source connue pour le pré-entraînement. CIC-DDoS2019
est la cible multiclasse. Le transfert est étudié parce qu'un backbone ayant
appris des régularités de trafic pourrait réduire le besoin d'annotations
cibles. La différence de distribution entre datasets constitue toutefois un
domain shift.

## 2. Représentation et modèle

Les 76 variables sont des caractéristiques numériques de flux : durées,
tailles, paquets, indicateurs TCP et statistiques d'inter-arrivée. Les
identifiants, IP, ports et timestamps sont exclus pour limiter le
shortcut learning et la fuite liée à l'identité.

Le MLP est :

```text
76 → 256 → 128 → 64 → 18
```

Le backbone correspond aux couches cachées. La tête finale produit un logit
par classe. Le softmax de l'inférence transforme les logits en probabilités.

## 3. Préprocessing

1. Conversion numérique ; les valeurs non numériques et non finies deviennent
   manquantes.
2. Imputation par médiane apprise sur le train.
3. Standardisation par moyenne et écart-type du train.
4. Clipping des valeurs standardisées dans `[-10, 10]`.

Le scaler ne doit jamais être ajusté sur validation, test ou fichier importé.
Dans l'expérience 2×2, l'ajustement du scaler cible sur tout `target/train`
avant le sous-échantillonnage supervisé est explicitement documenté comme une
adaptation non supervisée.

## 4. Classification et métriques

La classification binaire source distingue `BENIGN` de `DDoS`. La cible
possède 18 classes : `Benign`, `DrDoS_DNS`, `DrDoS_LDAP`, `DrDoS_MSSQL`,
`DrDoS_NTP`, `DrDoS_NetBIOS`, `DrDoS_SNMP`, `DrDoS_UDP`, `LDAP`, `MSSQL`,
`NetBIOS`, `Portmap`, `Syn`, `TFTP`, `UDP`, `UDP-lag`, `UDPLag`, `WebDDoS`.

- **Accuracy** : proportion totale de prédictions correctes.
- **Precision** : parmi les prédictions d'une classe, proportion correcte.
- **Recall** : parmi les exemples d'une classe, proportion retrouvée.
- **F1** : moyenne harmonique de précision et rappel.
- **Macro F1** : moyenne des F1 de classes, utile pour ne pas masquer les
  classes rares.
- **Weighted F1** : moyenne pondérée par les supports.
- **Matrice de confusion** : lignes vraies, colonnes prédites.

## 5. Protocole expérimental

Le train sert à ajuster le preprocessing et à entraîner. La validation sert
à l'early stopping et à choisir le meilleur état. Le test reste intouchable
jusqu'à l'évaluation finale.

La grille historique compare cinq stratégies :

- **Scratch** : paramètres aléatoires, tout le modèle entraîné.
- **Frozen** : backbone source gelé, seule la tête est entraînée.
- **Partial** : certains blocs restent gelés.
- **Fine-tuning** : tête puis modèle complet avec learning rate réduit.
- **Progressive** : dégel graduel des blocs.

Les fractions historiques sont 1 %, 5 %, 10 % et 100 %, avec trois seeds.
L'expérience contrôlée 2×2 compare initialisation aléatoire/pré-entraînée et
scaler cible/source en partageant les seeds, sous-ensembles et hyperparamètres.

## 6. Interprétation et limites statistiques

Les moyennes, écarts-types d'échantillon et IC à 95 % ont été recalculés pour
trois seeds avec `df=2` et `t=4.302653`. Les contrastes 2×2 sont appariés par
seed. Avec seulement trois seeds, ces intervalles sont larges et sensibles à
une seule seed ; aucun résultat n'est présenté comme preuve de significativité.

Les classes `UDPLag` et `WebDDoS` sont très difficiles et ont un support test
de 8. Le split aléatoire par lignes peut laisser des flux similaires dans
plusieurs partitions. La source binaire ne couvre pas la taxonomie cible.

