# 📋 Organisation de ta Présentation - T-test de Student

## 🎯 Vue d'ensemble de ta partie

Tu es responsable de l'**analyse statistique**: le **t-test de Student**.

---

## 📄 Page 1: HYPOTHÈSES ET THÉORIE (Calculs)

### Où mettre ça?
**Sur la page "📐 Détail des Calculs" de l'app Streamlit**

### Contenu à inclure:

```
ÉTAPE 1: POSER LES HYPOTHÈSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

H₀ (Hypothèse nulle):
   Les banques coopératives n'ont PAS changé significativement 
   entre la pré-crise et la post-crise
   
   Mathématiquement: μ_pré-crise = μ_post-crise

H₁ (Hypothèse alternative):
   Les banques coopératives ONT changé significativement
   
   Mathématiquement: μ_pré-crise ≠ μ_post-crise

Seuil de significativité: α = 0.05
   Si p-value < 0.05  → Rejeter H₀ (résultat significatif ✅)
   Si p-value ≥ 0.05  → Accepter H₀ (pas de preuve suffisante ❌)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ÉTAPE 2: LA FORMULE DU T-TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Formule pour comparer 2 groupes indépendants:

       μ₁ - μ₂
t = ─────────────────────────
    √(s₁²/n₁ + s₂²/n₂)

Où:
   μ₁, μ₂  = moyennes pré et post
   s₁, s₂  = écarts-types pré et post
   n₁, n₂  = nombre d'observations pré et post

Le résultat (t) suit une distribution de Student 
avec (n₁ + n₂ - 2) degrés de liberté

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ÉTAPE 3: COEFFICIENT D'EFFET (Cohen's d)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Formule:
        μ₁ - μ₂
d = ──────────────
    s_pooled

Interprétation (quelle est l'IMPORTANCE du changement?):
   |d| < 0.2   → Effet très petit (à peine détectable)
   0.2 ≤ |d| < 0.5  → Effet petit (faible importance)
   0.5 ≤ |d| < 0.8  → Effet moyen (importance modérée)
   |d| ≥ 0.8   → Effet grand (très important)

Pourquoi? 
   La p-value dit "est-ce significatif?"
   Cohen's d dit "est-ce IMPORTANT?"

Exemple:
   - Très grande n → t-test peut être sig. même pour petit effet
   - Cohen's d montre l'importance pratique du changement
```

---

## 📊 Page 2: RÉSULTATS DES TESTS (Tests Statistiques)

### Où mettre ça?
**Sur la page "🔬 Analyse Statistique" de l'app Streamlit**

### Contenu à inclure:

```
RÉSULTATS PAR VARIABLE
═════════════════════════════════════════════════════════

VARIABLE: ass_total (Actifs Totaux)
───────────────────────────────────

Données observées:
   Pré-crise (n=1,441):
      • Moyenne: 20,072.57 millions €
      • Écart-type: 123,071.16 (très variable!)
      
   Post-crise (n=6,808):
      • Moyenne: 5,295.17 millions €
      • Écart-type: 63,335.16

Test t:
   • Différence: 14,777.40 millions €
   • Variation: -73.6% ⚠️ (énorme réduction!)
   • t-statistique: 6.60
   • p-value: < 0.0001 *** (très significatif!)
   • Cohen's d: 0.19 (petit effet)

Conclusion:
   ✅ REJET DE H₀
   → Les actifs ont SIGNIFICATIVEMENT diminué
   → Les banques sont BEAUCOUP PLUS PETITES post-crise
   → C'est un changement majeur (mais effet statistique petit 
     car énorme variabilité dans les données)
   
Interprétation pour ton rapport:
   "Les banques coopératives ont réduit leurs actifs totaux 
    de 73.6% après la crise. Cette réduction est 
    statistiquement significative (p < 0.001), ce qui signifie
    qu'elle n'est pas due au hasard."

═════════════════════════════════════════════════════════
```

---

## 🎨 Page 3: VISUALISATIONS (Tests Statistiques)

### Où mettre ça?
**Sur la page "🔬 Analyse Statistique" → "📊 Distribution Graphique"**

### Graphe à montrer:

```
HISTOGRAMME COMPARATIF
━━━━━━━━━━━━━━━━━━━━━

Axe X: Valeur de la variable (ass_total, in_roa, etc.)
Axe Y: Nombre de banques

Deux courbes:
   • Bleu: Distribution pré-crise (n=1,441)
   • Orange: Distribution post-crise (n=6,808)

À regarder:
   - Les deux distributions se CHEVAUCHENT?
   - Elles sont DÉCALÉES (pré vers la droite, post vers la gauche)?
   - L'une est plus RESSERRÉE que l'autre?

Le graphe VISUALISE ce que le t-test dit mathématiquement!
```

---

## 🗂️ Structure complète recommandée pour l'app

### "🔬 Analyse Statistique" (PAGE PRINCIPALE)
```
├─ 📋 HYPOTHÈSES
│  ├─ H₀ (Hypothèse nulle)
│  ├─ H₁ (Hypothèse alternative)
│  └─ Seuil α = 0.05
│
├─ 📊 TABLEAU RÉSUMÉ
│  └─ Afficher: Variable, n, Moyennes, p-value, Cohen's d, Conclusion
│
├─ 🔍 DÉTAIL PAR VARIABLE
│  └─ Sélecteur dropdown → Affiche données + résultats
│
└─ 📈 GRAPHES
   └─ Histogramme pré/post pour la variable sélectionnée
```

### "📐 Détail des Calculs" (PAGE SECONDAIRE)
```
├─ 📚 THÉORIE
│  ├─ Formule du t-test
│  ├─ Conditions d'utilisation
│  └─ Cohen's d explication
│
├─ 🧮 CALCUL COMPLET (Exemple)
│  ├─ Données brutes
│  ├─ Calcul pas à pas
│  └─ Résultat final
│
└─ 🔗 TABLE DE COMPARAISON
   └─ Afficher le CSV complet avec tous les détails
```

---

## ⚠️ Points clés à expliquer

### 1. Pourquoi t-test ET ANOVA?
```
• t-test: Compare 2 groupes (pré vs post)
• ANOVA: Compare 4 groupes (clusters C1, C2, C3, C4)

Les deux sont complémentaires!
```

### 2. Pourquoi la p-value est si petite (< 0.0001)?
```
Raison 1: Les données sont TRÈS DIFFÉRENTES
          Moyenne pré: 20,072 €
          Moyenne post: 5,295 €
          Différence: 14,777 € (énorme!)

Raison 2: Les échantillons sont GRANDS
          n_pré = 1,441
          n_post = 6,808
          Plus d'observations = plus de précision
          
Raison 3: Les écarts-types sont GRANDS
          Ça rend le test plus facile à satisfaire
          (plus de variabilité = plus d'espace pour 
           une vraie différence)
```

### 3. Pourquoi Cohen's d est petit (0.19) si p-value est très petite?
```
⚠️ ATTENTION: C'est NORMAL!

Raison: 
   • p-value = "le résultat est-il dû au hasard?"
     → Réponse: NON, c'est réel (p < 0.05 ✅)
   
   • Cohen's d = "est-ce que le changement est GROS?"
     → Réponse: C'est moyen malgré la p-value petite
   
Pourquoi?
   • Les écarts-types sont ÉNORMES (123,000 pour ass_total!)
   • Cohen's d divise par l'écart-type
   • Donc même grand changement = petit Cohen's d

Analogie:
   Imagine une classe avec 30 élèves de tailles très variables
   (1m40 à 2m10). Le prof ajoute 10cm à tout le monde.
   C'est SIGNIFICATIF (réel) mais PETIT (Cohen's d petit).
```

---

## 📝 Ce que tu dois ÉCRIRE dans ton rapport

```
SECTION: ANALYSE STATISTIQUE - T-TEST DE STUDENT
═════════════════════════════════════════════════

1. INTRODUCTION
   Nous avons utilisé le t-test de Student pour comparer 
   les moyennes des variables pré-crise et post-crise...

2. HYPOTHÈSES
   H₀: Les moyennes sont égales (pas de changement)
   H₁: Les moyennes sont différentes (changement observé)

3. MÉTHODOLOGIE
   • Test bilatéral
   • Seuil de significativité: α = 0.05
   • Taille d'échantillon: pré-crise n=1,441, post-crise n=6,808

4. RÉSULTATS
   [Tableau complet avec p-values, Cohen's d, etc.]

5. INTERPRÉTATION
   Tous les résultats montrent p < 0.05, signifiant que 
   TOUS les changements sont statistiquement significatifs.
   
   Cependant, les tailles d'effet (Cohen's d) varient de 
   petit à grand, montrant que certains changements sont 
   plus importants que d'autres.

6. CONCLUSION
   Les banques coopératives ont modifié leur modèle d'affaires
   de manière SIGNIFICATIVE et DURABLE après la crise...
```

---

## ✅ Checklist pour ta présentation

- [ ] Hypothèses H₀ et H₁ clairement énoncées
- [ ] Seuil α = 0.05 expliqué
- [ ] Formule du t-test affichée
- [ ] Cohen's d interprétation complète
- [ ] Tableau résumé des résultats
- [ ] Au moins 2 graphiques de distribution
- [ ] P-values interprétées (< 0.05 = significatif)
- [ ] Conclusion claire pour chaque variable
- [ ] Distinction: p-value (signification) vs Cohen's d (importance)
- [ ] Sources/références bibliographiques

---

## 🎯 Durée estimée

- **Théorie** (hypothèses + formules): 2-3 minutes
- **Résultats** (tableau + discussion): 3-4 minutes
- **Conclusion** (ce que ça signifie): 1-2 minutes

**Total: 6-9 minutes** ← Adapte selon ton temps total!
