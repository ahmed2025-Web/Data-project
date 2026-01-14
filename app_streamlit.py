"""
APPLICATION WEB INTERACTIVE - ANALYSE BANQUES COOPÉRATIVES (VERSION SIMPLIFIÉE)
Streamlit App pour explorer les résultats de l'analyse
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG STREAMLIT
# ============================================================================

# ============================================================================
# CONFIG STREAMLIT + CSS PERSONNALISÉ
# ============================================================================

st.set_page_config(
 page_title="Analyse Banques Coopératives",
 page_icon="",
 layout="wide"
)

# Charger le CSS depuis un fichier séparé
def load_css(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f"<style>{f.read()}</style>"

st.markdown(load_css('styles.css'), unsafe_allow_html=True)

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv('Theme4_coop_zoom_data.xlsx - coop_zoom_data.csv')
    if 'Unnamed: 10' in df.columns:
        df = df.drop(columns=['Unnamed: 10'])
    
    num_cols = ['ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    df['periode'] = df['year'].apply(lambda x: 'Pré-crise' if x <= 2010 else 'Post-crise')
    return df

@st.cache_data
def load_results():
    tests = pd.read_csv('03_tests_statistiques_complets.csv')
    impacts = pd.read_csv('05_impacts_par_pays.csv')
    return tests, impacts

df = load_data()
df_clean = df[['institution_name', 'year', 'country_code', 'periode', 
 'ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']].dropna()

tests_df, impacts_df = load_results()

# ============================================================================
# BARRE LATÉRALE - NAVIGATION
# ============================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
 "Sélectionnez une page:",
 ["Accueil", "Tableau de bord", "Analyse Statistique", 
 "Détail des Calculs", "Analyse ACP", "Clustering", "Analyse par Pays"]
)

# ============================================================================
# PAGE 1: ACCUEIL
# ============================================================================

if page == "Accueil":
 st.title("Analyse des Banques Coopératives Européennes")
 st.markdown("*Impact de la crise financière 2008 sur le business model (2005-2015)*")
 
 col1, col2, col3 = st.columns(3)
 with col1:
 st.metric("Observations", f"{len(df_clean):,}")
 with col2:
 st.metric("Banques uniques", df['institution_name'].nunique())
 with col3:
 st.metric("Pays couverts", df['country_code'].nunique())
 
 st.markdown("---")
 
 st.markdown("## ❓ Problématique Centrale")
 st.markdown("""
 **Comment les banques coopératives européennes ont-elles modifié leur modèle d'affaires 
 suite à la crise financière de 2008 ?**
 """)
 
 st.markdown("## Questions Clés")
 
 col1, col2 = st.columns(2)
 
 with col1:
  st.markdown("""
  **1. Différences pré/post-crise ?**
 Oui - Toutes les variables statistiquement significatives (p < 0.05)
 
 **2. Changements observés ?**
 Réduction drastique: Actifs -73.6%, Trading -75.9%
 
 **3. Profils de banques ?**
 4 clusters avec stratégies distinctes
 """)
 
 with col2:
 st.markdown("""
 **4. Pays les plus affectés ?**
 Allemagne -72%, Italie -69%
 
 **5. Convergence entre banques ?**
 Non - Divergence des stratégies observée
 
 **6. Davantage de prudence ?**
 Oui - Ratio de capital (RWA) en baisse (-2.24%)
 """)
 
 st.markdown("---")
 
 st.markdown("## Approche Méthodologique")
 
 col1, col2, col3 = st.columns(3)
 
 with col1:
 st.markdown("""
 **Tests de Significativité (t-test)**
 - Comparer les moyennes pré et post-crise
 - Valider la significativité statistique
 - Mesurer la taille d'effet (Cohen's d)
 """)
 
 with col2:
 st.markdown("""
 **Segmentation par Clustering K-means**
 - Identifier des profils de banques
 - Analyser stratégies différenciées
 - Découvrir 4 groupes distincts
 """)
 
 with col3:
 st.markdown("""
 **Analyse en Composantes Principales (ACP)**
 - Réduire la dimensionnalité (7D → 2D)
 - Visualiser les profils de banques
 - Interpréter les corrélations variables
 """)
 
 col4, col5 = st.columns(2)
 
 with col4:
 st.markdown("""
 **ANOVA (Analyse de Variance)**
 - Comparer les moyennes entre clusters
 - Valider les différences inter-groupes
 - Quantifier l'effet du clustering
 """)
 
 with col5:
 st.markdown("""
 **Analyse Géographique par Pays**
 - Évaluer l'impact régional de la crise
 - Comparer les stratégies par zone
 - Identifier les comportements nationaux
 """)

# ============================================================================
# PAGE 2: TABLEAU DE BORD
# ============================================================================

elif page == "Tableau de bord":
 st.title(" Tableau de Bord Descriptif")
 
 col1, col2 = st.columns(2)
 
 with col1:
 periode_filter = st.multiselect(
 "Filtrer par période:",
 ["Pré-crise", "Post-crise"],
 default=["Pré-crise", "Post-crise"]
 )
 
 with col2:
 # Obtenir les 10 pays les plus représentés
 top_pays = df['country_code'].value_counts().head(10).index.tolist()
 pays_filter = st.multiselect(
 "Filtrer par pays (top 10):",
 top_pays,
 default=top_pays[:3]
 )
 
 # Filtrer les données
 df_filtered = df_clean[
 (df_clean['periode'].isin(periode_filter)) & 
 (df_clean['country_code'].isin(pays_filter))
 ]
 
 st.write(f"**Observations affichées:** {len(df_filtered):,}")
 
 # Graphiques
 st.markdown("## Distribution des Variables Clés")
 
 col1, col2 = st.columns(2)
 
 with col1:
 fig, ax = plt.subplots(figsize=(10, 5))
 df_filtered.boxplot(column='ass_total', by='periode', ax=ax)
 ax.set_title('Actifs Totaux (Millions €)')
 ax.set_xlabel('Période')
 plt.suptitle('')
 st.pyplot(fig, use_container_width=True)
 
 with col2:
 fig, ax = plt.subplots(figsize=(10, 5))
 df_filtered.boxplot(column='in_roa', by='periode', ax=ax)
 ax.set_title('Rentabilité (ROA)')
 ax.set_xlabel('Période')
 plt.suptitle('')
 st.pyplot(fig, use_container_width=True)
 
 # Statistiques descriptives
 st.markdown("## Statistiques Descriptives par Période")
 
 for periode in periode_filter:
 with st.expander(f" {periode}"):
 stats = df_filtered[df_filtered['periode'] == periode][
 ['ass_total', 'in_roa', 'rt_rwa', 'in_roe']
 ].describe()
 st.dataframe(stats, use_container_width=True)

# ============================================================================
# PAGE 3: ANALYSE STATISTIQUE
# ============================================================================

elif page == "Analyse Statistique":
 st.title(" Résultats des Tests Statistiques")
 st.markdown("Comparaison Pré-crise vs Post-crise (t-test de Student)")
 
 st.markdown("## Hypothèses du Test")
 st.markdown("""
 **H₀ (Hypothèse nulle):** 
 > Il n'existe PAS de différence significative entre les moyennes pré et post-crise
 > 
 > μ_pré-crise = μ_post-crise
 
 **H₁ (Hypothèse alternative):** 
 > Il existe une différence significative entre les moyennes
 > 
 > μ_pré-crise ≠ μ_post-crise
 
 **Seuil de significativité:** α = 0.05
 - Si **p-value < 0.05** → On rejette H₀ (différence **SIGNIFICATIVE** ✅)
 - Si **p-value ≥ 0.05** → On ne rejette pas H₀ (pas de preuve suffisante ❌)
 """)
 
 st.markdown("---")
 st.markdown("## Tableau Récapitulatif des Tests")
 
 # Afficher le tableau
 display_cols = ['Variable', 'Moyenne Pré-crise', 'Moyenne Post-crise', 
 'Différence (%)', 'p-value', "Cohen's d"]
 st.dataframe(tests_df[display_cols], use_container_width=True, hide_index=True)
 
 st.markdown("---")
 
 # Détail pour chaque variable
 st.markdown("## Analyse Détaillée par Variable")
 
 selected_var = st.selectbox(
 "Sélectionnez une variable:",
 tests_df['Variable'].tolist()
 )
 
 var_data = tests_df[tests_df['Variable'] == selected_var].iloc[0]
 
 col1, col2, col3, col4 = st.columns(4)
 with col1:
 st.metric("Moyenne Pré-crise", f"{var_data['Moyenne Pré-crise']:.4f}")
 with col2:
 st.metric("Moyenne Post-crise", f"{var_data['Moyenne Post-crise']:.4f}")
 with col3:
 st.metric("Variation %", f"{var_data['Différence (%)']:.2f}%")
 with col4:
 st.metric("p-value", f"{var_data['p-value']:.6f}")
 
 # Visualisation
 st.markdown("## Distribution Graphique")
 
 pre_data = df_clean[df_clean['periode'] == 'Pré-crise'][selected_var].dropna()
 post_data = df_clean[df_clean['periode'] == 'Post-crise'][selected_var].dropna()
 
 # Créer 2 graphes côte à côte pour meilleure visibilité
 col_graph1, col_graph2 = st.columns(2)
 
 # GRAPHE 1: Histogrammes côte à côte (side-by-side)
 with col_graph1:
 fig1, ax1 = plt.subplots(figsize=(8, 5))
 
 # Créer les bins compatibles
 bins = np.linspace(
 min(pre_data.min(), post_data.min()),
 max(pre_data.max(), post_data.max()),
 30
 )
 
 ax1.hist(pre_data, bins=bins, alpha=0.6, label='Pré-crise', color='#1f77b4', edgecolor='black', linewidth=0.5)
 ax1.hist(post_data, bins=bins, alpha=0.6, label='Post-crise', color='#ff7f0e', edgecolor='black', linewidth=0.5)
 ax1.set_xlabel(selected_var, fontsize=11)
 ax1.set_ylabel('Nombre de banques', fontsize=11)
 ax1.set_title(f'Histogrammes superposés: {selected_var}', fontweight='bold')
 ax1.legend(fontsize=10)
 ax1.grid(True, alpha=0.3, axis='y')
 st.pyplot(fig1, use_container_width=True)
 
 # GRAPHE 2: Comparaison des distributions (deux sous-histogrammes)
 with col_graph2:
 fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
 
 # Pré-crise
 ax2a.hist(pre_data, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=0.5)
 ax2a.set_ylabel('Pré-crise', fontsize=10, fontweight='bold')
 ax2a.grid(True, alpha=0.3, axis='y')
 ax2a.text(0.02, 0.95, f'n={len(pre_data):,}', transform=ax2a.transAxes, 
 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.3))
 
 # Post-crise
 ax2b.hist(post_data, bins=30, alpha=0.7, color='#ff7f0e', edgecolor='black', linewidth=0.5)
 ax2b.set_ylabel('Post-crise', fontsize=10, fontweight='bold')
 ax2b.set_xlabel(selected_var, fontsize=11)
 ax2b.grid(True, alpha=0.3, axis='y')
 ax2b.text(0.02, 0.95, f'n={len(post_data):,}', transform=ax2b.transAxes,
 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.3))
 
 fig2.suptitle(f'Distributions séparées: {selected_var}', fontsize=12, fontweight='bold', y=1.00)
 st.pyplot(fig2, use_container_width=True)

# ============================================================================
# PAGE 4: DÉTAIL DES CALCULS
# ============================================================================

elif page == "Détail des Calculs":
 st.title("Détail des Calculs")
 st.markdown("Formules et résultats des tests statistiques")
 
 st.markdown("## T-test: Pré-crise vs Post-crise")
 
 col1, col2 = st.columns(2)
 
 with col1:
 st.markdown("### Données Observées")
 var_selected = st.selectbox("Choisir variable:", tests_df['Variable'].tolist())
 
 var_info = tests_df[tests_df['Variable'] == var_selected].iloc[0]
 
 st.markdown(f"""
 **Pré-crise (n = {int(var_info['n_Pré-crise'])}):**
 - Moyenne (μ₁): {var_info['Moyenne Pré-crise']:.6f}
 - Écart-type (σ₁): {var_info['Écart-type Pré-crise']:.6f}
 - Erreur type: {var_info['Écart-type Pré-crise']/np.sqrt(var_info['n_Pré-crise']):.6f}
 
 **Post-crise (n = {int(var_info['n_Post-crise'])}):**
 - Moyenne (μ₂): {var_info['Moyenne Post-crise']:.6f}
 - Écart-type (σ₂): {var_info['Écart-type Post-crise']:.6f}
 - Erreur type: {var_info['Écart-type Post-crise']/np.sqrt(var_info['n_Post-crise']):.6f}
 
 **Différence observée:**
 - Δμ = μ₁ - μ₂ = {var_info['Moyenne Pré-crise'] - var_info['Moyenne Post-crise']:.6f}
 - IC 95% = [{var_info['IC 95% Lower']:.6f}, {var_info['IC 95% Upper']:.6f}]
 """)
 
 with col2:
 st.markdown("### Résultat du Test")
 st.markdown(f"""
 **Formule du t-test:**
 
 $$t = \\frac{{μ_1 - μ_2}}{{\\sqrt{{\\frac{{s_1^2}}{{n_1}} + \\frac{{s_2^2}}{{n_2}}}}}}$$
 
 **Où:**
 - μ₁, μ₂ = moyennes pré et post-crise
 - s₁, s₂ = écarts-types
 - n₁, n₂ = effectifs
 
 **Calcul Numérique:**
 - Δμ = {var_info['Moyenne Pré-crise']:.6f} - {var_info['Moyenne Post-crise']:.6f}
 - Δμ = {var_info['Moyenne Pré-crise'] - var_info['Moyenne Post-crise']:.6f}
 - SE = {var_info['Erreur Standard']:.6f}
 
 **Résultats Finaux:**
 - **t-statistique:** {var_info['t-statistic']:.6f}
 - **p-value:** {var_info['p-value']:.10f}
 - **Cohen's d:** {var_info["Cohen's d"]:.6f}
 - **Effet:** {var_info['Effet Size']}
 - **Conclusion:** {var_info['Significatif (p<0.05)']}
 
 ✅ **Interprétation:** La valeur p est {'INFÉRIEURE' if var_info['p-value'] < 0.05 else 'SUPÉRIEURE'} à 0.05
 """)
 
 st.markdown("---")
 st.markdown("### Résumé Statistique Complet")
 summary_cols = ['Variable', 'n_Pré-crise', 'Moyenne Pré-crise', 'Écart-type Pré-crise',
 'n_Post-crise', 'Moyenne Post-crise', 'Écart-type Post-crise',
 't-statistic', 'p-value', "Cohen's d", 'Effet Size']
 st.dataframe(tests_df[summary_cols], use_container_width=True, hide_index=True)
 
 st.markdown("---")
 
 st.markdown("## ANOVA: Comparaison des 4 Clusters")
 
 anova_df = pd.read_csv('10_anova_clusters.csv')
 
 st.markdown("""
 **Hypothèse nulle (H₀):** Les 4 clusters n'ont pas de différences significatives
 
 **Hypothèse alternative (H₁):** Au moins un cluster est significativement différent
 
 **Formule ANOVA:**
 
 $$F = \\frac{{MSB}}{{MSW}} = \\frac{{\\sum n_k(\\bar{x}_k - \\bar{x})^2 / (k-1)}}{{\\sum\\sum(x_{ki} - \\bar{x}_k)^2 / (N-k)}}$$
 
 Où:
 - MSB = Variance Between clusters
 - MSW = Variance Within clusters
 """)
 
 col1, col2 = st.columns(2)
 with col1:
 st.markdown("### Top 3 Résultats")
 top_anova = anova_df.nlargest(3, 'F-statistic')[['Variable', 'F-statistic', 'p-value']]
 st.dataframe(top_anova, use_container_width=True, hide_index=True)
 
 with col2:
 st.markdown("### Graphique F-statistiques")
 fig, ax = plt.subplots(figsize=(8, 4))
 ax.barh(anova_df['Variable'], anova_df['F-statistic'], color='steelblue')
 ax.set_xlabel('F-statistic')
 ax.set_title('F-statistiques ANOVA')
 ax.grid(True, alpha=0.3, axis='x')
 st.pyplot(fig, use_container_width=True)
 
 st.markdown("---")
 
 st.markdown("## 3️⃣ CORRÉLATION PEARSON: Assets vs Rentabilité")
 
 corr_df = pd.read_csv('11_correlations.csv')
 
 st.markdown("""
 **Formule de Pearson:**
 
 $$r = \\frac{{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}}{{\\sqrt{{\\sum(x_i - \\bar{x})^2}} \\cdot \\sqrt{{\\sum(y_i - \\bar{y})^2}}}}$$
 
 Interprétation:
 - r = 0: Pas de corrélation
 - 0 < r < 0.3: Faible corrélation
 - 0.3 < r < 0.7: Corrélation modérée
 - r > 0.7: Forte corrélation
 """)
 
 col1, col2 = st.columns(2)
 with col1:
 st.markdown("### Résultats")
 st.dataframe(corr_df, use_container_width=True, hide_index=True)
 
 with col2:
 st.markdown("### Visualisation")
 # Charger l'image si elle existe
 try:
 from PIL import Image
 img = Image.open('14_correlation_assets_roa.png')
 st.image(img, width='stretch')
 except:
 st.info("Graphique non disponible")
 
 st.markdown("---")
 
 st.markdown("## 4️⃣ SILHOUETTE SCORE: Qualité du Clustering")
 
 sil_df = pd.read_csv('12_silhouette_scores.csv')
 
 st.markdown(f"""
 **Silhouette Score moyen: {sil_df['Silhouette Score'].mean():.4f}**
 
 **Formule:**
 
 $$s_i = \\frac{{b(i) - a(i)}}{{max(a(i), b(i))}}$$
 
 Où:
 - a(i) = Distance moyenne à tous points du même cluster
 - b(i) = Distance moyenne à tous points du cluster plus proche
 
 **Interprétation:**
 - s = -1: Mauvais clustering
 - s = 0: Incertain
 - s = 1: Excellent clustering
 
 **Résultat:** {'Excellent ✅' if sil_df['Silhouette Score'].mean() > 0.5 else 'Bon ✅' if sil_df['Silhouette Score'].mean() > 0.3 else 'Acceptable'}
 """)
 
 col1, col2 = st.columns(2)
 with col1:
 st.markdown("### Scores par Cluster")
 st.dataframe(sil_df, use_container_width=True, hide_index=True)
 
 with col2:
 st.markdown("### Graphique")
 try:
 img = Image.open('15_silhouette_scores.png')
 st.image(img, width='stretch')
 except:
 st.info("Graphique non disponible")
 
 st.markdown("---")
 
 st.markdown("---")
 
 st.markdown("## 5️⃣ ANALYSE EN COMPOSANTES PRINCIPALES (ACP): Détails des Calculs")
 
 st.markdown("""
 **Objectif:** Réduire les 7 variables financières en 2 composantes principales tout en conservant le maximum d'information.
 
 **Formule:**
 
 Chaque PC est une combinaison linéaire des variables originales:
 
 $$PC_1 = w_{1,1} \\cdot x_1 + w_{1,2} \\cdot x_2 + ... + w_{1,7} \\cdot x_7$$
 
 Où w_{i,j} sont les **loadings** (contributions).
 """)
 
 try:
 acp_df = pd.read_csv('19_acp_details.csv')
 
 col1, col2 = st.columns(2)
 
 with col1:
 st.markdown("### Variance Expliquée")
 var_row = acp_df[acp_df['Element'] == 'Variance expliquée (%)'].iloc[0]
 st.markdown(f"""
 - **PC1:** {var_row['PC1']}
 - **PC2:** {var_row['PC2']}
 - **Total 2D:** {var_row['Total_2D']}
 """)
 
 with col2:
 st.markdown("### Valeurs Propres (Eigenvalues)")
 eigen_row = acp_df[acp_df['Element'] == 'Valeurs propres (variance)'].iloc[0]
 st.markdown(f"""
 - **λ₁:** {eigen_row['PC1']}
 - **λ₂:** {eigen_row['PC2']}
 - **Total:** {eigen_row['Total_2D']}
 """)
 
 st.markdown("---")
 
 st.markdown("### Loadings des Variables (Contributions)")
 st.markdown("Chaque coefficient montre comment la variable contribue à PC1 et PC2:")
 
 loadings_df = acp_df[acp_df['Element'].str.startswith('Loading_')].copy()
 loadings_df['Variable'] = loadings_df['Element'].str.replace('Loading_', '')
 loadings_df = loadings_df[['Variable', 'PC1', 'PC2']]
 
 st.dataframe(loadings_df, use_container_width=True, hide_index=True)
 
 st.markdown("""
 **Interprétation des Loadings:**
 - **Variables avec grand loading en PC1** (≈0.6): `ass_total`, `ass_trade`, `inc_trade`
 → PC1 = **Taille et activité de trading**
 
 - **Variables avec grand loading en PC2** (≈0.7): `in_roa`, `in_roe`
 → PC2 = **Rentabilité**
 
 - **Variables avec petit loading**: `rt_rwa`, `in_trade`
 → Peu d'importance dans les 2 principales composantes
 """)
 
 except Exception as e:
 st.warning(f"Fichier ACP details non disponible: {e}")
 
 st.markdown("---")
 
 st.markdown("## Code Python Utilisé")
 
 with st.expander(" Voir le code"):
 st.code("""
# T-test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(pre_crisis, post_crisis)

# ANOVA
from scipy.stats import f_oneway
f_stat, p_value = f_oneway(cluster0, cluster1, cluster2, cluster3)

# Corrélation Pearson
from scipy.stats import pearsonr
r, p_value = pearsonr(assets, roa)

# Silhouette Score
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, clusters)

# ACP
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Loadings (contributions)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
 """, language='python')

# ============================================================================
# PAGE 5: ANALYSE EN COMPOSANTES PRINCIPALES (ACP)
# ============================================================================

elif page == "Analyse ACP":
 st.title("Analyse en Composantes Principales")
 st.markdown("Réduction dimensionnelle pour visualisation et interprétation")
 
 st.markdown("## Objectif")
 st.markdown("""
 L'ACP permet de:
 - Visualiser les données multi-dimensionnelles en 2D
 - Identifier les directions de plus grande variance
 - Comprendre les corrélations entre variables
 - Valider la qualité du clustering en 2D
 """)
 
 st.markdown("## Variance Expliquée")
 
 col1, col2, col3 = st.columns(3)
 with col1:
 st.metric("PC1 Variance", "35.80%")
 with col2:
 st.metric("PC2 Variance", "20.44%")
 with col3:
 st.metric("Total", "56.24%")
 
 st.markdown("""
 **Interprétation:** Les 2 premières composantes principales capturent 56.24% de la variance totale.
 Cela signifie que nous retenons plus de la moitié de l'information en réduisant de 7D à 2D.
 """)
 
 st.markdown("---")
 
 st.markdown("## 1. Projection ACP - Clusters en 2D")
 st.markdown("Visualisation de chaque banque par ses 2 composantes principales, colorée par cluster")
 
 try:
 from PIL import Image
 img = Image.open('16_acp_clusters.png')
 st.image(img, width='stretch', caption='Clusters projetés sur les deux premières composantes principales')
 except:
 st.error("Graphique non disponible")
 
 st.markdown("---")
 
 st.markdown("## 2. Biplot - Contributions des Variables")
 st.markdown("Chaque flèche représente une variable et sa contribution aux composantes principales")
 
 try:
 img = Image.open('17_acp_biplot.png')
 st.image(img, width='stretch', caption='Biplot montrant la contribution de chaque variable aux PC1 et PC2')
 except:
 st.error("Graphique non disponible")
 
 st.markdown("""
 **Comment lire le biplot:**
 - **Longueur de la flèche** = importance de la variable
 - **Direction** = dans quelle composante elle contribue
 - **Flèches proches** = variables corrélées
 - **Flèches opposées** = variables anti-corrélées
 """)
 
 st.markdown("---")
 
 st.markdown("## 3. Variance Cumulée")
 st.markdown("Variance explicative en fonction du nombre de composantes utilisées")
 
 try:
 img = Image.open('18_acp_variance.png')
 st.image(img, width='stretch', caption='Variance cumulée en fonction du nombre de composantes')
 except:
 st.error("Graphique non disponible")
 
 st.markdown("""
 **Observations:**
 - Avec 4 composantes: ~85% de variance
 - Avec 5 composantes: ~95% de variance
 - Avec 7 composantes: 100% (toutes les variables)
 
 Le choix de k=2 ou k=3 représente un bon trade-off entre **visualisation** et **conservation d'information**.
 """)
 
 st.markdown("---")
 
 st.markdown("## Code Python")
 
 with st.expander(" Voir le code"):
 st.code("""
from sklearn.decomposition import PCA

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ACP
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Variance expliquée
print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")

# Contributions des variables
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
 """, language='python')

# ============================================================================
# PAGE 6: CLUSTERING
# ============================================================================

elif page == "Clustering":
 st.title(" Analyse de Clustering K-means")
 st.markdown("Identification de 4 profils de banques distincts")
 
 # Charger les résultats du clustering
 cluster_profiles = pd.read_csv('04_cluster_profiles.csv', index_col=0)
 
 # Charger les profils par période
 try:
 cluster_by_period = pd.read_csv('cluster_profiles_by_period.csv')
 has_period_data = True
 except:
 has_period_data = False
 
 st.markdown("## Profils Globaux")
 
 st.dataframe(cluster_profiles.round(4), use_container_width=True)
 
 st.markdown("---")
 
 st.markdown("## Comparaison Pré-crise vs Post-crise")
 
 if has_period_data:
 st.markdown("""
 Distribution des clusters avant et après la crise financière de 2008.
 Observe comment les banques se répartissent différemment selon la période.
 """)
 
 # Afficher le tableau complet
 display_cols = ['Période', 'Cluster', 'Nombre_banques', 'Pourcentage', 
 'ass_total_mean', 'in_roa_mean', 'in_roe_mean']
 display_data = cluster_by_period[display_cols].copy()
 display_data.columns = ['Période', 'Cluster', 'Nombre', '%', 'Actifs (moy)', 'ROA (moy)', 'ROE (moy)']
 
 st.dataframe(display_data, use_container_width=True, hide_index=True)
 
 # Visualisation de la distribution
 col1, col2 = st.columns(2)
 
 with col1:
 st.markdown("### Pré-crise (1,441 banques)")
 pre_data = cluster_by_period[cluster_by_period['Période'] == 'Pré-crise']
 fig, ax = plt.subplots(figsize=(8, 5))
 ax.pie(pre_data['Nombre_banques'], labels=pre_data['Cluster'], autopct='%1.1f%%',
 colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
 ax.set_title('Distribution des Clusters (Pré-crise)')
 st.pyplot(fig, use_container_width=True)
 
 with col2:
 st.markdown("### Post-crise (6,808 banques)")
 post_data = cluster_by_period[cluster_by_period['Période'] == 'Post-crise']
 fig, ax = plt.subplots(figsize=(8, 5))
 ax.pie(post_data['Nombre_banques'], labels=post_data['Cluster'], autopct='%1.1f%%',
 colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
 ax.set_title('Distribution des Clusters (Post-crise)')
 st.pyplot(fig, use_container_width=True)
 
 st.markdown("---")
 
 st.markdown("## Distribution des Clusters par Période")
 
 # Charger les clusters
 available_vars = ['ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']
 X = df_clean[available_vars].dropna()
 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(X)
 
 kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
 clusters = kmeans.fit_predict(X_scaled)
 df_test = df_clean[available_vars].notna().all(axis=1)
 df_clean.loc[df_test, 'cluster'] = clusters
 
 # Graphique distribution
 fig, ax = plt.subplots(figsize=(10, 5))
 cluster_pct = pd.crosstab(df_clean['periode'], df_clean['cluster'], normalize='index') * 100
 cluster_pct.plot(kind='bar', ax=ax)
 ax.set_title('Distribution des clusters par période (%)')
 ax.set_ylabel('Pourcentage (%)')
 ax.set_xlabel('Période')
 ax.legend(title='Cluster')
 st.pyplot(fig, use_container_width=True)

 st.markdown("---")
 
 st.markdown("## Clusters en Projection PCA")
 st.markdown("""
 Visualisation des 4 clusters dans l'espace des deux premières composantes principales.
 Chaque point représente une banque colorée selon son cluster d'appartenance.
 """)
 
 try:
 img = plt.imread('05_clusters_pca.png')
 st.image(img, caption="Clusters projetés sur les composantes principales")
 except:
 st.warning("Graphique de projection PCA non disponible")

 st.markdown("---")
 
 st.markdown("## Centroïdes Finales")
 st.markdown("""
 Positions finales des 4 centroïdes après convergence de l'algorithme K-means.
 Les croix colorées indiquent le centre de chaque cluster.
 """)
 
 try:
 img = plt.imread('20_kmeans_centroides_finales.png')
 st.image(img, caption="Position finale des 4 centroïdes")
 except:
 st.warning("Graphique des centroïdes non disponible")

 st.markdown("---")
 
 st.markdown("## Évolution des Centroïdes")
 st.markdown("""
 Déplacement des centroïdes au cours des itérations de l'algorithme:
 - **Gauche**: Itération 1 (positions initiales)
 - **Milieu**: Itération intermédiaire (mouvement des centroïdes)
 - **Droite**: Itération finale (convergence)
 - **Bas**: Zooms détaillés sur chaque phase
 """)
 
 try:
 img = plt.imread('21_kmeans_evolution_centroides.png')
 st.image(img, caption="Évolution des centroïdes")
 except:
 st.warning("Graphique d'évolution des centroïdes non disponible")

 st.markdown("---")
 
 st.markdown("## Profils Réels des Clusters")
 st.markdown("""
 Caractéristiques distinctives des 4 clusters basées sur les variables financières:
 
 - **C1** (8,124 institutions): Petites et moyennes banques avec profil équilibré
 - **C2** (108 institutions): Groupe affecté par la crise avec rentabilité dégradée
 - **C3** (2 institutions): Cas extrêmes avec revenus commerciaux négatifs
 - **C4** (15 institutions): Grandes banques du secteur coopératif
 """)
 
 try:
 img = plt.imread('22_centroides_variables_reelles.png')
 st.image(img, caption="Comparaison des variables financières par cluster")
 except:
 st.warning("Graphique des profils réels des centroïdes non disponible")

# ============================================================================
# PAGE 7: ANALYSE PAR PAYS
# ============================================================================

elif page == "Analyse par Pays":
 st.title(" Impact par Pays")
 st.markdown("Quel pays a été le plus affecté par la crise ?")
 
 st.markdown("## Variations par Pays")
 
 display_impacts = impacts_df[['Pays', 'Actifs Pré-crise (millions)', 
 'Actifs Post-crise (millions)', 'Variation (%)', 'Nb banques']].copy()
 display_impacts = display_impacts.sort_values('Variation (%)')
 
 st.dataframe(display_impacts, use_container_width=True, hide_index=True)
 
 st.markdown("---")
 
 st.markdown("## Impact Visuel")
 
 fig, ax = plt.subplots(figsize=(12, 8))
 colors = ['red' if x < 0 else 'green' for x in display_impacts['Variation (%)']]
 ax.barh(display_impacts['Pays'], display_impacts['Variation (%)'], color=colors, alpha=0.7)
 ax.set_xlabel('Variation des actifs (%)', fontsize=12)
 ax.set_title('Impact de la crise 2008 par pays', fontsize=14)
 ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
 st.pyplot(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
 Analyse des Banques Coopératives Européennes (2005-2015)<br>
 Données: 9,550 observations | Banques: 1,696 | Pays: 22
</div>
""", unsafe_allow_html=True)
