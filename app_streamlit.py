"""
APPLICATION WEB INTERACTIVE - ANALYSE BANQUES COOPÉRATIVES (VERSION SIMPLIFIÉE)
Streamlit App pour explorer les résultats de l'analyse
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Analyse Banques Coopératives",
    page_icon="🏦",
    layout="wide"
)

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

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Sélectionnez une page:",
    ["🏠 Accueil", "📊 Tableau de bord", "🔬 Analyse Statistique", 
     "📐 Détail des Calculs", "📊 Analyse ACP", "🎯 Clustering", "🌍 Analyse par Pays"]
)

# ============================================================================
# PAGE 1: ACCUEIL
# ============================================================================

if page == "🏠 Accueil":
    st.title("🏦 Analyse des Banques Coopératives Européennes")
    st.markdown("*Impact de la crise financière 2008 sur le business model (2005-2015)*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📍 Observations", f"{len(df_clean):,}")
    with col2:
        st.metric("🏪 Banques uniques", df['institution_name'].nunique())
    with col3:
        st.metric("🌍 Pays couverts", df['country_code'].nunique())
    
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


elif page == "📊 Tableau de bord":
    st.title("📊 Tableau de Bord : Synthèse des Évolutions")
    st.markdown("Cette page présente les principaux constats visuels sur l'évolution des banques coopératives entre les deux périodes.")
    variables = ['ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']
    

    # --- SECTION 2 : VUE D'ENSEMBLE DES DISTRIBUTIONS ---
    st.header("2️⃣ État des lieux des distributions")
    st.markdown("""
    L'utilisation d'une échelle logarithmique permet de visualiser l'ensemble des variables, des plus volumineuses (actifs) 
    aux plus petites (rentabilité), sur un même plan. Les points blancs indiquent la position des moyennes.
    """)
    # Affichage de l'image boxplot log
    st.image('welch_justification_plot.png', caption="Distribution globale des variables (Échelle Logarithmique)", use_container_width=True)

    st.divider()

    # --- SECTION 3 : EXPLORATION DYNAMIQUE PAR PAYS ---
    st.header("3️⃣ Statistiques descriptives par zone")
    
    col1, col2 = st.columns(2)
    with col1:
        periode_filter = st.multiselect(
            "Choisir les périodes :",
            ["Pré-crise", "Post-crise"],
            default=["Pré-crise", "Post-crise"]
        )
    with col2:
        top_pays = df_clean['country_code'].unique().tolist()
        pays_filter = st.multiselect(
            "Filtrer par pays :",
            sorted(top_pays),
            default=top_pays[:3]
        )
    
    # Filtrage des données selon les choix de l'utilisateur
    df_filtered = df_clean[
        (df_clean['periode'].isin(periode_filter)) & 
        (df_clean['country_code'].isin(pays_filter))
    ]
    
    st.write(f"📈 **Nombre d'observations analysées :** {len(df_filtered):,}")
    
    # Tableaux de statistiques descriptives
    for periode in periode_filter:
        with st.expander(f"📋 Données chiffrées : {periode}"):
            stats_df = df_filtered[df_filtered['periode'] == periode][variables].describe().T
            st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)

# ============================================================================
# PAGE 3: ANALYSE STATISTIQUE
# ============================================================================


elif page == "🔬 Analyse Statistique":
    st.title("🔬 Analyse Statistique - Test t de Welch")
    st.markdown("**Comparaison des variables financières: Pré-crise (2005-2010) vs Post-crise (2011-2015)**")
    
    st.markdown("""
    Cette analyse teste l'hypothèse que la crise financière de 2008 a entraîné des changements significatifs 
    dans le modèle d'affaires des banques coopératives européennes. Nous utilisons un **test t de Welch** pour 
    comparer les moyennes. 
    
    *Note : Le test de Welch est privilégié ici car il est plus fiable que le test de Student classique lorsque les deux périodes présentent des volatilités (variances) différentes.*
    """)
    
    st.markdown("---")
    
    st.markdown("## Hypothèses du Test")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **H₀ (Hypothèse nulle):**
        
        Il n'existe PAS de différence significative entre les moyennes pré et post-crise.
        
        μ_pré-crise = μ_post-crise
        """)
    
    with col2:
        st.markdown("""
        **H₁ (Hypothèse alternative):**
        
        Il existe une différence significative entre les moyennes.
        
        μ_pré-crise ≠ μ_post-crise
        """)
    
    st.markdown("""
    **Seuil de significativité:** α = 0.05
    - Si p-value < 0.05 → On rejette H₀ ✅ **Différence significative**
    - Si p-value ≥ 0.05 → On ne rejette pas H₀ ❌ Pas de preuve suffisante
    """)
    
    st.markdown("---")
    
    st.markdown("## Vue d'Ensemble - Comparaison Visuelle")
    # Note: Assurez-vous que ce fichier existe ou remplacez par votre graphique actuel
    if os.path.exists('04_boxplots_statistiques.png'):
        st.image('04_boxplots_statistiques.png', use_container_width=True)
    else:
        st.info("Visualisation globale issue des fichiers d'analyse.")
    

    
    st.markdown("---")
    
    st.markdown("## Résumé des Interprétations par Variable")
    
    for idx, row in tests_df.iterrows():
        var = row['Variable']
        p_val = row['p-value']
        diff_pct = row['Différence (%)']
        
        sig_label = "✅ OUI" if p_val < 0.05 else "❌ NON"
        direction = "Baisse" if diff_pct < 0 else "Hausse"
        
        with st.expander(f"{var} - {sig_label} Significatif"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Moyenne Pré:** {row['Moyenne Pré-crise']:.4f}")
                st.write(f"**Moyenne Post:** {row['Moyenne Post-crise']:.4f}")
                st.write(f"**Variation:** {diff_pct:.2f}% ({direction})")
            with col2:
                st.write(f"**t-statistic:** {row['t-statistic']:.4f}")
                st.write(f"**p-value:** {p_val:.2e}")
                st.write(f"**Cohen's d:** {row['Cohen\'s d']:.4f}")
            
            if p_val < 0.05:
                st.markdown(f"**Conclusion:** Différence **SIGNIFICATIVE**. Le changement est structurel.")
            else:
                st.markdown(f"**Conclusion:** Pas de différence significative. Les variations peuvent être dues à la volatilité des données.")

    st.markdown("---")
    st.header(" Conclusion de l'Analyse")
    
    # Calcul dynamique pour le résumé
    n_sig = len(tests_df[tests_df['p-value'] < 0.05])
    
    st.success(f"**Synthèse :** Sur les {len(tests_df)} variables analysées, **{n_sig} présentent une différence statistiquement significative**.")
    
    st.markdown(f"""
    L'analyse via le **test t de Welch** confirme que la crise de 2008 a provoqué une rupture majeure. 
    Les résultats sont robustes car ils tiennent compte de la forte disparité des données observée sur le graphique des distributions.
    
    * **Modèle d'affaires :** La baisse massive des actifs totaux (-73.6%) et de trading (-75.9%) démontre un changement structurel profond des banques coopératives.
    * **Fiabilité :** Le rejet de l'hypothèse nulle ($H_0$) pour la majorité des variables indique que ces évolutions ne sont pas dues au hasard.
    """)
    
    st.info("💡 Cette analyse fournit une base solide pour affirmer que le secteur a entamé une phase de 'deleveraging' durable après 2010.")
# ============================================================================
# PAGE 4: DÉTAIL DES CALCULS
# ============================================================================

elif page == "📐 Détail des Calculs":
    st.title("Détail des Calculs")
    st.markdown("Formules et résultats du **Test t de Welch** (Variances Inégales)")
    
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
        
        **Post-crise (n = {int(var_info['n_Post-crise'])}):**
        - Moyenne (μ₂): {var_info['Moyenne Post-crise']:.6f}
        - Écart-type (σ₂): {var_info['Écart-type Post-crise']:.6f}
        
        **Différence observée:**
        - Δμ = {var_info['Moyenne Pré-crise'] - var_info['Moyenne Post-crise']:.6f}
        - IC 95% = [{var_info['IC 95% Lower']:.6f}, {var_info['IC 95% Upper']:.6f}]
        """)
    
    with col2:
        st.markdown("### Résultat du Test de Welch")
        st.markdown(f"""
        **Formule utilisée :**
        
        $$t = \\frac{{μ_1 - μ_2}}{{\\sqrt{{\\frac{{s_1^2}}{{n_1}} + \\frac{{s_2^2}}{{n_2}}}}}}$$
        
        *Note : Contrairement au t-test de Student, le test de Welch ne suppose pas que les variances des deux groupes soient égales. Chaque groupe est traité séparément, ce qui rend le test plus robuste quand les variances diffèrent.*
        
        **Résultats Finaux:**
        - **t-statistique:** {var_info['t-statistic']:.6f}
        - **p-value:** {var_info['p-value']:.2e}
        - **Cohen's d:** {var_info["Cohen's d"]:.6f}
        - **Conclusion:** {var_info['Significatif (p<0.05)']}
        """)
    
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
    
    st.markdown("## 📝 Code Python Utilisé")
    
    with st.expander("🔍 Voir le code"):
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

elif page == "📊 Analyse ACP":
    st.title("Analyse en Composantes Principales")
    st.markdown("Réduction dimensionnelle pour résumer les modèles d'affaires bancaires")
    
    st.markdown("## Objectif")
    st.markdown("""
    L'Analyse en Composantes Principales (ACP) est utilisée pour résumer l'information contenue dans 
    plusieurs indicateurs financiers et analyser les différences de business model des banques 
    coopératives européennes entre 2005 et 2015.
    """)
    
    st.markdown("---")
    
    st.markdown("## Variables Utilisées")
    st.markdown("""
    L'ACP repose sur des variables représentant :
    
    - **Taille et activité:** ass_total, ass_trade, inc_trade
    - **Rentabilité:** in_roa, in_roe
    - **Risque et structure financière:** rt_rwa, in_trade
    
    Ces variables couvrent les dimensions clés du modèle bancaire en combinant des indicateurs de taille, 
    d'activité de marché, de rentabilité et de risque. Elles permettent ainsi d'analyser conjointement 
    les choix stratégiques des banques coopératives, leur performance économique et leur degré 
    d'exposition aux activités risquées, dans un cadre synthétique adapté à la comparaison pré et post-crise.
    """)
    
    st.markdown("---")
    
    st.markdown("## Variance Expliquée")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PC1 Variance", "35.7%")
    with col2:
        st.metric("PC2 Variance", "20.8%")
    with col3:
        st.metric("Cumul PC1+PC2", "56.5%")
    
    st.markdown("""
    La première composante principale (PC1) explique environ 35,7 % de la variance totale et la 
    seconde (PC2) environ 20,8 %. Les deux premières composantes cumulent ainsi près de 56,5 % 
    de l'information contenue dans les 7 variables originales. Ce niveau de variance expliquée 
    est suffisant pour une analyse en composantes principales, car il permet de résumer efficacement 
    la structure globale des données tout en conservant l'essentiel des relations entre les variables. 
    La projection sur le plan (PC1, PC2) offre donc une représentation fiable des principales 
    différences entre les banques.
    """)
    
    st.markdown("---")
    
    st.markdown("## Visualisation de la Variance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Variance par Composante")
        try:
            from PIL import Image
            img = Image.open('ACP_Graph1.png')
            st.image(img, use_container_width=True)
        except:
            st.info("Graphique ACP_Graph1.png non disponible")
    
    with col2:
        st.markdown("### Variance Cumulée")
        try:
            img = Image.open('ACP_Graph2.png')
            st.image(img, use_container_width=True)
        except:
            st.info("Graphique ACP_Graph2.png non disponible")
    
    st.markdown("---")
    
    st.markdown("## Projection des Banques")
    st.markdown("""
    La projection des banques sur le plan PC1–PC2 montre une forte concentration autour de l'origine, 
    correspondant à des banques de taille moyenne. Quelques établissements apparaissent très éloignés 
    sur PC1, traduisant des banques de grande taille ou fortement orientées vers le trading.
    
    La période post-crise présente moins de profils extrêmes, suggérant une réduction des stratégies 
    les plus risquées après 2008.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Projection Pré-crise")
        try:
            img = Image.open('ACP_Graph3.png')
            st.image(img, use_container_width=True)
        except:
            st.info("Graphique ACP_Graph3.png non disponible")
    
    with col2:
        st.markdown("### Projection Interactive par Pays")
        
        # Sélection des pays
        all_countries = sorted(df_clean['country_code'].unique())
        selected_countries = st.multiselect(
            "Sélectionnez les pays à afficher:",
            all_countries,
            default=df_clean['country_code'].value_counts().head(8).index.tolist()
        )
        
        if selected_countries:
            # Préparer les données pour l'ACP
            available_vars = ['ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']
            X = df_clean[available_vars].dropna()
            
            # Standardisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # ACP
            pca = PCA(n_components=2)
            scores = pca.fit_transform(X_scaled)
            
            # DataFrame avec résultats
            scores_df = pd.DataFrame(scores, columns=["PC1", "PC2"])
            scores_df['country_code'] = df_clean[available_vars].notna().all(axis=1)
            scores_df['country_code'] = df_clean.loc[df_clean[available_vars].notna().all(axis=1), 'country_code'].values
            
            # Filtrer par pays sélectionnés
            scores_filtered = scores_df[scores_df['country_code'].isin(selected_countries)]
            
            # Graphique
            fig, ax = plt.subplots(figsize=(10, 7))
            for country in selected_countries:
                country_data = scores_filtered[scores_filtered['country_code'] == country]
                ax.scatter(country_data['PC1'], country_data['PC2'], 
                          label=country, alpha=0.6, s=50)
            
            pc1_var = pca.explained_variance_ratio_[0] * 100
            pc2_var = pca.explained_variance_ratio_[1] * 100
            
            ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=11)
            ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=11)
            ax.set_title('Projection ACP - Sélection de Pays', fontweight='bold', fontsize=12)
            ax.legend(title='Pays', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Veuillez sélectionner au moins un pays")
    
    st.markdown("---")
    
    st.markdown("## Interprétation des Axes")
    
    st.markdown("### Biplot - Contributions des Variables")
    st.markdown("""
    Le premier axe principal (PC1) est principalement associé à la taille du bilan et à l'intensité 
    des activités de trading, comme le montrent les fortes contributions des variables ass_total, 
    ass_trade et inc_trade. Il reflète un gradient allant des banques de petite taille, peu actives 
    sur les marchés, vers des établissements plus importants et davantage orientés vers les activités 
    de marché.
    
    Le second axe (PC2) est dominé par les indicateurs de rentabilité, notamment in_roa et in_roe. 
    Il permet de distinguer les banques selon leur capacité à générer des performances économiques, 
    indépendamment de leur taille ou de leur niveau d'activité.
    
    Ces deux axes mettent ainsi en évidence une opposition entre une logique de volume et d'exposition 
    aux marchés financiers, et une logique de performance économique, offrant une lecture synthétique 
    des stratégies bancaires.
    """)
    
    try:
        img = Image.open('ACP_Graph5.png')
        st.image(img, use_container_width=True, caption='Biplot montrant la contribution de chaque variable')
    except:
        st.info("Graphique ACP_Graph5.png non disponible")
    
    st.markdown("---")
    
    st.markdown("## Conclusion")
    st.markdown("""
    L'ACP met en évidence deux dimensions majeures du business model des banques coopératives :
    
    1. **Taille et intensité du trading** (axe PC1)
    2. **Rentabilité économique** (axe PC2)
    
    Après la crise financière de 2008, les banques semblent s'orienter vers des modèles plus prudents, 
    avec une réduction des comportements extrêmes, tout en conservant une forte hétérogénéité de performance.
    """)

# ============================================================================
# PAGE 6: CLUSTERING
# ============================================================================

elif page == "🎯 Clustering":
    st.title("🎯 Clustering & Results (K-means + extensions)")
    st.markdown(
        "This section summarizes the clustering workflow and the key results. "
        "Detailed explanations are provided in the written report."
    )

    @st.cache_data
    def load_cluster_profile():
        return pd.read_csv("data-cluster/cluster_profile.csv")

    cluster_profile = load_cluster_profile()

    @st.cache_data
    def load_kmeans_metrics():
        return pd.read_csv("data-cluster/kmeans_metrics.csv")

    metrics = load_kmeans_metrics()
    sil = float(metrics.loc[metrics["metric"]=="silhouette","value"].iloc[0])

    @st.cache_data
    def load_df_bp():
        return pd.read_csv("data-cluster/df_bp_with_clusters.csv")

    df_bp = load_df_bp()

    trans = None
    res = None  # table with k, silhouette, ARI_median_vs_seed0, etc.

    cluster_features = [
        "trading_intensity_assets",
        "in_trade",
        "rt_rwa",
        "in_roa",
        "in_roe",
    ]

    @st.cache_data
    def load_cluster_period_distribution():
        return pd.read_csv("data-cluster/cluster_period_distribution.csv")

    cluster_period_distribution = load_cluster_period_distribution()

    @st.cache_data
    def load_cluster_profile():
        return pd.read_csv("data-cluster/cluster_profile.csv")

    cluster_profile = load_cluster_profile()

    @st.cache_data
    def load_transitions():
        return pd.read_csv("data-cluster/transitions.csv")

    trans = load_transitions()

    @st.cache_data
    def load_transitions_matrix():
        return pd.read_csv("data-cluster/transition_matrix.csv")

    trans_matrix = load_transitions()

    @st.cache_data
    def load_transitions_with_size():
        return pd.read_csv("data-cluster/transitions_with_size.csv")

    trans_with_size = load_transitions_with_size()

    # ------------------------------------------------------------------------
    # Tabs to keep the page clean
    # ------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1) Data & features",
        "2) K selection (elbow/silhouette)",
        "3) K-means results",
        "4) Pre/Post & transitions",
        "5) Robustness",
        "6) GMM & Logistic model"
    ])

    # ------------------------------------------------------------------------
    # TAB 1 — Data & features
    # ------------------------------------------------------------------------
    with tab1:
        st.subheader("Dataset & preprocessing (results)")
        st.caption("This page reports the prepared outputs from the analysis notebook. No preprocessing is recomputed inside the app.")

        # ---- Load prepared outputs (already cleaned / computed elsewhere)
        @st.cache_data
        def load_csv(path):
            return pd.read_csv(path)

        #raw_preview = load_csv("./data-cluster/raw_preview.csv")
        #missing_summary = load_csv("./data-cluster/missing_summary.csv")
        df_bp_features = load_csv("./data-cluster/df_bp.csv")

        st.markdown(
             "- The raw dataset contains bank-year observations (2005–2015) with balance-sheet and performance variables.\n"
            "- We cleaned the data in the analysis pipeline (export artefact column, numeric conversion, plausibility checks).\n"
            "- We created the final clustering feature table at bank–period level (pre/post) using median aggregation." \
            "- Total Assets are used later for size-weighted transition analysis since size is not relevant for business model clustering."
        )
        st.dataframe(df_bp_features.head(30), use_container_width=True)

    # ------------------------------------------------------------------------
    # TAB 2 — K selection (elbow & silhouette)
    # ------------------------------------------------------------------------
    from pathlib import Path

    with tab2:
        st.subheader("Choosing k: Elbow method")
        st.markdown(
            "**Elbow method (inertia plot).** "
            "For each value of *k*, we run K-means and compute the *inertia* (within-cluster sum of squared distances). "
            "Inertia always decreases when k increases, because adding clusters makes groups tighter. "
            "We choose *k* near the *elbow point*: the value after which the decrease becomes much smaller "
            "(diminishing returns), providing a good trade-off between model simplicity and fit."
        )

        elbow_path = Path("assets") / "elbow_kmeans.png"
        if elbow_path.exists():
            st.image(str(elbow_path), use_container_width=True)
            st.caption("Elbow plot: inertia decreases with k; we select k near the elbow for a good trade-off.")
        else:
            st.warning("Missing file: assets/elbow_kmeans.png. Export it from the notebook and place it in the assets folder.")
        
        st.subheader("Choosing k: Silhouette analysis")
        st.markdown(
            "**Silhouette analysis (score vs k).** "
            "For each value of *k*, we compute the average silhouette score, which measures how well each observation "
            "matches its own cluster compared to the nearest other cluster. "
            "Values close to **1** indicate well-separated and cohesive clusters, values near **0** suggest overlap, "
            "and negative values indicate likely misclassification. "
            "We select *k* where the silhouette score is relatively high while keeping the solution interpretable "
            "(here, k=4 provides a good balance)."
        )


        sil_path = Path("assets") / "silhouette_kmeans.png"
        if sil_path.exists():
            st.image(str(sil_path), use_container_width=True)
            st.caption("Silhouette score vs k. We select k where the score is relatively high and the solution remains interpretable.")
        else:
            st.warning("Missing file: assets/silhouette_kmeans.png. Export it from the notebook and place it in the assets folder.")
    # ------------------------------------------------------------------------
    # TAB 3 — K-means results
    # ------------------------------------------------------------------------
    with tab3:
        st.subheader("PCA 2D projection")
        st.caption("2D visualization of the clustered observations (PCA on the scaled features).")

        img_path = Path("assets") / "pca_kmeans.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
             st.warning("Missing image: assets/pca_kmeans.png. Save it from the notebook and copy it into the assets folder.")

        st.subheader("Cluster profiles (mean feature values)")
        st.dataframe(cluster_profile.round(4), use_container_width=True)
        st.caption("Average values of the clustering features for each cluster (K-means, k=4).")

        st.metric("Silhouette score (k=4)", f"{sil:.3f}")
        st.caption("Higher values indicate more separated clusters; values near 0 indicate overlap.")

        st.markdown(
            "**Cluster interpretation (business model types):**\n"
            "- **Cluster 0 – Market-exposed but stable:** high trading assets intensity, but trading income remains low; profitability is relatively solid.\n"
            "- **Cluster 1 – Trading-oriented:** trading income is much more important; profitability is lower and risk is moderate–high.\n"
            "- **Cluster 2 – Traditional but high-risk:** low trading exposure, higher RWA ratio; returns remain positive but risk is elevated.\n"
            "- **Cluster 3 – Distressed:** weak performance with negative profitability indicators; small group of underperforming banks.\n"
        )


        period_counts = df_bp["period"].value_counts(dropna=False)

        col1, col2 = st.columns([1,2])
        with col1:
            st.dataframe(period_counts.to_frame(name="count"))
        with col2:
            fig, ax = plt.subplots()
            ax.bar(period_counts.index.astype(str), period_counts.values)
            ax.set_xlabel("Period")
            ax.set_ylabel("Count")
            ax.set_title("Bank–period observations")
            st.pyplot(fig, use_container_width=True)

        st.markdown(
            "After observing how the cluster composition changes between the pre- and post-crisis periods, "
            "we now analyze the clusters **within each period** to understand how the underlying business model "
            "profiles (trading intensity, risk, and profitability) evolved after the financial crisis."
        )




    # ------------------------------------------------------------------------
    # TAB 4 — Pre/Post & transitions
    # ------------------------------------------------------------------------
    with tab4:
        st.subheader("Cluster profiles: pre vs post crisis")
        st.markdown("### Cluster profiles by period (mean features)")

        cluster_features = [
            "trading_intensity_assets",
            "in_trade",
            "rt_rwa",
            "in_roa",
            "in_roe",
        ]

        missing = [c for c in cluster_features if c not in df_bp.columns]
        if missing:
            st.error(f"Missing clustering feature columns in df_bp: {missing}")
            st.stop()

        profile_period = df_bp.groupby(["cluster", "period"])[cluster_features].mean()
        st.dataframe(profile_period.round(4), use_container_width=True)
        st.caption("Mean feature values for each (cluster, period) group. This highlights how business model profiles change from pre to post crisis.")
        st.caption(
            "This comparison highlights how each cluster’s average trading exposure, risk (RWA ratio), and profitability "
            "shifted from pre- to post-crisis. We focus on changes in levels and direction (increase/decrease) rather than "
            "small numerical differences."
        )

        st.subheader("Pre vs post: cluster composition")
        ct = pd.crosstab(df_bp["cluster"], df_bp["period"], normalize="columns")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Table")
            st.dataframe(ct.round(4), use_container_width=True)
        with col2:
            st.markdown("### Bar chart")
            fig, ax = plt.subplots()
            ct.plot(kind="bar", ax=ax)
            ax.set_ylabel("Proportion")
            ax.set_title("Cluster composition: pre vs post")
            ax.legend(title="period")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            st.pyplot(fig, use_container_width=True)

        st.caption(
            "This chart shows how the **relative weight of each business model (cluster)** changes from pre- to post-crisis. "
            "The main shift is the sharp increase of **Cluster 1 (trading-oriented)** in the post-crisis period, "
            "while Clusters 0 and 2 remain the two dominant profiles overall."
        )


        st.subheader("Transitions: counts matrix")
        st.caption("Transition counts from pre-crisis cluster (rows) to post-crisis cluster (columns).")

        # build transition matrix (counts)
        T_counts = pd.crosstab(trans["cluster_pre"], trans["cluster_post"])

        st.dataframe(T_counts, use_container_width=True)

        st.subheader("Transition probability matrix (pre → post)")
        st.caption("Row i, column j = P(cluster_post = j | cluster_pre = i).")

        # Probability matrix
        T_prob = pd.crosstab(trans["cluster_pre"], trans["cluster_post"], normalize="index")

        # Overall stability (share of banks that stay in the same cluster)
        stability = (trans["cluster_pre"] == trans["cluster_post"]).mean()

        # Probability to stay in the same cluster, by starting cluster
        stay_by_cluster = pd.Series({c: T_prob.loc[c, c] for c in T_prob.index}).sort_index()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Probabilities table")
            st.dataframe(T_prob.round(4), use_container_width=True)

            st.markdown("### Stability (overall)")
            st.metric("Share staying in the same cluster", f"{stability:.3f}")

            st.markdown("### Stay probability by starting cluster")
            st.dataframe(stay_by_cluster.to_frame(name="P(stay)").round(4), use_container_width=True)
            st.caption("Diagonal elements of the transition matrix: P(cluster_post = cluster_pre | cluster_pre).")

        with col2:
            st.markdown("### Heatmap")
            fig, ax = plt.subplots()
            im = ax.imshow(T_prob.values)

            ax.set_xticks(range(T_prob.shape[1]))
            ax.set_xticklabels(T_prob.columns)
            ax.set_yticks(range(T_prob.shape[0]))
            ax.set_yticklabels(T_prob.index)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("P(cluster_post | cluster_pre)")

            ax.set_xlabel("cluster_post")
            ax.set_ylabel("cluster_pre")
            ax.set_title("Transition probability matrix (Markov)")

            st.pyplot(fig, use_container_width=True)

        st.markdown(
            "**Conclusion.** Overall stability is high (about 74% of banks remain in the same cluster), "
            "suggesting strong persistence in business models. However, the transition matrix also highlights "
            "a clear reallocation across types: Cluster 2 shows a non-negligible shift toward Cluster 1, "
            "while Cluster 3 appears much less stable, with banks frequently moving to other clusters."
        )


        st.subheader("Banks by size class")

        counts = trans_with_size["size_class"].value_counts().reindex(["small", "medium", "large"])
        st.dataframe(counts.to_frame(name="count"), use_container_width=True)

        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Size class")
        ax.set_ylabel("Number of banks")
        ax.set_title("Number of banks by size class")
        st.pyplot(fig, use_container_width=True)

        st.subheader("Transition matrices by size class")
        st.caption("Row-normalized probabilities: P(cluster_post | cluster_pre), computed separately for small/medium/large banks.")

        # Build probability matrices by size class
        T_by_size = {}
        for s in ["small", "medium", "large"]:
            sub = trans[trans["size_class"] == s]
            T_by_size[s] = pd.crosstab(sub["cluster_pre"], sub["cluster_post"], normalize="index")

        # Layout: 3 columns
        c1, c2, c3 = st.columns(3)
        cols = {"small": c1, "medium": c2, "large": c3}

        for s in ["small", "medium", "large"]:
            mat = T_by_size[s]
            with cols[s]:
                st.markdown(f"### {s.capitalize()} banks")

                # show matrix table (optional but useful)
                st.dataframe(mat.round(3), use_container_width=True)

                # heatmap (your same imshow logic)
                fig, ax = plt.subplots()
                im = ax.imshow(mat.values)

                ax.set_xticks(range(mat.shape[1]))
                ax.set_xticklabels(mat.columns)
                ax.set_yticks(range(mat.shape[0]))
                ax.set_yticklabels(mat.index)

                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("P(post | pre)")

                ax.set_xlabel("cluster_post")
                ax.set_ylabel("cluster_pre")
                ax.set_title(f"Transitions ({s})")

                st.pyplot(fig, use_container_width=True)

        st.markdown(
            "**Size-stratified transitions.** When we compute the transition matrices separately for small, medium, and large banks, "
            "the dynamics are not identical across sizes. Small banks appear more **inertial** (higher diagonal probabilities), "
            "while medium and especially large banks show a **stronger reallocation** across clusters. "
            "In particular, the key shift from **Cluster 2 → Cluster 1** (toward a more trading-oriented profile) is much more pronounced "
            "for medium/large banks than for small ones. "
            "This suggests that **size may be an important driver** of the transition probability.\n\n"
            "For this reason, we next estimate a **logistic regression** model where the dependent variable is the 2→1 transition, "
            "and the explanatory variable is bank size (pre-crisis total assets)."
        )


        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report

        st.subheader("Logistic regression: transition probability 2 → 1")
        st.caption("We model the probability of switching from cluster 2 (pre) to cluster 1 (post) using bank size (pre).")

        # -----------------------------
        # Build dataset
        # -----------------------------
        data_2 = trans[trans["cluster_pre"] == 2].copy()
        data_2["y_2_to_1"] = (data_2["cluster_post"] == 1).astype(int)

        # feature: log size
        data_2["log_size_pre"] = np.log(data_2["ass_total_pre"].clip(lower=1e-6))

        X = data_2[["log_size_pre"]]
        y = data_2["y_2_to_1"]

        # -----------------------------
        # Train/test + fit
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)

        # predicted probabilities on test
        proba = clf.predict_proba(X_test)[:, 1]

        # -----------------------------
        # 1) Logistic curve plot (P vs size)
        # -----------------------------
        st.markdown("### Predicted probability vs size")

        x_min = float(data_2["log_size_pre"].min())
        x_max = float(data_2["log_size_pre"].max())
        x_grid = np.linspace(x_min, x_max, 300).reshape(-1, 1)
        p_grid = clf.predict_proba(x_grid)[:, 1]

        fig, ax = plt.subplots()
        ax.scatter(data_2["log_size_pre"], data_2["y_2_to_1"], alpha=0.25)
        ax.plot(x_grid, p_grid)

        ax.set_xlabel("log(ass_total_pre)")
        ax.set_ylabel("P(2→1)")
        ax.set_title("Estimated probability of transition 2→1 vs size (pre)")
        ax.set_ylim(-0.05, 1.05)
        st.pyplot(fig, use_container_width=True)

        # -----------------------------
        # 2) Coefficients (under the curve)
        # -----------------------------
        intercept = float(clf.intercept_[0])
        coef = float(clf.coef_[0, 0])

        st.markdown("### Model coefficients")
        st.write(f"**Intercept:** {intercept:.4f}")
        st.write(f"**Coefficient (log_size_pre):** {coef:.4f}")
        st.caption("A positive coefficient means larger banks have higher predicted probability of making the 2→1 transition.")

        # -----------------------------
        # 3) Confusion matrix (after coefficients)
        # -----------------------------
        st.markdown("### Confusion matrix")

        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
        pred = (proba >= threshold).astype(int)

        cm = confusion_matrix(y_test, pred)
        cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("### Classification report")
        st.code(classification_report(y_test, pred))

        # -----------------------------
        # 4) ROC curve + AUC (last)
        # -----------------------------
        st.markdown("### ROC curve")

        auc = roc_auc_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        st.metric("ROC AUC", f"{auc:.3f}")


        st.markdown(
            "**Logistic model results.** The estimated coefficient on `log_size_pre` is **positive**, meaning that larger banks "
            "have a higher predicted probability of switching from Cluster 2 to Cluster 1 after the crisis. "
            "In terms of predictive performance, the ROC AUC is around **0.73**, which indicates a reasonable ability to rank banks "
            "by their likelihood of making the transition (better than random). "
            "However, with a standard 0.5 decision threshold the model has **high accuracy driven by class 0**, while the **recall for the transition class (1)** "
            "is low, reflecting class imbalance and the fact that the transition is relatively rare. "
            "Overall, the logistic regression supports the interpretation that **size is a meaningful driver** of the 2→1 shift, "
            "even if classification depends strongly on the chosen threshold."
        )




    # ------------------------------------------------------------------------
    # TAB 5 — Robustness
    # ------------------------------------------------------------------------
    with tab5:
        st.subheader("Robustness checks (stability across random seeds)")
        st.markdown(
            "**Robustness check.** Unsupervised methods can be sensitive to random initialization and to the choice of *k*. "
            "To ensure that our results are not an artefact of a particular run, we evaluate **stability** by re-running K-means "
            "with different random seeds and comparing the obtained partitions using the **Adjusted Rand Index (ARI)**. "
            "We also compute stability across a range of k values to verify that the selected solution is both **interpretable** and **consistent**."
        )


        @st.cache_data
        def load_ari_vs_ref():
            return pd.read_csv("data-cluster/ari_vs_ref.csv")

        @st.cache_data
        def load_stability_by_k():
            return pd.read_csv("data-cluster/stability_by_k.csv")

        ari_df = load_ari_vs_ref()
        res = load_stability_by_k()

        st.markdown("### ARI stability for k=4 (50 random seeds)")

        ari_vals = ari_df["ari_vs_ref"].dropna().values

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("ARI (min)", f"{ari_vals.min():.3f}")
            st.metric("ARI (median)", f"{np.median(ari_vals):.3f}")
            st.metric("ARI (mean)", f"{ari_vals.mean():.3f}")
            st.metric("ARI (max)", f"{ari_vals.max():.3f}")

        with col2:
            fig, ax = plt.subplots()
            ax.hist(ari_vals, bins=30)
            ax.set_xlim(0.95, 1.0)
            ax.set_xlabel("ARI vs reference run (seed=0)")
            ax.set_ylabel("Frequency")
            ax.set_title("K-means label stability across random seeds (zoom)")
            st.pyplot(fig, use_container_width=True)

        st.markdown(
            "**Adjusted Rand Index (ARI)** measures the agreement between two clusterings of the same dataset. "
            "It is **1** when the partitions are identical, and close to **0** when the agreement is similar to random. "
            "Here we compute ARI between the reference run (seed=0) and other random initializations."
            "ARI values are consistently very high (median ≈ 0.98), meaning that the cluster assignments barely change across random seeds. "
        )


        st.markdown("### Stability vs k (silhouette + ARI)")

        st.dataframe(res.round(4), use_container_width=True)
        st.caption("For each k, we report silhouette score and stability measured as ARI vs the seed=0 solution.")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.plot(res["k"], res["silhouette"], marker="o")
            ax.set_xlabel("k")
            ax.set_ylabel("Silhouette")
            ax.set_title("Silhouette vs k")
            st.pyplot(fig, use_container_width=True)

        with col2:
            fig, ax = plt.subplots()
            ax.plot(res["k"], res["ARI_median_vs_seed0"], marker="o")
            ax.set_xlabel("k")
            ax.set_ylabel("Median ARI vs seed0")
            ax.set_title("Stability vs k (median ARI)")
            st.pyplot(fig, use_container_width=True)

        st.markdown(
            "We jointly evaluate two criteria across different values of *k*: "
            "**silhouette score**, to measure separation/compactness of clusters, and "
            "**stability**, measured as the median **ARI** across multiple random seeds (agreement with the seed=0 run). "
            "A good choice of *k* should be both reasonably well-separated and stable.\n\n"
            "The silhouette curve peaks around **k=4**, suggesting the best balance of cluster quality "
            "among the tested values. The stability plot shows that **k=4 is also highly stable** (ARI close to 1), "
            "whereas some alternative k values (notably around **k=5**) are less stable. "
            "Overall, these robustness checks support keeping **k=4** as the main K-means specification."
        )


    # ------------------------------------------------------------------------
    # TAB 6 — GMM & Logistic model
    # ------------------------------------------------------------------------
    
    @st.cache_data
    def load_kmeans_vs_gmm():
        return pd.read_csv("data-cluster/kmeans_vs_gmm_crosstab.csv", index_col=0)

    ct_km_gmm = load_kmeans_vs_gmm()
    with tab6:
        st.subheader("Gaussian Mixture Model (GMM)")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Model selection (BIC)")
            bic_path = Path("assets") / "gmm_bic.png"
            if bic_path.exists():
                st.image(str(bic_path), use_container_width=True)
            else:
                st.warning("Missing image: assets/gmm_bic.png")

        with col2:
            st.markdown("### Why GMM?")
            st.write(
                "**Why we also use a Gaussian Mixture Model (GMM).** "
                "K-means assigns each observation to exactly one cluster and implicitly favors spherical, equally-sized groups. "
                "A **GMM** is a complementary approach: it performs **probabilistic (soft) clustering**, allowing observations to belong "
                "to clusters with different probabilities, and it can capture **elliptical/overlapping** cluster shapes. "
                "Using both methods helps us check whether the identified business model types are robust to a different clustering assumption."    
            )
            st.info("In our analysis, we select k = 4 for comparability with K-means.")

        st.markdown("### GMM clusters in 2D (PCA projection)")

        col1, col2 = st.columns(2)

        with col1:
            pca_path = Path("assets") / "gmm_pca.png"
            if pca_path.exists():
                st.image(str(pca_path), use_container_width=True)
                st.caption("Points colored by GMM cluster (k=4) in PCA 2D space.")
            else:
                st.warning("Missing image: assets/gmm_pca.png")

        with col2:
            ell_path = Path("assets") / "gmm_ellipses.png"
            if ell_path.exists():
                st.image(str(ell_path), use_container_width=True)
                st.caption("GMM components with covariance ellipses (2σ) projected into PCA space.")
            else:
                st.warning("Missing image: assets/gmm_ellipses.png")
        st.markdown("### Agreement with K-means clusters")
        st.dataframe(ct_km_gmm.round(4), use_container_width=True)
        st.caption("Row-normalized cross-tabulation: how each K-means cluster maps to GMM components.")

        st.markdown(
            "The GMM (k=4) produces partially overlapping groups, which is expected with probabilistic (soft) clustering. "
            "The agreement table shows strong consistency for some clusters: the K-means distressed group (cluster 3) is perfectly recovered by one GMM component (100%), "
            "and the trading-oriented group (cluster 2) is also mostly mapped to a single GMM cluster (~93%). "
            "In contrast, the two large non-distressed K-means clusters (0 and 1) are split across multiple GMM components, suggesting overlap and less sharp boundaries between these business model types."
        )


# ============================================================================
# PAGE 7: ANALYSE PAR PAYS
# ============================================================================

elif page == "🌍 Analyse par Pays":
    st.title("🌍 Impact par Pays")
    st.markdown("Quel pays a été le plus affecté par la crise ?")
    
    st.markdown("## 📊 Variations par Pays")
    
    display_impacts = impacts_df[['Pays', 'Actifs Pré-crise (millions)', 
                                   'Actifs Post-crise (millions)', 'Variation (%)', 'Nb banques']].copy()
    display_impacts = display_impacts.sort_values('Variation (%)')
    
    st.dataframe(display_impacts, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown("## 📈 Impact Visuel")
    
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
    📊 Analyse des Banques Coopératives Européennes (2005-2015)<br>
    Données: 9,550 observations | Banques: 1,696 | Pays: 22
</div>
""", unsafe_allow_html=True)
