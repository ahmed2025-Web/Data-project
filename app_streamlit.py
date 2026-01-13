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
     "📐 Détail des Calculs", "🎯 Clustering", "🌍 Analyse par Pays"]
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
    
    st.markdown("## 🔍 Les 6 Réponses Clés")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        1️⃣ **Différences pré/post-crise ?**
        ✅ OUI - Toutes les variables significatives (p < 0.05)
        
        2️⃣ **Éléments changés ?**
        ⚠️ Actifs -73.6%, Trading -75.9%
        
        3️⃣ **Profils identifiés ?**
        4 clusters avec stratégies différentes
        """)
    
    with col2:
        st.markdown("""
        4️⃣ **Pays affectés ?**
        🇩🇪 Allemagne -72%, 🇮🇹 Italie -69%
        
        5️⃣ **Convergence ?**
        ❌ NON - Divergence observée
        
        6️⃣ **Plus prudentes ?**
        ✅ OUI - Ratio RWA baisse (-2.24%)
        """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Deux Méthodes Complémentaires")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Méthode 1: Tests t de Student**
        - Valider si changements significatifs
        - Mesurer taille d'effet (Cohen's d)
        - Résultat: Tous les changements validés
        """)
    
    with col2:
        st.markdown("""
        **✅ Méthode 2: Clustering K-means**
        - Découvrir profils de banques
        - Analyser stratégies différentes
        - Résultat: 4 clusters découverts
        """)

# ============================================================================
# PAGE 2: TABLEAU DE BORD
# ============================================================================

elif page == "📊 Tableau de bord":
    st.title("📊 Tableau de Bord Descriptif")
    
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
    st.markdown("## 📈 Distribution des Variables Clés")
    
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
    st.markdown("## 📋 Statistiques Descriptives par Période")
    
    for periode in periode_filter:
        with st.expander(f"📋 {periode}"):
            stats = df_filtered[df_filtered['periode'] == periode][
                ['ass_total', 'in_roa', 'rt_rwa', 'in_roe']
            ].describe()
            st.dataframe(stats, use_container_width=True)

# ============================================================================
# PAGE 3: ANALYSE STATISTIQUE
# ============================================================================

elif page == "🔬 Analyse Statistique":
    st.title("🔬 Résultats des Tests Statistiques")
    st.markdown("Comparaison Pré-crise vs Post-crise (t-test de Student)")
    
    st.markdown("## 📋 Tableau Récapitulatif des Tests")
    
    # Afficher le tableau
    display_cols = ['Variable', 'Moyenne Pré-crise', 'Moyenne Post-crise', 
                   'Différence (%)', 'p-value', "Cohen's d"]
    st.dataframe(tests_df[display_cols], use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Détail pour chaque variable
    st.markdown("## 🔍 Analyse Détaillée par Variable")
    
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
    st.markdown("## 📊 Distribution Graphique")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    pre_data = df_clean[df_clean['periode'] == 'Pré-crise'][selected_var].dropna()
    post_data = df_clean[df_clean['periode'] == 'Post-crise'][selected_var].dropna()
    
    ax.hist(pre_data, alpha=0.5, label='Pré-crise', bins=30)
    ax.hist(post_data, alpha=0.5, label='Post-crise', bins=30)
    ax.set_xlabel(selected_var)
    ax.set_ylabel('Fréquence')
    ax.set_title(f'Distribution de {selected_var}')
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# PAGE 4: DÉTAIL DES CALCULS
# ============================================================================

elif page == "📐 Détail des Calculs":
    st.title("📐 Détail des Calculs Mathématiques")
    st.markdown("Voir les formules et les calculs avec les vraies données")
    
    st.markdown("## 1️⃣ T-TEST DE STUDENT: Pré-crise vs Post-crise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Données Observées")
        var_selected = st.selectbox("Choisir variable:", tests_df['Variable'].tolist())
        
        var_info = tests_df[tests_df['Variable'] == var_selected].iloc[0]
        
        st.markdown(f"""
        **Pré-crise:**
        - Moyenne (μ₁): {var_info['Moyenne Pré-crise']:.6f}
        - Écart-type: (calculé)
        - n₁: 1,441 observations
        
        **Post-crise:**
        - Moyenne (μ₂): {var_info['Moyenne Post-crise']:.6f}
        - Écart-type: (calculé)
        - n₂: 6,808 observations
        """)
    
    with col2:
        st.markdown("### Résultat du Test")
        st.markdown(f"""
        **Formule du t-test:**
        
        $$t = \\frac{{μ_1 - μ_2}}{{\\sqrt{{\\frac{{s_1^2}}{{n_1}} + \\frac{{s_2^2}}{{n_2}}}}}}$$
        
        **Calcul:**
        - Δμ = {var_info['Moyenne Pré-crise']:.6f} - {var_info['Moyenne Post-crise']:.6f}
        - Δμ = {var_info['Moyenne Pré-crise'] - var_info['Moyenne Post-crise']:.6f}
        
        **Résultat:**
        - t-statistique: {var_info['t-statistic']:.6f}
        - p-value: {var_info['p-value']:.10f}
        - Significatif: ✅ {var_info['Significatif (p<0.05)']}
        """)
    
    st.markdown("---")
    
    st.markdown("## 2️⃣ ANOVA 1-WAY: Comparaison des 4 Clusters")
    
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
        """, language='python')

# ============================================================================
# PAGE 5: CLUSTERING
# ============================================================================

elif page == "🎯 Clustering":
    st.title("🎯 Analyse de Clustering K-means")
    st.markdown("Identification de 4 profils de banques distincts")
    
    # Charger les résultats du clustering
    cluster_profiles = pd.read_csv('04_cluster_profiles.csv', index_col=0)
    
    st.markdown("## 👥 Profils des Clusters")
    
    st.dataframe(cluster_profiles.round(4), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("## 📊 Distribution des Clusters par Période")
    
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

# ============================================================================
# PAGE 6: ANALYSE PAR PAYS
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
