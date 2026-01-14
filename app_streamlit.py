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
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
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
    
    # Image de comparaison globale
    st.markdown("## 📊 Comparaison Pré-crise vs Post-crise")
    st.image('OPTION1_barres_finales.png', use_container_width=True)
    
    # Statistiques descriptives
    st.markdown("## 📋 Statistiques Descriptives par Période")
    
    for periode in periode_filter:
        with st.expander(f"📋 {periode}"):
            stats = df_filtered[df_filtered['periode'] == periode][
                ['ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']
            ].describe()
            st.dataframe(stats, use_container_width=True)

# ============================================================================
# PAGE 3: ANALYSE STATISTIQUE
# ============================================================================

elif page == "🔬 Analyse Statistique":
    st.title("🔬 Analyse Statistique - T-test de Student")
    st.markdown("**Comparaison des variables financières: Pré-crise (2005-2010) vs Post-crise (2011-2015)**")
    
    st.markdown("""
    Cette analyse teste l'hypothèse que la crise financière de 2008 a entraîné des changements significatifs 
    dans le modèle d'affaires des banques coopératives européennes. Nous utilisons un t-test de Student pour 
    comparer les moyennes de chaque variable entre les deux périodes.
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
    st.image('OPTION1_barres_finales.png', use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("## Résumé des Résultats - 7 Variables Financières")
    
    # Tableau résumé simple
    summary_cols = ['Variable', 'Moyenne Pré-crise', 'Moyenne Post-crise', 
                   'Différence (%)', 'p-value', 'Significatif (p<0.05)']
    summary_df = tests_df[summary_cols].copy()
    summary_df['p-value'] = summary_df['p-value'].apply(lambda x: f"{x:.2e}")

    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown("## Tableau Complet des Tests Statistiques")
    st.markdown("*Cliquez ci-dessous pour voir tous les détails (t-statistic, Cohen's d, Intervalle de confiance, etc.)*")
    
    with st.expander("📊 Tableau Détaillé Complet", expanded=False):
        all_cols = tests_df.columns.tolist()
        detail_df = tests_df[all_cols].copy()
        # Formater la p-value en notation scientifique
        if 'p-value' in detail_df.columns:
            detail_df['p-value'] = detail_df['p-value'].apply(lambda x: f"{x:.2e}")
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown("## Résultats Visuels - Boxplots Interactifs")
    
    st.markdown("""
    **Explorez les distributions des 7 variables pour chaque période.**
    
    Hovrez sur les graphiques pour voir les détails statistiques.
    """)
    
    # Charger les données brutes pour les boxplots
    coop_df = pd.read_csv('Theme4_coop_zoom_data.xlsx - coop_zoom_data.csv')
    
    # Convertir les colonnes numériques
    variables = ['ass_total', 'ass_trade', 'inc_trade', 'in_roa', 'rt_rwa', 'in_roe', 'in_trade']
    for var in variables:
        coop_df[var] = pd.to_numeric(coop_df[var].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Créer les boxplots interactifs
    for var in variables:
        # Séparer pré-crise et post-crise
        pre_crisis = coop_df[coop_df['year'] <= 2010][var].dropna()
        post_crisis = coop_df[coop_df['year'] >= 2011][var].dropna()
        
        # Créer figure Plotly avec boxplots
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=pre_crisis,
            name='Pré-crise (2005-2010)',
            marker_color='#3498db',
            boxmean='sd'
        ))
        
        fig.add_trace(go.Box(
            y=post_crisis,
            name='Post-crise (2011-2015)',
            marker_color='#e74c3c',
            boxmean='sd'
        ))
        
        # Ajouter p-value en titre
        p_val = tests_df[tests_df['Variable'] == var]['p-value'].values[0]
        sig = "✓ Significatif" if p_val < 0.05 else "✗ Non-significatif"
        
        fig.update_layout(
            title=f"<b>{var.upper()}</b> - {sig} (p={p_val:.2e})",
            yaxis_title="Valeur",
            xaxis_title="Période",
            height=400,
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("## Résumé des Interprétations par Variable")
    
    for idx, row in tests_df.iterrows():
        var = row['Variable']
        p_val = row['p-value']
        diff_pct = row['Différence (%)']
        cohens_d = row["Cohen's d"]
        mean_pre = row['Moyenne Pré-crise']
        mean_post = row['Moyenne Post-crise']
        
        sig = "✅ OUI" if p_val < 0.05 else "❌ NON"
        direction = "Baisse" if diff_pct < 0 else "Hausse"
        
        with st.expander(f"{var} - {sig} Significatif"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Pré-crise:** {mean_pre:.4f}")
                st.write(f"**Post-crise:** {mean_post:.4f}")
                st.write(f"**Variation:** {diff_pct:.2f}% ({direction})")
            
            with col2:
                st.write(f"**t-statistic:** {row['t-statistic']:.4f}")
                st.write(f"**p-value:** {p_val:.2e}")
                st.write(f"**Cohen's d:** {cohens_d:.4f}")
            
            # Interprétation
            if p_val < 0.05:
                st.markdown(f"""
                **Conclusion:** Différence **SIGNIFICATIVE** (p < 0.05)
                
                La variation de {diff_pct:.2f}% n'est **pas due au hasard**. 
                Les banques ont changé significativement leur {var.lower()} après la crise.
                """)
            else:
                st.markdown(f"""
                **Conclusion:** Pas de différence significative (p ≥ 0.05)
                
                Bien que {var.lower()} ait varié de {diff_pct:.2f}%, cette différence pourrait être due au hasard.
                """)
    
    st.markdown("---")
    
    st.markdown("## Conclusion Générale")
    
    sig_count = len(tests_df[tests_df['p-value'] < 0.05])
    
    st.markdown(f"""
    **{sig_count} sur 7 variables** montrent des différences significatives entre pré-crise et post-crise.
    
    **Principaux constats:**
    - **Réduction drastique des actifs:** Baisse de 73.6% (très significative)
    - **Réduction des activités de trading:** Baisse de 75.9% 
    - **Détérioration de la rentabilité:** Baisse du ROA (-13.9%)
    - **Légère baisse du ratio de capital:** -2.2% (faible mais significative)
    
    Ces résultats confirment que la crise financière a fortement impacté le modèle d'affaires 
    des banques coopératives, particulièrement sur les activités de marché et la taille des actifs.
    """)

# ============================================================================
# PAGE 4: DÉTAIL DES CALCULS
# ============================================================================

elif page == "📐 Détail des Calculs":
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
        - **p-value:** {var_info['p-value']:.2e}
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
    calc_df = tests_df[summary_cols].copy()
    # Formater la p-value en notation scientifique
    calc_df['p-value'] = calc_df['p-value'].apply(lambda x: f"{x:.2e}")
    st.dataframe(calc_df, use_container_width=True, hide_index=True)
    
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
    st.title("🎯 Analyse de Clustering K-means")
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
