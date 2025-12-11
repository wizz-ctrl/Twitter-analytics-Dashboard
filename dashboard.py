import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Twitter Analytics Dashboard",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme - 3 shades of green for parties (PMLN, PPP, PTI)
PARTY_COLORS = {
    'PTI': '#628B61',
    'PPP': '#C7E1BA', 
    'PMLN': '#9CB770'
}

# Colors for languages - diverse colorful palette
LANGUAGE_COLORS = {
    'english': '#628B61',
    'urdu': '#9CB770',
    'japanese': '#C7E1BA',
    'roman-urdu': "#559FD8",
    'hindi': "#DCFF8A",
    'arabic': '#9B59B6',
    'punjabi': '#F39C12',
    'indonesian': '#1ABC9C',
    'french': '#E91E63',
    'portuguese': '#FF5722',
    'thai': '#00BCD4',
    'sindhi': '#8E44AD'
}

BACKGROUND_COLOR = '#0d1f0d'
CARD_BACKGROUND = '#1a2e1a'
TEXT_COLOR = '#FFFFFF'

# Only these languages to show
ALLOWED_LANGUAGES = ['english', 'urdu', 'japanese', 'sindhi', 'portuguese', 'hindi', 
                     'arabic', 'punjabi', 'roman-urdu', 'indonesian', 'french', 'thai']

# Custom CSS for dark theme with custom fonts
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Rajdhani:wght@400;500;600&display=swap');
    
    .stApp {{
        background-color: {BACKGROUND_COLOR};
    }}
    .stSidebar {{
        background-color: {CARD_BACKGROUND};
    }}
    .stSidebar .stMarkdown, .stSidebar label {{
        color: {TEXT_COLOR} !important;
        font-family: 'Rajdhani', sans-serif !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_COLOR} !important;
        font-family: 'Rajdhani', sans-serif !important;
    }}
    p, span, label, div {{
        color: {TEXT_COLOR} !important;
        font-family: 'Rajdhani', sans-serif !important;
    }}
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stDateInput label {{
        color: {TEXT_COLOR} !important;
        font-family: 'Rajdhani', sans-serif !important;
    }}
    .main-title {{
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        color: #9CB770 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 1rem !important;
        letter-spacing: 2px;
    }}
    .stSelectbox, .stMultiSelect {{
        font-family: 'Rajdhani', sans-serif !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Main Title with Magneto-style font (using Orbitron as web alternative)
st.markdown('<h1 class="main-title">🇵🇰 Pakistan\'s Political Discourse Analytics</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_parquet('dashboard_scatter_data.parquet')
    
    # Rename columns to standard names
    column_mapping = {
        'party_source': 'party',
        'detected_language': 'language',
        'hour_of_day': 'hour'
    }
    df = df.rename(columns=column_mapping)
    
    # Clean language data - fix the **language** issue
    if 'language' in df.columns:
        # Remove ** from language names
        df['language'] = df['language'].str.replace(r'\*\*', '', regex=True)
        # Convert to lowercase for consistency
        df['language'] = df['language'].str.lower().str.strip()
        # Map common variations
        language_mapping = {
            'portugese': 'portuguese',
            'roman urdu': 'roman-urdu',
        }
        df['language'] = df['language'].replace(language_mapping)
    
    # Convert created_at to datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        if 'hour' not in df.columns:
            df['hour'] = df['created_at'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['created_at'].dt.day_name()
        df['date'] = df['created_at'].dt.date
    
    return df

# Load data
df = load_data()

# Sidebar filters
st.sidebar.title("🎛️ Filters")

# Initialize filter variables
selected_party = 'All'
selected_language = 'All'
date_range = None

# Party filter
if 'party' in df.columns:
    parties = ['All'] + sorted(df['party'].dropna().unique().tolist())
    selected_party = st.sidebar.selectbox("Select Party", parties)

# Language filter - only allowed languages
if 'language' in df.columns:
    available_langs = [l for l in ALLOWED_LANGUAGES if l in df['language'].unique()]
    languages = ['All'] + sorted(available_langs)
    selected_language = st.sidebar.selectbox("Select Language", languages)

# Date range filter
if 'created_at' in df.columns:
    min_date = df['created_at'].min().date()
    max_date = df['created_at'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

# Apply filters
filtered_df = df.copy()

if 'party' in df.columns and selected_party != 'All':
    filtered_df = filtered_df[filtered_df['party'] == selected_party]

if 'language' in df.columns and selected_language != 'All':
    filtered_df = filtered_df[filtered_df['language'] == selected_language]

if date_range is not None and len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['created_at'].dt.date >= date_range[0]) & 
        (filtered_df['created_at'].dt.date <= date_range[1])
    ]

# Filter to only allowed languages for visualizations
filtered_df_lang = filtered_df[filtered_df['language'].isin(ALLOWED_LANGUAGES)]

# Layout configuration - DISABLE scroll zoom to fix infinite loop
plot_config = {
    'displayModeBar': True,
    'displaylogo': False,
    'scrollZoom': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}

layout_template = {
    'paper_bgcolor': CARD_BACKGROUND,
    'plot_bgcolor': CARD_BACKGROUND,
    'font': {'color': TEXT_COLOR, 'size': 12},
}

# Day order for charts
day_order = ['Saturday', 'Friday', 'Tuesday', 'Monday', 'Sunday', 'Thursday', 'Wednesday']
day_order_heatmap = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']

# ==================== VISUALIZATION 1: Tweet Count by Hour of Day (Heatmap) ====================
st.subheader("Tweet Count by Hour of the Day")

# Heatmap-specific filters
heatmap_col1, heatmap_col2, heatmap_col3 = st.columns([2, 2, 6])
with heatmap_col1:
    heatmap_party_filter = st.selectbox("Party (Heatmap)", ['All'] + sorted(df['party'].dropna().unique().tolist()), key='heatmap_party')
with heatmap_col2:
    heatmap_days = st.multiselect("Days", day_order_heatmap, default=day_order_heatmap, key='heatmap_days')

# Apply heatmap-specific filters
heatmap_df = filtered_df.copy()
if heatmap_party_filter != 'All':
    heatmap_df = heatmap_df[heatmap_df['party'] == heatmap_party_filter]
if heatmap_days:
    heatmap_df = heatmap_df[heatmap_df['day_of_week'].isin(heatmap_days)]

if 'hour' in heatmap_df.columns and 'day_of_week' in heatmap_df.columns:
    heatmap_data = heatmap_df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
    
    for h in range(24):
        if h not in heatmap_pivot.columns:
            heatmap_pivot[h] = 0
    heatmap_pivot = heatmap_pivot[sorted(heatmap_pivot.columns)]
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order_heatmap if d in heatmap_pivot.index])
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=[str(c) for c in heatmap_pivot.columns],
        y=heatmap_pivot.index,
        colorscale=[[0, '#E8F5E9'], [0.25, '#C7E1BA'], [0.5, '#9CB770'], [0.75, '#628B61'], [1, '#2E5A2E']],
        hovertemplate='Hour: %{x}<br>Day: %{y}<br>Tweet Count: %{z}<extra></extra>',
        showscale=True,
        text=heatmap_pivot.values.astype(int),
        texttemplate='%{text}',
        textfont={'size': 9, 'color': '#1a1a2e'}
    ))
    
    fig_heatmap.update_layout(
        **layout_template,
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=320,
        margin=dict(l=100, r=20, t=30, b=50),
        xaxis=dict(side='top', tickfont=dict(color=TEXT_COLOR), title_font=dict(color=TEXT_COLOR)),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font=dict(color=TEXT_COLOR))
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True, config=plot_config)

st.markdown("---")

# ==================== VISUALIZATION 2: Tweet Count by Day and Party (Ribbon Chart) ====================
st.subheader("Tweet Count by Day and Party")

# Ribbon chart 1 specific filters
ribbon1_col1, ribbon1_col2, ribbon1_col3 = st.columns([2, 2, 6])
with ribbon1_col1:
    ribbon1_parties = st.multiselect("Parties", ['PMLN', 'PPP', 'PTI'], default=['PMLN', 'PPP', 'PTI'], key='ribbon1_parties')
with ribbon1_col2:
    ribbon1_days = st.multiselect("Days", day_order, default=day_order, key='ribbon1_days')

# Apply ribbon1-specific filters
ribbon1_df = filtered_df.copy()
if ribbon1_days:
    ribbon1_df = ribbon1_df[ribbon1_df['day_of_week'].isin(ribbon1_days)]

if 'day_of_week' in ribbon1_df.columns and 'party' in ribbon1_df.columns:
    daily_party = ribbon1_df.groupby(['day_of_week', 'party']).size().reset_index(name='count')
    
    # Pivot to get parties as columns
    pivot_data = daily_party.pivot(index='day_of_week', columns='party', values='count').fillna(0)
    pivot_data = pivot_data.reindex([d for d in ribbon1_days if d in pivot_data.index])
    
    # Rank each party per day (1 = highest value)
    ranks = pivot_data.rank(ascending=False, axis=1, method='first')
    
    # Sort parties based on ranking at each time period
    sorted_parties_per_day = ranks.apply(lambda row: row.sort_values().index.tolist(), axis=1)
    
    # Build stacked y-values based on rankings
    stacked_values = {}
    for day in pivot_data.index:
        order = sorted_parties_per_day.loc[day]
        cumulative = 0
        stacked_values[day] = {}
        for party in order:
            stacked_values[day][party] = (cumulative, cumulative + pivot_data.loc[day, party])
            cumulative += pivot_data.loc[day, party]
    
    # Build ribbons with Plotly - TRUE Power BI style with flat sections
    fig_ribbon1 = go.Figure()
    
    parties_list = ribbon1_parties if ribbon1_parties else ['PMLN', 'PPP', 'PTI']
    x_days = pivot_data.index.tolist()
    n_days = len(x_days)
    
    # Parameters for flat sections and transitions
    flat_width = 0.44  # Wide flat section
    n_flat_points = 8  # More points to anchor flatness and prevent overshoot
    
    for party in parties_list:
        if party in pivot_data.columns:
            x_expanded = []
            y_top_expanded = []
            y_bottom_expanded = []
            
            for i, day in enumerate(x_days):
                bottom, top = stacked_values[day][party]
                
                # Add multiple points for flat section to force horizontal line
                flat_start = i - flat_width
                flat_end = i + flat_width
                flat_points = np.linspace(flat_start, flat_end, n_flat_points)
                
                for fp in flat_points:
                    x_expanded.append(fp)
                    y_top_expanded.append(top)
                    y_bottom_expanded.append(bottom)
            
            # Upper boundary line (with legend) - lower smoothing to prevent overshoot
            fig_ribbon1.add_trace(go.Scatter(
                x=x_expanded,
                y=y_top_expanded,
                mode='lines',
                line=dict(width=0.5, color=PARTY_COLORS[party], shape='spline', smoothing=0.8),
                showlegend=True,
                name=party,
                hoverinfo='skip'
            ))
            
            # Ribbon fill between bottom & top
            fig_ribbon1.add_trace(go.Scatter(
                x=x_expanded,
                y=y_bottom_expanded,
                mode='lines',
                line=dict(width=0.5, color=PARTY_COLORS[party], shape='spline', smoothing=0.8),
                fill='tonexty',
                fillcolor=PARTY_COLORS[party],
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig_ribbon1.update_layout(
        **layout_template,
        xaxis_title="Day of the Week",
        yaxis_title="Tweet Count",
        height=400,
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='left', 
            x=0, 
            font=dict(color=TEXT_COLOR),
            itemsizing='constant'
        ),
        hovermode='x unified',
        xaxis=dict(
            tickfont=dict(color=TEXT_COLOR), 
            title_font=dict(color=TEXT_COLOR), 
            gridcolor='#2d2d44',
            tickmode='array',
            tickvals=list(range(n_days)),
            ticktext=x_days
        ),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font=dict(color=TEXT_COLOR), gridcolor='#2d2d44')
    )
    
    st.plotly_chart(fig_ribbon1, use_container_width=True, config=plot_config)

st.markdown("---")

# ==================== VISUALIZATION 3: Engagement Score by Day and Party (Ribbon Chart) ====================
st.subheader("Engagement Score by Day and Party")

# Ribbon chart 2 specific filters
ribbon2_col1, ribbon2_col2, ribbon2_col3 = st.columns([2, 2, 6])
with ribbon2_col1:
    ribbon2_parties = st.multiselect("Parties", ['PMLN', 'PPP', 'PTI'], default=['PMLN', 'PPP', 'PTI'], key='ribbon2_parties')
with ribbon2_col2:
    ribbon2_days = st.multiselect("Days", day_order, default=day_order, key='ribbon2_days')

# Apply ribbon2-specific filters
ribbon2_df = filtered_df.copy()
if ribbon2_days:
    ribbon2_df = ribbon2_df[ribbon2_df['day_of_week'].isin(ribbon2_days)]

if 'day_of_week' in ribbon2_df.columns and 'party' in ribbon2_df.columns and 'engagement_score' in ribbon2_df.columns:
    daily_engagement = ribbon2_df.groupby(['day_of_week', 'party'])['engagement_score'].sum().reset_index()
    
    pivot_eng = daily_engagement.pivot(index='day_of_week', columns='party', values='engagement_score').fillna(0)
    pivot_eng = pivot_eng.reindex([d for d in ribbon2_days if d in pivot_eng.index])
    
    # Rank each party per day (1 = highest value)
    ranks = pivot_eng.rank(ascending=False, axis=1, method='first')
    
    # Sort parties based on ranking at each time period
    sorted_parties_per_day = ranks.apply(lambda row: row.sort_values().index.tolist(), axis=1)
    
    # Build stacked y-values based on rankings
    stacked_values = {}
    for day in pivot_eng.index:
        order = sorted_parties_per_day.loc[day]
        cumulative = 0
        stacked_values[day] = {}
        for party in order:
            stacked_values[day][party] = (cumulative, cumulative + pivot_eng.loc[day, party])
            cumulative += pivot_eng.loc[day, party]
    
    # Build ribbons with Plotly - TRUE Power BI style with flat sections
    fig_ribbon2 = go.Figure()
    
    parties_list = ribbon2_parties if ribbon2_parties else ['PMLN', 'PPP', 'PTI']
    x_days = pivot_eng.index.tolist()
    n_days = len(x_days)
    
    # Parameters for flat sections and transitions
    flat_width = 0.44  # Wide flat section
    n_flat_points = 8  # More points to anchor flatness and prevent overshoot
    
    for party in parties_list:
        if party in pivot_eng.columns:
            x_expanded = []
            y_top_expanded = []
            y_bottom_expanded = []
            
            for i, day in enumerate(x_days):
                bottom, top = stacked_values[day][party]
                
                # Add multiple points for flat section to force horizontal line
                flat_start = i - flat_width
                flat_end = i + flat_width
                flat_points = np.linspace(flat_start, flat_end, n_flat_points)
                
                for fp in flat_points:
                    x_expanded.append(fp)
                    y_top_expanded.append(top)
                    y_bottom_expanded.append(bottom)
            
            # Upper boundary line (with legend) - lower smoothing to prevent overshoot
            fig_ribbon2.add_trace(go.Scatter(
                x=x_expanded,
                y=y_top_expanded,
                mode='lines',
                line=dict(width=0.5, color=PARTY_COLORS[party], shape='spline', smoothing=0.8),
                showlegend=True,
                name=party,
                hoverinfo='skip'
            ))
            
            # Ribbon fill between bottom & top
            fig_ribbon2.add_trace(go.Scatter(
                x=x_expanded,
                y=y_bottom_expanded,
                mode='lines',
                line=dict(width=0.5, color=PARTY_COLORS[party], shape='spline', smoothing=0.8),
                fill='tonexty',
                fillcolor=PARTY_COLORS[party],
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig_ribbon2.update_layout(
        **layout_template,
        xaxis_title="Day of Week",
        yaxis_title="Engagement Score",
        height=400,
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='left', 
            x=0, 
            font=dict(color=TEXT_COLOR),
            itemsizing='constant'
        ),
        hovermode='x unified',
        xaxis=dict(
            tickfont=dict(color=TEXT_COLOR), 
            title_font=dict(color=TEXT_COLOR), 
            gridcolor='#2d4d2d',
            tickmode='array',
            tickvals=list(range(n_days)),
            ticktext=x_days
        ),
        yaxis=dict(
            tickfont=dict(color=TEXT_COLOR), 
            title_font=dict(color=TEXT_COLOR), 
            gridcolor='#2d4d2d',
            range=[-200, None]  # Start y-axis from -200
        )
    )
    
    st.plotly_chart(fig_ribbon2, use_container_width=True, config=plot_config)

st.markdown("---")

# ==================== VISUALIZATION 4 & 5: Tweet Count by Language (Donut) & by Party and Language (Bar) - SIDE BY SIDE ====================
st.subheader("Language Distribution Analysis")

# Create two columns for side-by-side visualizations
pie_col, bar_col = st.columns(2)

with pie_col:
    st.markdown("**Tweet Count by Language**")
    
    # Pie chart specific filter
    pie_party_filter = st.selectbox("Filter by Party", ['All'] + sorted(df['party'].dropna().unique().tolist()), key='pie_party')
    
    # Apply pie-specific filter
    pie_df = filtered_df_lang.copy()
    if pie_party_filter != 'All':
        pie_df = pie_df[pie_df['party'] == pie_party_filter]
    
    if 'language' in pie_df.columns:
        lang_counts = pie_df['language'].value_counts().reset_index()
        lang_counts.columns = ['language', 'count']
        
        # Get colors for each language
        colors = [LANGUAGE_COLORS.get(lang, '#808080') for lang in lang_counts['language']]
        
        # Create custom text - only show for english, urdu, japanese
        main_languages = ['english', 'urdu', 'japanese']
        total = lang_counts['count'].sum()
        custom_text = []
        for idx, row in lang_counts.iterrows():
            if row['language'] in main_languages:
                pct = (row['count'] / total) * 100
                custom_text.append(f"{row['count']:,}<br>{pct:.1f}%")
            else:
                custom_text.append('')
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=lang_counts['language'],
            values=lang_counts['count'],
            marker=dict(colors=colors),
            hovertemplate='Language: %{label}<br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>',
            text=custom_text,
            textinfo='text',
            textposition='outside',
            textfont=dict(color=TEXT_COLOR, size=10),
            hole=0.5
        )])
        
        fig_pie.update_layout(
            **layout_template,
            height=400,
            showlegend=True,
            legend=dict(
                orientation='h', 
                yanchor='bottom', 
                y=1.02, 
                xanchor='left', 
                x=0, 
                font=dict(size=9, color=TEXT_COLOR),
                itemsizing='constant'
            ),
            margin=dict(t=60, b=30, l=30, r=30)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True, config=plot_config)

with bar_col:
    st.markdown("**Tweet Count by Party and Language**")
    
    # Bar chart specific filters
    bar1_lang_filter = st.multiselect("Select Languages", ALLOWED_LANGUAGES, default=ALLOWED_LANGUAGES[:6], key='bar1_langs')
    
    # Apply bar1-specific filter
    bar1_df = filtered_df_lang.copy()
    if bar1_lang_filter:
        bar1_df = bar1_df[bar1_df['language'].isin(bar1_lang_filter)]
    
    if 'party' in bar1_df.columns and 'language' in bar1_df.columns:
        party_lang = bar1_df.groupby(['party', 'language']).size().reset_index(name='count')
        
        # Get top languages by count
        top_langs = bar1_lang_filter if bar1_lang_filter else party_lang.groupby('language')['count'].sum().nlargest(12).index.tolist()
        party_lang_filtered = party_lang[party_lang['language'].isin(top_langs)]
        
        fig_bar1 = go.Figure()
        
        for lang in top_langs:
            lang_data = party_lang_filtered[party_lang_filtered['language'] == lang]
            if len(lang_data) > 0:
                fig_bar1.add_trace(go.Bar(
                    x=lang_data['party'],
                    y=lang_data['count'],
                    name=lang,
                    marker_color=LANGUAGE_COLORS.get(lang, '#808080'),
                    hovertemplate='Party: %{x}<br>Language: ' + lang + '<br>Count: %{y:,}<extra></extra>'
                ))
        
        fig_bar1.update_layout(
            **layout_template,
            xaxis_title="Party",
            yaxis_title="Tweet Count",
            height=400,
            barmode='group',
            legend=dict(
                orientation='h', 
                yanchor='bottom', 
                y=1.02, 
                xanchor='left', 
                x=0, 
                font=dict(size=8, color=TEXT_COLOR),
                itemsizing='constant'
            ),
            xaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font=dict(color=TEXT_COLOR), gridcolor='#2d4d2d'),
            yaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font=dict(color=TEXT_COLOR), gridcolor='#2d4d2d'),
            margin=dict(b=30, t=60)
        )
        
        st.plotly_chart(fig_bar1, use_container_width=True, config=plot_config)

st.markdown("---")

# ==================== VISUALIZATION 6: Sum of Engagement Score by Language and Party (Grouped Bar) ====================
st.subheader("Sum of Engagement Score by Language and Party")

# Engagement bar chart specific filters
eng_col1, eng_col2, eng_col3 = st.columns([2, 2, 6])
with eng_col1:
    eng_parties = st.multiselect("Parties", ['PMLN', 'PPP', 'PTI'], default=['PMLN', 'PPP', 'PTI'], key='eng_parties')
with eng_col2:
    eng_langs = st.multiselect("Languages", ALLOWED_LANGUAGES, default=ALLOWED_LANGUAGES, key='eng_langs')

# Apply engagement chart specific filters
eng_df = filtered_df_lang.copy()
if eng_langs:
    eng_df = eng_df[eng_df['language'].isin(eng_langs)]

if 'party' in eng_df.columns and 'language' in eng_df.columns and 'engagement_score' in eng_df.columns:
    lang_party_engagement = eng_df.groupby(['language', 'party'])['engagement_score'].sum().reset_index()
    
    # Filter to selected languages
    lang_party_engagement = lang_party_engagement[lang_party_engagement['language'].isin(eng_langs if eng_langs else ALLOWED_LANGUAGES)]
    
    # Order by total engagement - highest to lowest (left to right)
    lang_totals = lang_party_engagement.groupby('language')['engagement_score'].sum().sort_values(ascending=False)
    lang_order = lang_totals.index.tolist()
    
    fig_bar2 = go.Figure()
    
    parties_to_show = eng_parties if eng_parties else ['PMLN', 'PPP', 'PTI']
    for party in parties_to_show:
        party_data = lang_party_engagement[lang_party_engagement['party'] == party].copy()
        # Create a complete dataframe with all languages in order
        party_dict = dict(zip(party_data['language'], party_data['engagement_score']))
        ordered_values = [party_dict.get(lang, 0) for lang in lang_order]
        
        fig_bar2.add_trace(go.Bar(
            x=lang_order,
            y=ordered_values,
            name=party,
            marker_color=PARTY_COLORS.get(party),
            hovertemplate='Language: %{x}<br>Party: ' + party + '<br>Engagement: %{y:,.0f}<extra></extra>'
        ))
    
    fig_bar2.update_layout(
        **layout_template,
        xaxis_title="Language",
        yaxis_title="Sum of Engagement Score",
        height=400,
        barmode='group',
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='left', 
            x=0, 
            font=dict(color=TEXT_COLOR),
            itemsizing='constant'
        ),
        xaxis=dict(
            tickfont=dict(color=TEXT_COLOR, size=10), 
            title_font=dict(color=TEXT_COLOR), 
            gridcolor='#2d4d2d', 
            tickangle=45,
            categoryorder='array',
            categoryarray=lang_order
        ),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font=dict(color=TEXT_COLOR), gridcolor='#2d4d2d'),
        margin=dict(b=100, t=80)
    )
    
    st.plotly_chart(fig_bar2, use_container_width=True, config=plot_config)

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #9CB770; font-family: Rajdhani, sans-serif;'>Pakistan's Political Discourse Analytics | Data Visualization Project</p>", unsafe_allow_html=True)
