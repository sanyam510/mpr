# Complete Analysis Dashboard
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Flipkart Headphones Deep Dive", layout="wide")
# Load and Preprocess Data
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join("scraping", "data_eda.csv"))
    
    # Clean Price Range
    df['Price Range'] = df['Price Range'].str.replace('â_x0080_x0093', '-', regex=True)
    
    # Calculate Value for Money properly
    df['Value for Money'] = df['Rating'] / df['Price']
    
    # Create Discount Efficiency metric
    df['Discount Efficiency'] = df['Discount Amount'] / df['Price']
    
    # Price Segmentation
    bins = [0, 500, 1000, 2000, float('inf')]
    labels = ['Budget', 'Mid-Range', 'Premium', 'Luxury']
    df['Price Segment'] = pd.cut(df['Price'], bins=bins, labels=labels)
    
    return df

df = load_data()

# Dashboard Setup

st.title("🎧 WavePulse Tracking Consumer Behavior & Product Trends in the Audio Market")

# Sidebar Filters
st.sidebar.header("🔍 Advanced Filters")
selected_brands = st.sidebar.multiselect("Brands", options=df['Brand'].unique())
price_range = st.sidebar.slider("Price Range (₹)", 
                              min_value=0, 
                              max_value=int(df['Price'].max()), 
                              value=(0, int(df['Price'].max())))
rating_filter = st.sidebar.slider("Rating", 0.0, 5.0, (0.0, 5.0))
discount_filter = st.sidebar.slider("Discount %", 0, 100, (0, 100))
product_type = st.sidebar.selectbox("Product Type", ['All'] + list(df['Product Type'].unique()))
price_segment = st.sidebar.multiselect("Price Segment", options=df['Price Segment'].unique())

# Apply Filters
filtered_df = df[
    (df['Brand'].isin(selected_brands) if selected_brands else True) &
    (df['Price'].between(*price_range)) &
    (df['Rating'].between(*rating_filter)) &
    (df['Discount %'].between(*discount_filter)) &
    ((df['Product Type'] == product_type) if product_type != 'All' else True) &
    (df['Price Segment'].isin(price_segment) if price_segment else True)
]

# --------------------------
# Section 1: Core Metrics
# --------------------------
st.header("📊 Core Market Metrics")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric("Avg Price", f"₹{filtered_df['Price'].mean():.1f}", 
             delta=f"₹{filtered_df['Price'].mean() - df['Price'].mean():.1f} vs Overall")
    
with kpi2:
    total_count = len(filtered_df)
    premium_luxury_count = len(filtered_df[filtered_df['Price Segment'].isin(['Premium', 'Luxury'])])

    if total_count > 0:
        premium_index = (premium_luxury_count / total_count) * 100
        premium_index_str = f"{premium_index:.1f}%"
    else:
        premium_index_str = "N/A"

    st.metric(
        "Premium Index",
        premium_index_str,
        "Premium+Luxury Share"
    )


with kpi3:
    st.metric("Discount Power", 
             f"₹{filtered_df['Discount Amount'].sum():,.0f}",
             "Total Discounts Given")

with kpi4:
    st.metric("Rating Power", 
             f"{filtered_df['Rating'].mean():.1f}/5 ⭐",
             f"Based on {filtered_df['Number of Ratings'].sum():,} reviews")

with kpi5:
    st.metric("Market Diversity", 
             f"{filtered_df['Brand'].nunique()} Brands",
             f"{filtered_df['Product Type'].nunique()} Categories")

# --------------------------
# Section 2: Brand Analysis
# --------------------------
st.header("🏷 Brand Intelligence")
col1, col2, col3 = st.columns([2,1,1])

with col1:
    # Brand Market Share
    brand_market = filtered_df.groupby('Brand').agg({
        'Price': 'sum',
        'Number of Ratings': 'sum',
        'Discount %': 'mean'
    }).reset_index()
    
    fig = px.treemap(brand_market,
                    path=['Brand'], 
                    values='Price',
                    color='Discount %',
                    hover_data=['Number of Ratings'],
                    color_continuous_scale='RdYlGn',
                    title="Brand Market Share (Size=Revenue, Color=Discounts)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Brand Rating Distribution
    fig = px.box(filtered_df, 
                x='Brand', 
                y='Rating',
                color='Brand',
                title="Brand Rating Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    # Brand-Category Matrix
    cross_tab = pd.crosstab(filtered_df['Brand'], filtered_df['Product Type'])
    fig = px.imshow(cross_tab,
                   labels=dict(x="Category", y="Brand", color="Count"),
                   title="Brand vs Category Matrix")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Section 3: Price Analysis
# --------------------------
st.header("💰 Price Optimization Analysis")
col1, col2 = st.columns(2)

with col1:
    # Price-Rating-Discount 3D Analysis
    fig = px.scatter_3d(filtered_df,
                       x='Price',
                       y='Rating',
                       z='Discount %',
                       color='Brand',
                       size='Number of Ratings',
                       hover_name='Product Name',
                       title="3D Pricing Strategy Analysis")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Price Segment Analysis
    segment_analysis = filtered_df.groupby('Price Segment').agg({
        'Price': 'mean',
        'Rating': 'mean',
        'Discount %': 'mean'
    }).reset_index()
    
    fig = px.bar(segment_analysis,
                x='Price Segment',
                y=['Price', 'Rating', 'Discount %'],
                barmode='group',
                title="Price Segment Comparison")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Section 4: Discount Deep Dive
# --------------------------
st.header("🎯 Discount Effectiveness")
col1, col2 = st.columns(2)

with col1:
    # Discount vs Sales Performance
    fig = px.scatter(filtered_df,
                   x='Discount %',
                   y='Number of Ratings',
                   hover_name='Brand', 
                   size='Price',
                   color='Rating',
                   trendline="lowess",
                   title="Discount Impact on Popularity & Ratings")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Discount Strategy Matrix (Improved)
    fig = px.scatter(filtered_df,
                    x='Price',
                    y='Discount %',
                    color='Rating',
                    size='Number of Ratings',
                    hover_name='Product Name',
                    hover_data=['Brand', 'Product Type'],
                    title="Discount Strategy Analysis",
                    trendline="lowess",
                    trendline_color_override="red")
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Price (₹)",
        yaxis_title="Discount (%)",
        hovermode='closest',
        coloraxis_colorbar=dict(title="Rating"),
        height=600
    )
    
    # Add reference lines
    fig.add_shape(type="line", x0=0, y0=20, x1=6000, y1=20, 
                  line=dict(color="gray", dash="dot"))
    fig.add_annotation(x=5000, y=22, text="20% Discount Threshold", 
                       showarrow=False, font=dict(color="gray"))
    
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Section 5: Product Intelligence
# --------------------------
# Product Comparison Tool
st.subheader("Product Comparison Matrix")

# Use unique indices to avoid duplicate issues
selected_indices = st.multiselect(
    "Select products (max 5 for clarity)",
    options=filtered_df.index.tolist(),
    format_func=lambda x: f"{filtered_df.loc[x, 'Product Name'][:30]}... ({filtered_df.loc[x, 'Brand']})"
)

if selected_indices:
    compare_df = filtered_df.loc[selected_indices]
    
    # Normalize features
    features = compare_df[['Price', 'Rating', 'Discount %', 'Number of Ratings']]
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(features)
    
    # Create custom hover text
    hover_texts = []
    for _, row in compare_df.iterrows():
        text = (
            f"<b>{row['Product Name']}</b><br>"
            f"Brand: {row['Brand']}<br>"
            f"Price: ₹{row['Price']}<br>"
            f"Rating: {row['Rating']}/5<br>"
            f"Discount: {row['Discount %']}%<br>"
            f"Popularity: {row['Number of Ratings']} ratings"
        )
        hover_texts.append(text)
    
    fig = go.Figure()
    
    # Color palette for distinct product colors
    colors = px.colors.qualitative.Plotly
    
    for i, (row, text) in enumerate(zip(normalized, hover_texts)):
        fig.add_trace(go.Scatterpolar(
            r=row,
            theta=['Price', 'Rating', 'Discount %', 'Popularity'],
            fill='toself',
            opacity=0.8,  # Increased transparency for overlaps
            name=f"{compare_df.iloc[i]['Product Name'][:20]}...",
            hoverinfo="text",
            hovertext=text,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=5, color=colors[i % len(colors)])
        ))
    
    # Add metric scale annotations
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickvals=[0, 0.5, 1],
                ticktext=['Low', 'Medium', 'High'],
                tickfont=dict(size=10)
            )
        ),
        title=dict(
            text="Product Comparison Radar Chart<br><sup>Normalized Scale: 0=Low, 1=High</sup>",
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            title="Products",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=100, r=150),  # Space for legend
        height=650
    )
    
    # Add explanatory annotation
    fig.add_annotation(
        x=1.15,
        y=0.5,
        xref="paper",
        yref="paper",
        text="<b>Metric Scales:</b><br>"
             "Price: ₹0-6,000<br>"
             "Rating: 0-5 stars<br>"
             "Discount: 0-100%<br>"
             "Popularity: 0-Max Ratings",
        showarrow=False,
        align="left",
        bordercolor="#c7c7c7",
        borderwidth=1
    )
    
    st.plotly_chart(fig, use_container_width=True)
# Section 6: Advanced Analytics
# --------------------------
st.header("🔮 Predictive Insights")

col1, col2 = st.columns(2)

with col1:
    # Price vs Rating Predictive Trend
    fig = px.scatter(filtered_df,
                   x='Price',
                   y='Rating',
                   hover_name = 'Brand',
                   trendline="lowess",
                   title="Price-Rating Relationship with Trend")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Value for Money Analysis
    fig = px.box(filtered_df,
                x='Price Segment',
                y='Value for Money',
                hover_name = 'Brand',
                color='Product Type',
                title="Value for Money Analysis")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Section 7: Raw Data Explorer
# --------------------------
st.header("📁 Data Explorer")
st.data_editor(
    filtered_df.sort_values('Number of Ratings', ascending=False),
    column_config={
        "Product Link": st.column_config.LinkColumn(),
        "Discount %": st.column_config.ProgressColumn(
            format="%d%%",
            min_value=0,
            max_value=100,
        )
    },
    hide_index=True,
    use_container_width=True
)


# --------------------------
# Section 8: Machine Learning Models
# --------------------------
st.header("🤖 Practical ML Implementations")

# 1. Price Prediction Model (Regression)
st.subheader("💰 Price Prediction Engine")
with st.expander("Train Price Predictor"):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split

    # Feature Engineering
    ml_df = filtered_df[['Brand', 'Product Type', 'Rating', 'Discount %', 
                         'Number of Ratings', 'Price']].copy()

    if ml_df.empty:
        st.warning("Not enough data to train the model. Please adjust filters.")
    else:
        # Convert categorical features
        ml_df = pd.get_dummies(ml_df, columns=['Brand', 'Product Type'])

        X = ml_df.drop('Price', axis=1)
        y = ml_df['Price']

        if len(X) >= 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        if len(X_train) > 0 and len(y_train) > 0:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Prediction and Evaluation
            y_pred = model.predict(X_test)
            accuracy = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Visualization
            fig = px.scatter(x=y_test, y=y_pred, 
                             labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                             title=f"Price Prediction Performance (R² = {accuracy:.2f}, MAE = ₹{mae:.1f})")
            st.plotly_chart(fig)

            # Feature Importance
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig2 = px.bar(importance.head(10), x='Feature', y='Importance', 
                          title='Top 10 Important Features for Price Prediction')
            st.plotly_chart(fig2)
        else:
            st.warning("Not enough samples to train the model.")


# 2. Rating Classifier (Classification)
st.subheader("⭐ Rating Category Predictor")

with st.expander("Predict Rating Category"):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report

    # Create rating categories
    rating_df = filtered_df.copy()
    rating_df['Rating Category'] = pd.cut(
        rating_df['Rating'],
        bins=[0, 3.5, 4.2, 5],
        labels=['Low', 'Medium', 'High']
    )
    
    # Feature selection
    X = rating_df[['Price', 'Discount %', 'Brand', 'Product Type', 'Number of Ratings']]
    X = pd.get_dummies(X, columns=['Brand', 'Product Type'])
    y = rating_df['Rating Category']

    # Remove rows where y is NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # 🚨 Check data validity
    if X.shape[0] < 5:
        st.warning(f"⚠️ Not enough data to train the model! Only {X.shape[0]} samples available.")
    elif y.nunique() < 2:
        st.warning(f"⚠️ Not enough classes to train the model! Only one category found: {y.unique()[0]}")
    else:
        # Train model
        clf = GradientBoostingClassifier()
        clf.fit(X, y)
        st.success("✅ Model trained successfully!")

        # (optional) Show some basic output
        # y_pred = clf.predict(X)
        # st.text("Classification Report:")
        # st.text(classification_report(y, y_pred))

    
    # Live prediction interface
    st.markdown("*Predict for New Product:*")
    col1, col2, col3 = st.columns(3)
    with col1:
        price = st.number_input("Price (₹)", min_value=0)
    with col2:
        discount = st.slider("Discount (%)", 0, 100)
    with col3:
        brand = st.selectbox("Brand", filtered_df['Brand'].unique())
    
    product_type = st.selectbox("Product Type", filtered_df['Product Type'].unique())
    num_ratings = st.number_input("Expected Number of Ratings", min_value=0)
    
    if st.button("Predict Rating Category"):
        input_data = pd.DataFrame([[price, discount, brand, product_type, num_ratings]],
                                 columns=['Price', 'Discount %', 'Brand', 'Product Type', 'Number of Ratings'])
        input_encoded = pd.get_dummies(input_data)
        
        # Align columns
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
        
        prediction = clf.predict(input_encoded)[0]
        probability = clf.predict_proba(input_encoded)[0].max()
        
        st.success(f"Predicted Rating Category: *{prediction}* (Confidence: {probability:.0%})")

#3. Product Clustering (Unsupervised Learning)
st.subheader("🔍 Product Clustering Analysis")
with st.expander("Explore Product Segments"):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Prepare data with product names
    cluster_df = filtered_df[['Product Name', 'Price', 'Rating', 'Discount %', 'Number of Ratings']]

    if cluster_df.shape[0] < 2:
        st.warning(f"⚠️ Not enough products ({cluster_df.shape[0]}) to perform clustering! Need at least 2.")
    else:
        # Normalization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df[['Price', 'Rating', 'Discount %', 'Number of Ratings']])

        # Elbow Method Explanation
        st.markdown("""
        **How to choose clusters?**
        - Look for the 'elbow' point where the line bends
        - After this point, adding more clusters doesn't help much
        """)
        
        # Determine optimal clusters
        wcss = []
        max_clusters = min(10, cluster_df.shape[0])
        for i in range(1, max_clusters):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)

        # Elbow plot
        fig1 = px.line(
            x=range(1, len(wcss)+1), 
            y=wcss,
            title='Finding the Optimal Number of Clusters',
            labels={'x': 'Number of Clusters', 'y': 'Compactness Score'},
            markers=True
        )
        st.plotly_chart(fig1)

        # Cluster selection
        n_clusters = st.slider("Select Number of Clusters", 
                             2, max(2, max_clusters-1), 
                             value=min(3, max_clusters-1))
        
        if cluster_df.shape[0] >= n_clusters:
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            cluster_df['Cluster'] = clusters.astype(str)  # Convert to string for categorical color

            # Interactive 2D Scatter Plot
            st.markdown("### 📊 Product Cluster Visualization")
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-Axis Feature", 
                                    ['Price', 'Rating', 'Discount %', 'Number of Ratings'], 
                                    index=0)
            with col2:
                y_axis = st.selectbox("Y-Axis Feature", 
                                    ['Rating', 'Price', 'Discount %', 'Number of Ratings'], 
                                    index=0)

            fig2 = px.scatter(
                cluster_df,
                x=x_axis,
                y=y_axis,
                color='Cluster',
                hover_name='Product Name',
                title=f"Product Clusters ({x_axis} vs {y_axis})",
                labels={'Cluster': 'Product Group'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig2)

            # Cluster Summary Table
            st.markdown("### 📝 Cluster Characteristics")
            cluster_summary = cluster_df.groupby('Cluster').agg({
                'Price': 'mean',
                'Rating': 'mean',
                'Discount %': 'mean',
                'Number of Ratings': 'mean',
                'Product Name': 'count'
            }).rename(columns={'Product Name': 'Count'}).reset_index()
            
            # Format numbers
            cluster_summary['Price'] = cluster_summary['Price'].round(2)
            cluster_summary['Rating'] = cluster_summary['Rating'].round(1)
            cluster_summary['Discount %'] = cluster_summary['Discount %'].round(1)
            cluster_summary['Number of Ratings'] = cluster_summary['Number of Ratings'].astype(int)
            
            # Display summary
            st.dataframe(
                cluster_summary.style.highlight_max(subset=['Count'], color='#90EE90'),
                use_container_width=True
            )

            # Interpretation Guide
            st.markdown("""
            **How to interpret:**
            - Each color represents a product group
            - Points close together are similar products
            - Check the summary table for group characteristics
            - Hover over points to see product details
            """)
        else:
            st.warning(f"⚠️ Cannot create {n_clusters} clusters from {cluster_df.shape[0]} products!")

# 4. Discount Effectiveness Predictor
st.subheader("🎯 Discount Impact Analyzer")
with st.expander("Predict Discount Impact"):
    from sklearn.linear_model import LogisticRegression
    
    # Create target variable (High Popularity)
    discount_df = filtered_df.copy()
    
    if discount_df.shape[0] < 2:
        st.warning("⚠️ Not enough data to analyze discount impact. Please select more products.")
    else:
        discount_df['High Popularity'] = np.where(discount_df['Number of Ratings'] > 
                                                discount_df['Number of Ratings'].median(), 1, 0)

        # Prepare data
        X = discount_df[['Price', 'Discount %', 'Brand', 'Product Type']]
        X = pd.get_dummies(X)
        y = discount_df['High Popularity']

        # 🛡️ Check if y has at least 2 unique classes
        if len(y.unique()) < 2:
            st.warning("⚠️ Cannot train model: Only one class present in the data.")
        else:
            # Train model
            model = LogisticRegression()
            model.fit(X, y)

            # Prediction interface
            st.markdown("*Will this discount strategy work?*")
            col1, col2 = st.columns(2)
            with col1:
                disc_price = st.number_input("Product Price (₹)", min_value=0)
            with col2:
                disc_pct = st.slider("Planned Discount (%)", 0, 100)

            disc_brand = st.selectbox("Product Brand", filtered_df['Brand'].unique())
            disc_type = st.selectbox("Product Type", filtered_df['Product Type'].unique(), key="product_type_selectbox")

            if st.button("Predict Popularity"):
                input_data = pd.DataFrame([[disc_price, disc_pct, disc_brand, disc_type]],
                                          columns=['Price', 'Discount %', 'Brand', 'Product Type'])
                input_encoded = pd.get_dummies(input_data)
                input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

                prediction = model.predict(input_encoded)[0]
                proba = model.predict_proba(input_encoded)[0][1]

                if prediction == 1:
                    st.success(f"High Popularity Expected ({proba:.0%} confidence)")
                else:
                    st.error(f"Low Popularity Risk ({1 - proba:.0%} confidence)")
