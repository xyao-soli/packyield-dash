import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import base64
from io import StringIO
from datetime import datetime, date

# Set page config
st.set_page_config(
    page_title="Interactive Sales Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and clean the data from Streamlit secrets"""
    try:
        # Load data from Streamlit secrets (encoded)
        encoded_data = st.secrets["data_csv"]
        
        # Decode the data
        decoded_data = base64.b64decode(encoded_data).decode()
        
        # Convert to DataFrame
        df = pd.read_csv(StringIO(decoded_data))
        
        # Clean column names (remove trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Rename columns for easier handling
        df = df.rename(columns={
            'Unit Size (ounces)': 'Unit_Size_ounces',
            'Customer': 'Customer',
            'Sum of Pounds Consumed': 'Pounds_Consumed',
            'Sum of Pounds Sold': 'Pounds_Sold',
            'Pack house': 'Pack_house',
            'Days_on_hand': 'Days_on_hand',
            'Days from Harvest to Pack': 'Days_from_Harvest_to_Pack',
            'Date Packed': 'Date_Packed'
        })
        
        # Convert Date Packed to datetime
        df['Date_Packed'] = pd.to_datetime(df['Date_Packed'], errors='coerce')
        
        # Handle missing customers (replace NaN with 'No Customer')
        df['Customer'] = df['Customer'].fillna('No Customer')
        
        # Standardize customer names - combine grocers under different names
        customer_mapping = {
            'SW': 'Safeway',
            'Safeway Tom Thum': 'Safeway',
            'FM': 'Fred Meyer',
            'TJ': 'Trader Joe\'s',
            'TT': 'Target'
        }
        df['Customer'] = df['Customer'].replace(customer_mapping)
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Pounds_Sold'])
        
        # Filter Days_on_hand to be between 0 and 5 (inclusive)
        df = df[(df['Days_on_hand'] >= 0) & (df['Days_on_hand'] <= 5)]
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please check if the data_csv secret is properly configured.")
        st.stop()

def create_clustered_bar_chart(filtered_df, tier1, tier2, tier3):
    """Create the main clustered bar chart with variable width and grid lines"""
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for selected filters")
    def create_time_series_chart(filtered_df, selected_unit_size, selected_customer, selected_pack_house):
    """Create time series chart showing sell-through rate over year-month"""
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for selected filters")
        return fig
    
    # Filter data based on selections
    time_series_df = filtered_df.copy()
    
    if selected_unit_size:
        time_series_df = time_series_df[time_series_df['Unit_Size_ounces'] == selected_unit_size]
    if selected_customer and selected_customer != 'All':
        time_series_df = time_series_df[time_series_df['Customer'] == selected_customer]
    if selected_pack_house and selected_pack_house != 'All':
        time_series_df = time_series_df[time_series_df['Pack_house'] == selected_pack_house]
    
    if time_series_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for selected combination")
        return fig
    
    # Create year-month column
    time_series_df['Year_Month'] = time_series_df['Date_Packed'].dt.to_period('M')
    
    # Group by year-month and calculate sell-through rate
    monthly_data = time_series_df.groupby('Year_Month').agg({
        'Pounds_Sold': 'sum',
        'Pounds_Consumed': 'sum'
    }).reset_index()
    
    # Calculate sell-through rate
    monthly_data['Sell_Through_Rate'] = np.where(
        monthly_data['Pounds_Consumed'] > 0,
        (monthly_data['Pounds_Sold'] / monthly_data['Pounds_Consumed']) * 100,
        0
    )
    
    # Convert Year_Month back to datetime for plotting
    monthly_data['Date'] = monthly_data['Year_Month'].dt.to_timestamp()
    
    # Create the time series plot
    fig = go.Figure()
    
    # Add line trace
    fig.add_trace(go.Scatter(
        x=monthly_data['Date'],
        y=monthly_data['Sell_Through_Rate'],
        mode='lines+markers',
        name='Sell-Through Rate',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#1f77b4'),
        hovertemplate=(
            '<b>%{x|%Y-%m}</b><br>' +
            'Sell-Through Rate: %{y:.1f}%<br>' +
            '<extra></extra>'
        )
    ))
    
    # Add a horizontal line at 100% for reference
    fig.add_hline(
        y=100, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="100% Sell-Through",
        annotation_position="top right"
    )
    
    # Create title based on selections
    title_parts = []
    if selected_unit_size:
        title_parts.append(f"{selected_unit_size} oz")
    if selected_customer and selected_customer != 'All':
        title_parts.append(selected_customer)
    if selected_pack_house and selected_pack_house != 'All':
        title_parts.append(selected_pack_house)
    
    title = "Sell-Through Rate Over Time"
    if title_parts:
        title += f": {' | '.join(title_parts)}"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Year-Month',
        yaxis_title='Sell-Through Rate (%)',
        template='plotly_white',
        height=400,
        hovermode='x unified',
        xaxis=dict(
            tickformat='%Y-%m',
            dtick='M1'  # Show monthly ticks
        ),
        yaxis=dict(
            ticksuffix='%'
        )
    )
    
    return fig
    
    # Group data and calculate sell-through rates
    grouped = filtered_df.groupby([tier1, tier2, tier3]).agg({
        'Pounds_Sold': 'sum',
        'Pounds_Consumed': 'sum'
    }).reset_index()
    
    # Calculate (Sum of Pounds Sold) / (Sum of Pounds Consumed) * 100%
    grouped['Sell_Through_Rate'] = np.where(
        grouped['Pounds_Consumed'] > 0,
        (grouped['Pounds_Sold'] / grouped['Pounds_Consumed']) * 100,
        0
    )
    
    # Create the figure
    fig = go.Figure()
    
    # Create a set of valid combinations to avoid empty bars
    valid_combinations = set()
    for _, row in grouped.iterrows():
        valid_combinations.add((row[tier1], row[tier2], row[tier3]))
    
    # Get unique values for each tier
    tier1_values = sorted(grouped[tier1].unique())
    tier2_values = sorted(grouped[tier2].unique())
    tier3_values = sorted(grouped[tier3].unique())
    
    # Color palette
    colors = px.colors.qualitative.Set3
    color_map = {val: colors[i % len(colors)] for i, val in enumerate(tier3_values)}
    
    # Get maximum pounds sold for width scaling
    max_pounds_sold = grouped['Pounds_Sold'].max()
    
    # Get all unique x-axis labels in order for grid lines
    all_x_labels_ordered = []
    for tier1_val in tier1_values:
        tier2_for_tier1 = sorted([
            tier2_val for tier2_val in tier2_values 
            if any((tier1_val, tier2_val, tier3_val) in valid_combinations for tier3_val in tier3_values)
        ])
        for tier2_val in tier2_for_tier1:
            x_label = f"{tier1_val} | {tier2_val}"
            if x_label not in all_x_labels_ordered:
                all_x_labels_ordered.append(x_label)
    
    # Create bars for each tier3 value
    for tier3_val in tier3_values:
        tier3_data = grouped[grouped[tier3] == tier3_val]
        
        x_labels = []
        y_values = []
        hover_text = []
        bar_widths = []
        
        for tier1_val in tier1_values:
            tier2_for_tier1 = sorted([
                tier2_val for tier2_val in tier2_values 
                if (tier1_val, tier2_val, tier3_val) in valid_combinations
            ])
            
            for tier2_val in tier2_for_tier1:
                if (tier1_val, tier2_val, tier3_val) in valid_combinations:
                    subset = tier3_data[
                        (tier3_data[tier1] == tier1_val) & 
                        (tier3_data[tier2] == tier2_val)
                    ]
                    
                    if not subset.empty:
                        sell_through_rate = subset['Sell_Through_Rate'].iloc[0]
                        pounds_sold = subset['Pounds_Sold'].iloc[0]
                        pounds_consumed = subset['Pounds_Consumed'].iloc[0]
                        
                        x_label = f"{tier1_val} | {tier2_val}"
                        x_labels.append(x_label)
                        y_values.append(sell_through_rate)
                        
                        # Calculate variable bar width based on pounds sold
                        if max_pounds_sold > 0:
                            width = 0.1 + (pounds_sold / max_pounds_sold) * 0.7
                        else:
                            width = 0.4
                        bar_widths.append(width)
                        
                        hover_text.append(
                            f"{tier1.replace('_', ' ')}: {tier1_val}<br>"
                            f"{tier2.replace('_', ' ')}: {tier2_val}<br>"
                            f"{tier3.replace('_', ' ')}: {tier3_val}<br>"
                            f"Sell-Through Rate: {sell_through_rate:.1f}%<br>"
                            f"Pounds Sold: {pounds_sold:,.0f}<br>"
                            f"Pounds Consumed: {pounds_consumed:,.0f}"
                        )
        
        # Add bars with variable width
        if x_labels:
            fig.add_trace(go.Bar(
                x=x_labels,
                y=y_values,
                name=str(tier3_val),
                marker_color=color_map[tier3_val],
                hovertext=hover_text,
                hoverinfo='text',
                width=bar_widths
            ))
    
    # Add vertical grid lines between ALL customer groups
    if len(all_x_labels_ordered) > 1:
        for i in range(1, len(all_x_labels_ordered)):
            fig.add_shape(
                type="line",
                x0=i - 0.5,
                x1=i - 0.5,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="lightgrey", width=1),
                layer="below"
            )
    
    # Update layout
    fig.update_layout(
        title=f'Sales Distribution: {tier1.replace("_", " ")} → {tier2.replace("_", " ")} → {tier3.replace("_", " ")}',
        xaxis_title=f'{tier1.replace("_", " ")} | {tier2.replace("_", " ")}',
        yaxis_title='Sell-Through Rate: (Pounds Sold / Pounds Consumed) × 100%',
        barmode='group',
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(r=150),
        height=600,
        xaxis=dict(
            type="category",
            categoryorder='category ascending'
        )
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    # Title
    st.title("📊 Interactive Sales Data Analysis")
    st.markdown("---")
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ Could not find 'data_unit_customer_consumed_sold.csv'. Please upload the file.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Apply the same cleaning logic here if needed
        else:
            st.stop()
    
    # Sidebar controls
    st.sidebar.header("📋 Filter Controls")
    
    # Tier selection
    st.sidebar.subheader("🎯 Grouping Tiers")
    available_columns = ['Unit_Size_ounces', 'Customer', 'Pack_house']
    column_labels = ['Unit Size (ounces)', 'Customer', 'Pack house']
    column_mapping = dict(zip(column_labels, available_columns))
    
    tier1_label = st.sidebar.selectbox("1st Tier (X-axis grouping):", column_labels, index=0)
    tier2_label = st.sidebar.selectbox("2nd Tier (Sub-grouping):", column_labels, index=1)
    tier3_label = st.sidebar.selectbox("3rd Tier (Bar clustering):", column_labels, index=2)
    
    tier1 = column_mapping[tier1_label]
    tier2 = column_mapping[tier2_label]
    tier3 = column_mapping[tier3_label]
    
    # Data filters
    st.sidebar.subheader("🔍 Data Filters")
    
    # Unit Size filter
    unit_sizes = sorted(df['Unit_Size_ounces'].unique())
    selected_unit_sizes = st.sidebar.multiselect(
        "Unit Sizes:", 
        unit_sizes, 
        default=unit_sizes,
        help="Select unit sizes to include"
    )
    
    # Add Select All / Clear All buttons for Unit Size
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Select All Units"):
        selected_unit_sizes = unit_sizes
        st.experimental_rerun()
    if col2.button("Clear All Units"):
        selected_unit_sizes = []
        st.experimental_rerun()
    
    # Pack House filter
    pack_houses = sorted(df['Pack_house'].unique())
    selected_pack_houses = st.sidebar.multiselect(
        "Pack Houses:", 
        pack_houses, 
        default=pack_houses,
        help="Select pack houses to include"
    )
    
    # Add Select All / Clear All buttons for Pack House
    col3, col4 = st.sidebar.columns(2)
    if col3.button("Select All Houses"):
        selected_pack_houses = pack_houses
        st.experimental_rerun()
    if col4.button("Clear All Houses"):
        selected_pack_houses = []
        st.experimental_rerun()
    
    # Date range filter
    min_date = df['Date_Packed'].min().date()
    max_date = df['Date_Packed'].max().date()
    default_start = date(2024, 8, 1)
    default_end = date(2025, 7, 31)
    
    # Ensure defaults are within data range
    if default_start < min_date:
        default_start = min_date
    if default_end > max_date:
        default_end = max_date
    
    date_range = st.sidebar.date_input(
        "Date Range:",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date,
        help="Select date range for analysis"
    )
    
    # Handle date range input
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0] if len(date_range) == 1 else min_date
    
    # Days on Hand filter (default to 3)
    days_on_hand_range = st.sidebar.slider(
        "Days on Hand:",
        min_value=int(df['Days_on_hand'].min()),
        max_value=int(df['Days_on_hand'].max()),
        value=3,  # Default to 3
        help="Filter by days on hand"
    )
    
    # Days from Harvest to Pack filter (default to 5)
    days_harvest_range = st.sidebar.slider(
        "Days from Harvest to Pack:",
        min_value=int(df['Days_from_Harvest_to_Pack'].min()),
        max_value=int(df['Days_from_Harvest_to_Pack'].max()),
        value=5,  # Default to 5
        help="Filter by days from harvest to pack"
    )
    
    # Apply filters
    filtered_df = df[
        (df['Days_on_hand'] == days_on_hand_range) &
        (df['Days_from_Harvest_to_Pack'] == days_harvest_range) &
        (df['Date_Packed'] >= pd.to_datetime(start_date)) &
        (df['Date_Packed'] <= pd.to_datetime(end_date)) &
        (df['Unit_Size_ounces'].isin(selected_unit_sizes if selected_unit_sizes else [])) &
        (df['Pack_house'].isin(selected_pack_houses if selected_pack_houses else []))
    ].copy()
    
    # Main content area
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters. Please adjust your selections.")
        return
    
    # Create and display the main clustered bar chart
    fig = create_clustered_bar_chart(filtered_df, tier1, tier2, tier3)
    st.plotly_chart(fig, use_container_width=True)
    
    # Time Series Analysis Section
    st.markdown("---")
    st.subheader("📈 Time Series Analysis: Sell-Through Rate Over Time")
    
    # Time series controls in columns
    ts_col1, ts_col2, ts_col3, ts_col4 = st.columns(4)
    
    with ts_col1:
        # Unit Size selection for time series
        unit_size_options = ['All'] + sorted(df['Unit_Size_ounces'].unique().tolist())
        selected_unit_size_ts = st.selectbox(
            "Unit Size for Time Series:",
            unit_size_options,
            index=1 if len(unit_size_options) > 1 else 0,
            help="Select specific unit size for time series analysis"
        )
        if selected_unit_size_ts == 'All':
            selected_unit_size_ts = None
    
    with ts_col2:
        # Customer selection for time series
        customer_options = ['All'] + sorted(df['Customer'].unique().tolist())
        selected_customer_ts = st.selectbox(
            "Customer for Time Series:",
            customer_options,
            index=0,
            help="Select specific customer for time series analysis"
        )
    
    with ts_col3:
        # Pack House selection for time series
        pack_house_options = ['All'] + sorted(df['Pack_house'].unique().tolist())
        selected_pack_house_ts = st.selectbox(
            "Pack House for Time Series:",
            pack_house_options,
            index=0,
            help="Select specific pack house for time series analysis"
        )
    
    with ts_col4:
        # Date range for time series (use same as main chart)
        st.write("**Date Range:**")
        st.write(f"{start_date} to {end_date}")
        st.write("*(Uses same range as main chart)*")
    
    # Apply date filter for time series
    ts_filtered_df = df[
        (df['Date_Packed'] >= pd.to_datetime(start_date)) &
        (df['Date_Packed'] <= pd.to_datetime(end_date))
    ].copy()
    
    # Create and display time series chart
    ts_fig = create_time_series_chart(ts_filtered_df, selected_unit_size_ts, selected_customer_ts, selected_pack_house_ts)
    st.plotly_chart(ts_fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    
    total_records = len(filtered_df)
    total_sales = filtered_df['Pounds_Sold'].sum()
    total_consumed = filtered_df['Pounds_Consumed'].sum()
    overall_sell_through = (total_sales / total_consumed * 100) if total_consumed > 0 else 0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        st.metric("Total Pounds Sold", f"{total_sales:,.0f}")
    
    with col3:
        st.metric("Total Pounds Consumed", f"{total_consumed:,.0f}")
    
    with col4:
        st.metric("Overall Sell-Through Rate", f"{overall_sell_through:.1f}%")
    
    # Additional details in expander
    with st.expander("🔍 Filter Details"):
        st.write(f"**Unit Sizes:** {', '.join([f'{x} oz' for x in sorted(selected_unit_sizes)]) if selected_unit_sizes else 'None selected'}")
        st.write(f"**Pack Houses:** {', '.join(sorted(selected_pack_houses)) if selected_pack_houses else 'None selected'}")
        st.write(f"**Date Range:** {start_date} to {end_date}")
        st.write(f"**Days on Hand:** {days_on_hand_range}")
        st.write(f"**Days from Harvest to Pack:** {days_harvest_range}")
    
    # Data preview
    with st.expander("📋 Data Preview"):
        st.dataframe(filtered_df.head(100), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        💡 <strong>Tips:</strong> 
        • Bar width indicates sales volume (thicker = more sales)
        • Light grey vertical lines separate customer groups
        • Use sidebar filters to focus your analysis
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

# Instructions for deployment to Streamlit Cloud:
"""
1. Create a GitHub repository with:
   - This file (app.py)
   - Your CSV file (data_unit_customer_consumed_sold.csv)
   - requirements.txt with:
     streamlit
     pandas
     plotly
     numpy

2. Go to https://share.streamlit.io/

3. Connect your GitHub account

4. Deploy from your repository

5. Share the public URL!

The app will automatically update when you push changes to GitHub.
"""
