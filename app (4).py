"""
J.P. Morgan Chase Banking Transactions and Risk Monitoring Dashboard
Senior Financial Data Engineer & Risk Consultant
Production-Ready Python Application with Advanced Forensic Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from scipy import stats
from collections import Counter

# Page Configuration
st.set_page_config(
    page_title="JPM Risk Monitor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ffaa00; font-weight: bold; }
    .risk-low { color: #00C851; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_transactions(n_transactions=10000):
    """
    Advanced Data Simulation Engine
    Generates high-fidelity synthetic banking transactions with statistical rigor
    """
    np.random.seed(42)
    
    # Temporal Distribution: 180 days of historical data
    start_date = datetime.now() - timedelta(days=180)
    timestamps = [
        start_date + timedelta(
            days=np.random.randint(0, 180),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        ) for _ in range(n_transactions)
    ]
    
    # Multi-Dimensional Categories
    segments = ['Retail', 'Corporate', 'Private Banking', 'Institutional']
    channels = ['SWIFT', 'API', 'ATM', 'Wire Transfer', 'ACH', 'Mobile']
    regions = ['EMEA', 'APAC', 'Americas', 'LATAM', 'Middle East']
    
    # Merchant Pool with Risk-Flagged Keywords
    merchant_base = [
        'Global Trade Corp', 'Acme Industries', 'Tech Solutions Ltd',
        'Retail Enterprises', 'Manufacturing Inc', 'Services Group',
        'Consulting Partners', 'Import Export Co', 'Digital Ventures',
        'Financial Services', 'Healthcare Systems', 'Energy Corp'
    ]
    
    # High-Risk Keywords for Regex Detection
    risk_merchants = [
        'Golden Casino Resort', 'Crypto Exchange Ltd', 'Shell Trading Haven',
        'Offshore Holdings Ltd', 'Digital Coin Services', 'Paradise Haven Corp'
    ]
    
    all_merchants = merchant_base + risk_merchants
    
    # Lognormal Distribution for Realistic Transaction Amounts
    # mu and sigma tuned for realistic banking transactions
    mu, sigma = 8.5, 1.8
    amounts = np.random.lognormal(mu, sigma, n_transactions)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Transaction_ID': [f'TXN{str(i).zfill(6)}' for i in range(1, n_transactions + 1)],
        'Timestamp': timestamps,
        'Amount': amounts,
        'Segment': np.random.choice(segments, n_transactions, p=[0.5, 0.25, 0.15, 0.1]),
        'Channel': np.random.choice(channels, n_transactions),
        'Region': np.random.choice(regions, n_transactions),
        'Merchant': np.random.choice(all_merchants, n_transactions, p=[0.08]*12 + [0.01]*6)
    })
    
    # Sort by timestamp
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    return df

def calculate_benfords_law(amounts):
    """
    Benford's Law Module
    Forensic analysis to detect manual data manipulation
    """
    # Extract first digit using string manipulation
    first_digits = []
    for amount in amounts:
        amount_str = str(int(amount))
        if amount_str and amount_str[0] != '0':
            first_digits.append(int(amount_str[0]))
    
    # Calculate actual frequencies
    digit_counts = Counter(first_digits)
    total = len(first_digits)
    actual_freq = {d: (digit_counts.get(d, 0) / total * 100) for d in range(1, 10)}
    
    # Benford's Law theoretical frequencies
    theoretical_freq = {d: np.log10(1 + 1/d) * 100 for d in range(1, 10)}
    
    return actual_freq, theoretical_freq

def calculate_z_scores(amounts):
    """
    Statistical Anomaly Module
    Calculate Z-Scores to detect fat-tail risks
    """
    z_scores = stats.zscore(amounts)
    return z_scores

def detect_ghost_hours(df):
    """
    Temporal Risk Module
    Detect transactions during high-risk "Ghost Hours" (Midnight to 5 AM)
    """
    df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    ghost_hours = df[(df['Hour'] >= 0) & (df['Hour'] < 5)]
    return ghost_hours

def regex_risk_mining(merchants):
    """
    Regex-Based Text Mining
    Scan for high-risk keywords using pattern matching
    """
    risk_patterns = [
        r'\bCasino\b',
        r'\bCrypto\b',
        r'\bShell\b',
        r'\bHaven\b',
        r'\bLtd\.?\b',
        r'\bOffshore\b',
        r'\bParadise\b'
    ]
    
    combined_pattern = '|'.join(risk_patterns)
    risk_flags = merchants.str.contains(combined_pattern, case=False, regex=True, na=False)
    
    return risk_flags

def calculate_compliance_score(df, critical_anomalies, ghost_transactions, risk_merchants):
    """
    Calculate overall compliance score (0-100)
    Lower score = higher risk
    """
    total_txn = len(df)
    
    anomaly_penalty = (critical_anomalies / total_txn) * 30
    ghost_penalty = (len(ghost_transactions) / total_txn) * 30
    merchant_penalty = (risk_merchants.sum() / total_txn) * 40
    
    score = 100 - (anomaly_penalty + ghost_penalty + merchant_penalty)
    return max(0, min(100, score))

# Main Application
def main():
    st.title("🏦 J.P. Morgan Chase Risk Monitoring Dashboard")
    st.markdown("### Advanced Forensic Analytics for Banking Transactions")
    
    # Generate Data
    df = generate_synthetic_transactions(10000)
    
    # Sidebar Controls
    st.sidebar.header("🎛️ Control Panel")
    st.sidebar.markdown("---")
    
    # Filters
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique().tolist(),
        default=df['Region'].unique().tolist()
    )
    
    selected_segments = st.sidebar.multiselect(
        "Select Segments",
        options=df['Segment'].unique().tolist(),
        default=df['Segment'].unique().tolist()
    )
    
    selected_channels = st.sidebar.multiselect(
        "Select Channels",
        options=df['Channel'].unique().tolist(),
        default=df['Channel'].unique().tolist()
    )
    
    st.sidebar.markdown("---")
    
    # Risk Appetite Slider
    z_threshold = st.sidebar.slider(
        "Risk Sensitivity (Z-Score Threshold)",
        min_value=2.0,
        max_value=4.0,
        value=3.0,
        step=0.1,
        help="Higher values = less sensitive to anomalies"
    )
    
    # Apply Filters
    filtered_df = df[
        (df['Region'].isin(selected_regions)) &
        (df['Segment'].isin(selected_segments)) &
        (df['Channel'].isin(selected_channels))
    ].copy()
    
    # Calculate Analytics
    z_scores = calculate_z_scores(filtered_df['Amount'].values)
    filtered_df['Z_Score'] = z_scores
    critical_anomalies = np.sum(np.abs(z_scores) > z_threshold)
    
    ghost_transactions = detect_ghost_hours(filtered_df)
    risk_merchant_flags = regex_risk_mining(filtered_df['Merchant'])
    filtered_df['Risk_Merchant'] = risk_merchant_flags
    
    compliance_score = calculate_compliance_score(
        filtered_df,
        critical_anomalies,
        ghost_transactions,
        risk_merchant_flags
    )
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_exposure = filtered_df['Amount'].sum()
        st.metric(
            "Total Exposure",
            f"${total_exposure:,.0f}",
            delta=f"{len(filtered_df)} txns"
        )
    
    with col2:
        st.metric(
            "Critical Anomalies",
            f"{critical_anomalies}",
            delta=f"{critical_anomalies/len(filtered_df)*100:.2f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Ghost Hour Transactions",
            f"{len(ghost_transactions)}",
            delta=f"{len(ghost_transactions)/len(filtered_df)*100:.2f}%",
            delta_color="inverse"
        )
    
    with col4:
        score_color = "normal" if compliance_score >= 70 else "inverse"
        st.metric(
            "Compliance Score",
            f"{compliance_score:.1f}/100",
            delta="Good" if compliance_score >= 70 else "Review Required",
            delta_color=score_color
        )
    
    st.markdown("---")
    
    # Tab Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔬 Benford's Law",
        "⚡ Anomaly Detection",
        "🕐 Temporal Analysis",
        "🚨 Audit Trail"
    ])
    
    with tab1:
        st.subheader("Transaction Distribution Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Volume by Segment**")
            segment_data = filtered_df['Segment'].value_counts()
            st.bar_chart(segment_data)
        
        with col2:
            st.markdown("**Volume by Region**")
            region_data = filtered_df['Region'].value_counts()
            st.bar_chart(region_data)
        
        st.markdown("**Transaction Amount Distribution**")
        amount_bins = pd.cut(filtered_df['Amount'], bins=50)
        amount_hist = amount_bins.value_counts().sort_index()
        st.line_chart(amount_hist)
    
    with tab2:
        st.subheader("🔬 Benford's Law Analysis")
        st.markdown("""
        Benford's Law states that in naturally occurring datasets, the first digit follows a logarithmic distribution.
        Significant deviations may indicate manual manipulation or fraud.
        """)
        
        actual_freq, theoretical_freq = calculate_benfords_law(filtered_df['Amount'].values)
        
        benford_df = pd.DataFrame({
            'Digit': range(1, 10),
            'Actual %': [actual_freq[d] for d in range(1, 10)],
            'Theoretical %': [theoretical_freq[d] for d in range(1, 10)]
        })
        
        st.dataframe(benford_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Actual Distribution**")
            st.bar_chart(benford_df.set_index('Digit')['Actual %'])
        
        with col2:
            st.markdown("**Theoretical Distribution**")
            st.bar_chart(benford_df.set_index('Digit')['Theoretical %'])
        
        # Chi-Square Goodness of Fit Test
        chi_stat, p_value = stats.chisquare(
            [actual_freq[d] for d in range(1, 10)],
            [theoretical_freq[d] for d in range(1, 10)]
        )
        
        st.info(f"**Chi-Square Test**: χ² = {chi_stat:.4f}, p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            st.warning("⚠️ Significant deviation detected! Data may not follow natural patterns.")
        else:
            st.success("✅ Data distribution aligns with Benford's Law.")
    
    with tab3:
        st.subheader("⚡ Statistical Anomaly Detection (Z-Score Analysis)")
        st.markdown(f"""
        Transactions with |Z-Score| > {z_threshold} are flagged as critical anomalies.
        These represent statistical outliers in the fat-tail distribution.
        """)
        
        # Anomaly DataFrame
        anomalies = filtered_df[np.abs(filtered_df['Z_Score']) > z_threshold].copy()
        anomalies = anomalies.sort_values('Z_Score', key=abs, ascending=False)
        
        st.metric("Total Anomalies Detected", len(anomalies))
        
        if len(anomalies) > 0:
            # Z-Score Distribution
            st.markdown("**Z-Score Distribution**")
            z_score_df = pd.DataFrame({
                'Z_Score': filtered_df['Z_Score']
            })
            st.line_chart(z_score_df)
            
            # Top Anomalies
            st.markdown("**Top 10 Critical Anomalies**")
            anomaly_display = anomalies[['Transaction_ID', 'Timestamp', 'Amount', 'Z_Score', 'Segment', 'Merchant']].head(10)
            st.dataframe(anomaly_display, use_container_width=True)
        else:
            st.success("No critical anomalies detected at current threshold.")
    
    with tab4:
        st.subheader("🕐 Temporal Risk Analysis")
        
        # Hourly Distribution
        hourly_counts = filtered_df['Hour'].value_counts().sort_index()
        
        st.markdown("**Transaction Volume by Hour**")
        st.bar_chart(hourly_counts)
        
        # Ghost Hours Analysis
        st.markdown("### 👻 Ghost Hours (00:00 - 05:00)")
        st.warning(f"**{len(ghost_transactions)}** transactions detected during ghost hours")
        
        if len(ghost_transactions) > 0:
            ghost_display = ghost_transactions[['Transaction_ID', 'Timestamp', 'Amount', 'Segment', 'Merchant']].head(20)
            st.dataframe(ghost_display, use_container_width=True)
            
            ghost_exposure = ghost_transactions['Amount'].sum()
            st.metric("Ghost Hours Exposure", f"${ghost_exposure:,.0f}")
    
    with tab5:
        st.subheader("🚨 High-Risk Transaction Audit Trail")
        
        # Combine all risk factors
        high_risk = filtered_df[
            (np.abs(filtered_df['Z_Score']) > z_threshold) |
            (filtered_df['Risk_Merchant']) |
            ((filtered_df['Hour'] >= 0) & (filtered_df['Hour'] < 5))
        ].copy()
        
        high_risk['Risk_Factors'] = high_risk.apply(
            lambda row: ', '.join([
                'Z-Score Anomaly' if abs(row['Z_Score']) > z_threshold else '',
                'Risk Merchant' if row['Risk_Merchant'] else '',
                'Ghost Hour' if 0 <= row['Hour'] < 5 else ''
            ]).strip(', '),
            axis=1
        )
        
        st.metric("Total High-Risk Transactions", len(high_risk))
        
        # Search functionality
        search_term = st.text_input("🔍 Search by Transaction ID or Merchant")
        
        if search_term:
            high_risk = high_risk[
                high_risk['Transaction_ID'].str.contains(search_term, case=False) |
                high_risk['Merchant'].str.contains(search_term, case=False)
            ]
        
        # Display high-risk transactions
        audit_display = high_risk[['Transaction_ID', 'Timestamp', 'Amount', 'Segment', 'Merchant', 'Risk_Factors']].sort_values('Amount', ascending=False)
        
        st.dataframe(
            audit_display,
            use_container_width=True,
            height=400
        )
        
        # Export capability
        if len(high_risk) > 0:
            csv = audit_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Audit Trail (CSV)",
                data=csv,
                file_name="jpmc_risk_audit_trail.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <small>J.P. Morgan Chase Risk Monitoring Dashboard | Advanced Forensic Analytics Engine</small>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
