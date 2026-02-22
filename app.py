"""
J.P. Morgan Chase Banking Transactions and Risk Monitoring Dashboard
Senior Financial Data Engineer & Risk Consultant
Production-Ready Python Application with Advanced Forensic Analytics
ENHANCED VERSION WITH COMPLETE ANALYSIS
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
    .analysis-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .finding-card {
        background: white;
        padding: 15px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1a1a1a;
    }
    .finding-card h4 {
        color: #667eea;
        margin-top: 0;
    }
    .recommendation-card {
        background: #e3f2fd;
        padding: 15px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
        border-radius: 5px;
        color: #1a1a1a;
    }
    .recommendation-card h4 {
        color: #1565c0;
        margin-top: 0;
    }
    .recommendation-card p {
        color: #333;
        margin: 8px 0;
    }
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
    mu, sigma = 8.5, 1.8
    amounts = np.random.lognormal(mu, sigma, n_transactions)
    
    # Create probability distribution for merchants
    merchant_probs = [0.08] * 12 + [0.01] * 6
    merchant_probs = np.array(merchant_probs)
    merchant_probs = merchant_probs / merchant_probs.sum()
    
    # Create DataFrame
    df = pd.DataFrame({
        'Transaction_ID': [f'TXN{str(i).zfill(6)}' for i in range(1, n_transactions + 1)],
        'Timestamp': timestamps,
        'Amount': amounts,
        'Segment': np.random.choice(segments, n_transactions, p=[0.5, 0.25, 0.15, 0.1]),
        'Channel': np.random.choice(channels, n_transactions),
        'Region': np.random.choice(regions, n_transactions),
        'Merchant': np.random.choice(all_merchants, n_transactions, p=merchant_probs)
    })
    
    # Sort by timestamp
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    return df

def calculate_benfords_law(amounts):
    """
    Benford's Law Module - Forensic analysis to detect manual data manipulation.

    FIX: Returns raw counts (not percentages) for the chi-square test so that
    scipy.stats.chisquare receives properly scaled observed/expected frequencies.
    Percentage values are returned separately for display purposes only.
    """
    first_digits = []
    for amount in amounts:
        amount_str = str(int(amount))
        if amount_str and amount_str[0] != '0':
            first_digits.append(int(amount_str[0]))

    total = len(first_digits)
    digit_counts = Counter(first_digits)

    # Counts (for chi-square test)
    observed_counts = {d: digit_counts.get(d, 0) for d in range(1, 10)}
    expected_counts = {d: total * np.log10(1 + 1 / d) for d in range(1, 10)}

    # Percentages (for display only)
    actual_freq = {d: observed_counts[d] / total * 100 for d in range(1, 10)}
    theoretical_freq = {d: expected_counts[d] / total * 100 for d in range(1, 10)}

    return actual_freq, theoretical_freq, observed_counts, expected_counts


def calculate_z_scores(amounts):
    """Statistical Anomaly Module - Calculate Z-Scores to detect fat-tail risks"""
    z_scores = stats.zscore(amounts)
    return z_scores

def detect_ghost_hours(df):
    """Temporal Risk Module - Detect transactions during high-risk Ghost Hours"""
    df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    ghost_hours = df[(df['Hour'] >= 0) & (df['Hour'] < 5)]
    return ghost_hours

def regex_risk_mining(merchants):
    """Regex-Based Text Mining - Scan for high-risk keywords"""
    risk_patterns = [
        r'\bCasino\b', r'\bCrypto\b', r'\bShell\b',
        r'\bHaven\b', r'\bLtd\.?\b', r'\bOffshore\b', r'\bParadise\b'
    ]
    combined_pattern = '|'.join(risk_patterns)
    risk_flags = merchants.str.contains(combined_pattern, case=False, regex=True, na=False)
    return risk_flags

def calculate_compliance_score(df, critical_anomalies, ghost_transactions, risk_merchants):
    """Calculate overall compliance score (0-100)"""
    total_txn = len(df)
    anomaly_penalty = (critical_anomalies / total_txn) * 30
    ghost_penalty = (len(ghost_transactions) / total_txn) * 30
    merchant_penalty = (risk_merchants.sum() / total_txn) * 40
    score = 100 - (anomaly_penalty + ghost_penalty + merchant_penalty)
    return max(0, min(100, score))

def generate_comprehensive_analysis(filtered_df, critical_anomalies, ghost_transactions, risk_merchant_flags, z_threshold, compliance_score):
    """
    Generate complete analysis with key findings, insights, and recommendations
    """
    
    # Calculate key metrics
    total_txns = len(filtered_df)
    total_exposure = filtered_df['Amount'].sum()
    avg_transaction = filtered_df['Amount'].mean()
    median_transaction = filtered_df['Amount'].median()
    
    # Segment analysis
    segment_volumes = filtered_df['Segment'].value_counts()
    segment_exposure = filtered_df.groupby('Segment')['Amount'].sum().sort_values(ascending=False)
    dominant_segment = segment_volumes.index[0]
    dominant_segment_pct = (segment_volumes.iloc[0] / total_txns) * 100
    
    # Channel analysis
    channel_volumes = filtered_df['Channel'].value_counts()
    channel_exposure = filtered_df.groupby('Channel')['Amount'].sum().sort_values(ascending=False)
    dominant_channel = channel_volumes.index[0]
    dominant_channel_pct = (channel_volumes.iloc[0] / total_txns) * 100
    
    # Risk analysis
    risk_merchant_count = risk_merchant_flags.sum()
    risk_merchant_pct = (risk_merchant_count / total_txns) * 100
    ghost_hour_pct = (len(ghost_transactions) / total_txns) * 100
    anomaly_pct = (critical_anomalies / total_txns) * 100
    
    # ===================================================================
    # FIX: Use raw counts for chi-square test (not percentages).
    # Passing percentages to scipy.stats.chisquare was incorrect because
    # the function expects observed/expected frequencies (counts), not
    # rates. Using percentages inflated the chi-square statistic and
    # produced artificially low p-values, falsely suggesting fraud.
    # ===================================================================
    _, _, observed_counts, expected_counts = calculate_benfords_law(filtered_df['Amount'].values)
    chi_stat, p_value = stats.chisquare(
        [observed_counts[d] for d in range(1, 10)],
        [expected_counts[d] for d in range(1, 10)]
    )
    
    # Regional analysis
    region_volumes = filtered_df['Region'].value_counts()
    region_exposure = filtered_df.groupby('Region')['Amount'].sum().sort_values(ascending=False)
    
    # Temporal patterns
    filtered_df['Hour'] = pd.to_datetime(filtered_df['Timestamp']).dt.hour
    peak_hour = filtered_df['Hour'].value_counts().index[0]
    peak_hour_volume = filtered_df['Hour'].value_counts().iloc[0]
    
    # High-value transaction analysis
    high_value_threshold = filtered_df['Amount'].quantile(0.95)
    high_value_txns = len(filtered_df[filtered_df['Amount'] > high_value_threshold])
    high_value_exposure = filtered_df[filtered_df['Amount'] > high_value_threshold]['Amount'].sum()
    
    analysis = {
        'executive_summary': {
            'total_transactions': total_txns,
            'total_exposure': total_exposure,
            'compliance_score': compliance_score,
            'risk_level': 'HIGH' if compliance_score < 60 else 'MEDIUM' if compliance_score < 80 else 'LOW'
        },
        'key_findings': [
            {
                'title': '1. Transaction Volume & Exposure Analysis',
                'metrics': [
                    f"Total Transactions Analyzed: {total_txns:,}",
                    f"Total Financial Exposure: ${total_exposure:,.2f}",
                    f"Average Transaction Size: ${avg_transaction:,.2f}",
                    f"Median Transaction Size: ${median_transaction:,.2f}",
                    f"95th Percentile Threshold: ${high_value_threshold:,.2f}"
                ],
                'insight': f"The portfolio shows a lognormal distribution typical of banking transactions, with {high_value_txns} transactions ({high_value_txns/total_txns*100:.1f}%) exceeding the high-value threshold of ${high_value_threshold:,.0f}, representing ${high_value_exposure:,.0f} in exposure."
            },
            {
                'title': '2. Segment Distribution & Risk Profile',
                'metrics': [
                    f"Dominant Segment: {dominant_segment} ({dominant_segment_pct:.1f}% of volume)",
                    f"{dominant_segment} Exposure: ${segment_exposure[dominant_segment]:,.2f}",
                    f"Segment Diversity Index: {len(segment_volumes)} active segments",
                    f"Retail Banking Concentration: {segment_volumes.get('Retail', 0)/total_txns*100:.1f}%"
                ],
                'insight': f"{dominant_segment} dominates the transaction portfolio with {dominant_segment_pct:.1f}% of total volume. This concentration requires enhanced monitoring protocols for segment-specific risks and regulatory compliance requirements."
            },
            {
                'title': '3. Channel Performance & Operational Efficiency',
                'metrics': [
                    f"Primary Channel: {dominant_channel} ({dominant_channel_pct:.1f}% of transactions)",
                    f"{dominant_channel} Total Throughput: ${channel_exposure[dominant_channel]:,.2f}",
                    f"Active Channel Count: {len(channel_volumes)}",
                    f"Digital vs Traditional Split: {channel_volumes.get('Mobile', 0) + channel_volumes.get('API', 0)} digital transactions"
                ],
                'insight': f"Channel distribution shows {dominant_channel} as the predominant transaction method. The multi-channel strategy requires consistent risk controls across all platforms, with particular attention to digital channel security."
            },
            {
                'title': '4. Statistical Anomaly Detection (Z-Score Analysis)',
                'metrics': [
                    f"Critical Anomalies Detected: {critical_anomalies} transactions",
                    f"Anomaly Rate: {anomaly_pct:.2f}% of total volume",
                    f"Z-Score Threshold Applied: ±{z_threshold}σ",
                    f"Maximum Z-Score: {filtered_df['Z_Score'].abs().max():.2f}σ"
                ],
                'insight': f"Using a {z_threshold}-sigma threshold, {critical_anomalies} transactions exhibit statistical behavior significantly deviating from normal patterns. These represent fat-tail risks requiring immediate forensic review and potential escalation to compliance teams."
            },
            {
                'title': "5. Benford's Law Compliance Test",
                'metrics': [
                    f"Chi-Square Statistic: χ² = {chi_stat:.4f}",
                    f"P-Value: {p_value:.4f}",
                    f"Test Result: {'FAIL - Significant Deviation Detected' if p_value < 0.05 else 'PASS - Natural Distribution Confirmed'}",
                    f"Confidence Level: {(1-p_value)*100:.2f}%"
                ],
                'insight': f"Benford's Law analysis {'reveals significant deviations from expected first-digit distribution (p < 0.05), indicating potential data manipulation or artificial transaction patterns' if p_value < 0.05 else 'confirms natural transaction patterns align with theoretical distributions, suggesting authentic data integrity'}."
            },
            {
                'title': '6. Temporal Risk Assessment (Ghost Hours)',
                'metrics': [
                    f"Ghost Hour Transactions: {len(ghost_transactions)} (00:00-05:00)",
                    f"Ghost Hour Rate: {ghost_hour_pct:.2f}% of portfolio",
                    f"Ghost Hour Exposure: ${ghost_transactions['Amount'].sum():,.2f}",
                    f"Peak Transaction Hour: {peak_hour:02d}:00 ({peak_hour_volume} transactions)"
                ],
                'insight': f"After-hours activity analysis identifies {len(ghost_transactions)} transactions during high-risk ghost hours (midnight-5am), representing {ghost_hour_pct:.2f}% of volume. This unusual temporal pattern warrants enhanced scrutiny for potential fraud or unauthorized access."
            },
            {
                'title': '7. Merchant Risk Mining (Regex Pattern Detection)',
                'metrics': [
                    f"High-Risk Merchants Flagged: {risk_merchant_count}",
                    f"Risk Merchant Exposure Rate: {risk_merchant_pct:.2f}%",
                    f"Total Risk Merchant Exposure: ${filtered_df[risk_merchant_flags]['Amount'].sum():,.2f}",
                    f"Risk Patterns Monitored: 7 keyword categories"
                ],
                'insight': f"Text mining algorithms identified {risk_merchant_count} transactions with high-risk merchant patterns (casinos, crypto exchanges, offshore entities). These merchants require enhanced due diligence and continuous KYC monitoring."
            },
            {
                'title': '8. Geographic Distribution Analysis',
                'metrics': [
                    f"Active Regions: {len(region_volumes)}",
                    f"Dominant Region: {region_volumes.index[0]} ({region_volumes.iloc[0]/total_txns*100:.1f}%)",
                    f"Region with Highest Exposure: {region_exposure.index[0]} (${region_exposure.iloc[0]:,.2f})",
                    f"Geographic Concentration Risk: {region_volumes.iloc[0]/total_txns*100:.1f}%"
                ],
                'insight': f"Geographic analysis reveals concentration in {region_volumes.index[0]} region. Cross-border transactions require enhanced sanctions screening and compliance with regional regulatory frameworks."
            }
        ],
        'strategic_recommendations': [
            {
                'priority': 'CRITICAL',
                'title': 'Immediate Anomaly Investigation',
                'action': f"Deploy forensic analysis team to investigate {critical_anomalies} critical anomalies flagged by Z-score analysis (|Z| > {z_threshold}). Prioritize transactions with Z-scores exceeding 4σ for immediate escalation.",
                'timeline': 'Within 24 hours',
                'impact': 'High - Potential fraud prevention'
            },
            {
                'priority': 'HIGH',
                'title': 'Ghost Hours Monitoring Enhancement',
                'action': f"Implement real-time alerting for transactions during 00:00-05:00 hours. Current ghost hour rate of {ghost_hour_pct:.2f}% requires automated suspicious activity reports (SARs) generation.",
                'timeline': 'Within 72 hours',
                'impact': 'High - Fraud detection improvement'
            },
            {
                'priority': 'HIGH',
                'title': 'Risk Merchant Enhanced Due Diligence',
                'action': f"Conduct comprehensive KYC review for {risk_merchant_count} flagged merchants. Implement enhanced transaction monitoring for all crypto, casino, and offshore-related entities.",
                'timeline': 'Within 1 week',
                'impact': 'Medium - Compliance risk mitigation'
            },
            {
                'priority': 'MEDIUM',
                'title': "Benford's Law Investigation",
                'action': "Launch data integrity audit to investigate deviations from Benford's Law distribution. Review transaction origination processes for potential manual entry errors or systematic manipulation." if p_value < 0.05 else "Continue monitoring first-digit distributions quarterly to maintain data integrity baseline.",
                'timeline': '2-4 weeks',
                'impact': 'Medium - Data quality assurance'
            },
            {
                'priority': 'MEDIUM',
                'title': 'Segment-Specific Risk Controls',
                'action': f"Develop targeted risk models for {dominant_segment} segment which represents {dominant_segment_pct:.1f}% of volume. Implement segment-specific transaction limits and approval workflows.",
                'timeline': '1 month',
                'impact': 'Medium - Operational efficiency'
            },
            {
                'priority': 'LOW',
                'title': 'Channel Security Audit',
                'action': f"Conduct security assessment of {dominant_channel} channel infrastructure. Ensure consistent authentication and authorization protocols across all channels.",
                'timeline': '6-8 weeks',
                'impact': 'Low - Preventive security measure'
            }
        ],
        'risk_score_breakdown': {
            'compliance_score': compliance_score,
            'anomaly_contribution': (anomaly_pct / 100) * 30,
            'ghost_hour_contribution': (ghost_hour_pct / 100) * 30,
            'merchant_risk_contribution': (risk_merchant_pct / 100) * 40
        },
        # Pass through for use in the temporal tab
        'chi_stat': chi_stat,
        'p_value': p_value,
        'ghost_hour_pct': ghost_hour_pct,
        'peak_hour': peak_hour,
    }
    
    return analysis

def display_executive_summary(analysis):
    """Display executive summary section"""
    st.markdown("## 📋 Executive Summary & Action Plan")
    
    summary = analysis['executive_summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{summary['total_transactions']:,}")
    
    with col2:
        st.metric("Total Exposure", f"${summary['total_exposure']/1e6:.1f}M")
    
    with col3:
        st.metric("Compliance Score", f"{summary['compliance_score']:.1f}/100")
    
    with col4:
        risk_color = "🔴" if summary['risk_level'] == 'HIGH' else "🟡" if summary['risk_level'] == 'MEDIUM' else "🟢"
        st.metric("Risk Level", f"{risk_color} {summary['risk_level']}")
    
    st.markdown("---")

def display_key_findings(analysis):
    """Display key operational findings"""
    with st.expander("🔍 **View Complete Analysis & Strategic Recommendations**", expanded=True):
        st.markdown("### 🔬 Key Operational Findings")
        
        for finding in analysis['key_findings']:
            st.markdown(f"""
            <div class="finding-card">
                <h4>{finding['title']}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Key Metrics:**")
                for metric in finding['metrics']:
                    st.markdown(f"• {metric}")
            
            with col2:
                st.markdown("**Analysis:**")
                st.write(finding['insight'])
            
            st.markdown("---")

def display_strategic_recommendations(analysis):
    """Display strategic recommendations"""
    st.markdown("### 🎯 Strategic Recommendations & Action Items")
    
    # Group by priority
    critical = [r for r in analysis['strategic_recommendations'] if r['priority'] == 'CRITICAL']
    high = [r for r in analysis['strategic_recommendations'] if r['priority'] == 'HIGH']
    medium = [r for r in analysis['strategic_recommendations'] if r['priority'] == 'MEDIUM']
    low = [r for r in analysis['strategic_recommendations'] if r['priority'] == 'LOW']
    
    for priority_group, items in [('🔴 CRITICAL PRIORITY', critical), 
                                   ('🟠 HIGH PRIORITY', high),
                                   ('🟡 MEDIUM PRIORITY', medium),
                                   ('🟢 LOW PRIORITY', low)]:
        if items:
            st.markdown(f"#### {priority_group}")
            for rec in items:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{rec['title']}</h4>
                    <p><strong>Action:</strong> {rec['action']}</p>
                    <p><strong>Timeline:</strong> {rec['timeline']} | <strong>Impact:</strong> {rec['impact']}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("")

def main():
    st.title("🏦 J.P. Morgan Chase Risk Monitoring Dashboard")
    st.markdown("### Advanced Forensic Analytics for Banking Transactions")
    
    # Generate Data
    df = generate_synthetic_transactions(10000)
    
    # Sidebar Controls
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <svg width="180" height="60" viewBox="0 0 180 60" xmlns="http://www.w3.org/2000/svg">
                <rect x="0" y="0" width="180" height="60" fill="#117ACA" rx="5"/>
                <text x="90" y="25" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="white" text-anchor="middle">J.P. Morgan</text>
                <text x="90" y="45" font-family="Arial, sans-serif" font-size="16" fill="white" text-anchor="middle">Chase & Co.</text>
            </svg>
        </div>
    """, unsafe_allow_html=True)
    
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
    
    # Generate comprehensive analysis
    analysis = generate_comprehensive_analysis(
        filtered_df, 
        critical_anomalies, 
        ghost_transactions, 
        risk_merchant_flags, 
        z_threshold,
        compliance_score
    )
    
    # Display Executive Summary
    display_executive_summary(analysis)
    
    # Display Key Findings and Recommendations
    display_key_findings(analysis)
    display_strategic_recommendations(analysis)
    
    st.markdown("---")
    
    # Tab Layout for Detailed Analysis
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
        amount_hist, bin_edges = np.histogram(filtered_df['Amount'], bins=50)
        amount_df = pd.DataFrame({
            'Count': amount_hist
        })
        st.line_chart(amount_df)
    
    with tab2:
        st.subheader("🔬 Benford's Law Analysis")
        st.markdown("""
        Benford's Law states that in naturally occurring datasets, the first digit follows a logarithmic distribution.
        Significant deviations may indicate manual manipulation or fraud.
        """)
        
        # ===================================================================
        # FIX: Use the updated function signature that returns counts too.
        # Display percentages to the user, but run the chi-square test on
        # raw counts (see calculate_benfords_law docstring for explanation).
        # ===================================================================
        actual_freq, theoretical_freq, observed_counts, expected_counts = calculate_benfords_law(
            filtered_df['Amount'].values
        )
        
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
        
        # Chi-Square Goodness of Fit Test — using counts, not percentages
        chi_stat, p_value = stats.chisquare(
            [observed_counts[d] for d in range(1, 10)],
            [expected_counts[d] for d in range(1, 10)]
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
        
        # Ensure Hour column is present
        filtered_df['Hour'] = pd.to_datetime(filtered_df['Timestamp']).dt.hour
        filtered_df['DayOfWeek'] = pd.to_datetime(filtered_df['Timestamp']).dt.day_name()

        # Hourly Distribution
        hourly_counts = filtered_df['Hour'].value_counts().sort_index()
        
        st.markdown("**Transaction Volume by Hour of Day**")
        st.bar_chart(hourly_counts)

        st.markdown("---")

        # ===================================================================
        # NEW: Ghost Hour Heatmap — Hour × Day-of-Week transaction volume.
        # This makes the temporal risk pattern much more visual and
        # immediately digestible for anyone reviewing the dashboard.
        # ===================================================================
        st.markdown("### 🔥 Transaction Heatmap: Hour × Day of Week")
        st.caption("Darker cells = higher volume. Ghost hours (00:00–05:00) highlighted in context.")

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = (
            filtered_df.groupby(['DayOfWeek', 'Hour'])
            .size()
            .reset_index(name='Count')
        )

        # Pivot to Hour × Day matrix
        heatmap_pivot = heatmap_data.pivot(index='Hour', columns='DayOfWeek', values='Count').fillna(0)
        # Reorder columns to Monday–Sunday
        heatmap_pivot = heatmap_pivot.reindex(
            columns=[d for d in day_order if d in heatmap_pivot.columns]
        )

        st.dataframe(
            heatmap_pivot.style.background_gradient(cmap='Reds', axis=None),
            use_container_width=True,
            height=600
        )

        st.markdown("---")

        # Ghost Hours Analysis
        st.markdown("### 👻 Ghost Hours Detail (00:00 – 05:00)")
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
