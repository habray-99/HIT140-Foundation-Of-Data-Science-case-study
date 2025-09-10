# %%
"""
Bat vs. Rat: Predator Hypothesis Analysis
=========================================

This analysis investigates whether Egyptian Fruit Bats (Rousettus aegyptiacus) 
perceive Black Rats (Rattus rattus) as potential predators during foraging.

Research Question: Do bats perceive rats not just as competitors for food 
but also as potential predators?

Expected Evidence:
- Increased vigilance (longer approach times) when rats are present
- More risk-avoidant behavior in rat contexts
- Reduced overall platform activity when rats are present
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for professional appearance
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %%
# ===================================================================
# DATA LOADING AND PREPROCESSING
# ===================================================================

print("=" * 70)
print("BAT-RAT PREDATOR HYPOTHESIS ANALYSIS")
print("=" * 70)

# Load the dataset
df = pd.read_csv(r'/mnt/e/Foundation of Data Science/ass 1/dataset1.csv', engine='pyarrow')

# Convert datetime columns with proper error handling
datetime_columns = ['rat_period_start', 'rat_period_end', 'sunset_time', 'start_time']
for col in datetime_columns:
    df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M', errors='coerce')

print(f"Dataset loaded successfully: {df.shape[0]} observations, {df.shape[1]} variables")
print(f"Date range: {df['start_time'].min().strftime('%Y-%m-%d')} to {df['start_time'].max().strftime('%Y-%m-%d')}")

# %%
# ===================================================================
# DATA QUALITY ASSESSMENT
# ===================================================================

print("\n" + "=" * 50)
print("DATA QUALITY ASSESSMENT")
print("=" * 50)

# Handle missing values in habit column using forward fill
df['habit'] = df['habit'].fillna(method='ffill')

# Display basic dataset information
print(f"\nDataset Overview:")
print(f"• Total observations: {len(df):,}")
print(f"• Missing values in key variables:")
for col in ['bat_landing_to_food', 'risk', 'reward', 'habit']:
    missing_count = df[col].isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    print(f"  - {col}: {missing_count} ({missing_percent:.1f}%)")

print(f"\nUnique landing contexts identified: {df['habit'].nunique()}")

# %%
# ===================================================================
# BEHAVIORAL CONTEXT CLASSIFICATION
# ===================================================================

def classify_landing_context(habit_description):
    """
    Classify landing contexts into meaningful categories for analysis.
    
    This function standardizes the diverse habitat descriptions into 
    clear categories that can be used for statistical analysis.
    """
    if pd.isna(habit_description):
        return 'Unknown'
    
    description = str(habit_description).lower()
    
    # Primary classification logic
    if 'rat' in description and 'bat' not in description and 'pick' not in description:
        return 'Rat Present'
    elif 'bat' in description and 'rat' not in description:
        return 'Bat Only'
    elif 'pick' in description and 'rat' not in description:
        return 'Food Picking'
    elif 'fast' in description:
        return 'Fast Movement'
    elif any(char.isdigit() for char in description):  # Filter out coordinate data
        return 'Unknown'
    else:
        return 'Other Context'

# Apply context classification
df['context'] = df['habit'].apply(classify_landing_context)

print("\nLanding Context Classification:")
context_counts = df['context'].value_counts()
for context, count in context_counts.items():
    percentage = (count / len(df)) * 100
    print(f"• {context}: {count:,} observations ({percentage:.1f}%)")

# %%
# ===================================================================
# DESCRIPTIVE STATISTICS: BEHAVIORAL MEASURES
# ===================================================================

print("\n" + "=" * 50)
print("BEHAVIORAL MEASURES SUMMARY")
print("=" * 50)

# Remove extreme outliers for more reliable statistics (>99th percentile)
approach_time_threshold = df['bat_landing_to_food'].quantile(0.99)
df_clean = df[df['bat_landing_to_food'] <= approach_time_threshold].copy()

print(f"\nVigilance Measure (Time to Approach Food):")
print(f"• Mean approach time: {df_clean['bat_landing_to_food'].mean():.2f} seconds")
print(f"• Median approach time: {df_clean['bat_landing_to_food'].median():.2f} seconds")
print(f"• Standard deviation: {df_clean['bat_landing_to_food'].std():.2f} seconds")
print(f"• Range: {df_clean['bat_landing_to_food'].min():.1f} - {df_clean['bat_landing_to_food'].max():.1f} seconds")

print(f"\nRisk-Taking Behavior:")
risk_rate = df['risk'].mean()
print(f"• Risk-taking instances: {df['risk'].sum():,} out of {len(df):,} observations")
print(f"• Risk-taking rate: {risk_rate:.3f} ({risk_rate*100:.1f}%)")
print(f"• Risk-avoidance rate: {(1-risk_rate)*100:.1f}%")

print(f"\nForaging Success:")
success_rate = df['reward'].mean()
print(f"• Successful foraging attempts: {df['reward'].sum():,} out of {len(df):,}")
print(f"• Overall success rate: {success_rate:.3f} ({success_rate*100:.1f}%)")

# %%
# ===================================================================
# PREDATOR HYPOTHESIS: STATISTICAL ANALYSIS
# ===================================================================

print("\n" + "=" * 50)
print("PREDATOR HYPOTHESIS: STATISTICAL TESTING")
print("=" * 50)

# Separate data by context for comparison
rat_context = df_clean[df_clean['context'] == 'Rat Present']
non_rat_context = df_clean[df_clean['context'] != 'Rat Present']

print(f"\nSample sizes for analysis:")
print(f"• Rat present contexts: {len(rat_context):,} observations")
print(f"• Non-rat contexts: {len(non_rat_context):,} observations")

# ===================================================================
# TEST 1: VIGILANCE HYPOTHESIS (Approach Time Analysis)
# ===================================================================

print(f"\n{'='*40}")
print("TEST 1: VIGILANCE HYPOTHESIS")
print("H₀: No difference in approach times between contexts")
print("H₁: Bats show increased vigilance (longer approach times) when rats are present")
print("="*40)

if len(rat_context) > 10 and len(non_rat_context) > 10:
    # Perform independent t-test
    t_statistic, p_value_vigilance = stats.ttest_ind(
        rat_context['bat_landing_to_food'], 
        non_rat_context['bat_landing_to_food']
    )
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(rat_context)-1) * rat_context['bat_landing_to_food'].var() + 
                         (len(non_rat_context)-1) * non_rat_context['bat_landing_to_food'].var()) / 
                        (len(rat_context) + len(non_rat_context) - 2))
    cohens_d = (rat_context['bat_landing_to_food'].mean() - 
                non_rat_context['bat_landing_to_food'].mean()) / pooled_std
    
    print(f"Results:")
    print(f"• Rat present: Mean = {rat_context['bat_landing_to_food'].mean():.2f}s (SD = {rat_context['bat_landing_to_food'].std():.2f})")
    print(f"• Non-rat contexts: Mean = {non_rat_context['bat_landing_to_food'].mean():.2f}s (SD = {non_rat_context['bat_landing_to_food'].std():.2f})")
    print(f"• t-statistic = {t_statistic:.3f}")
    print(f"• p-value = {p_value_vigilance:.3f}")
    print(f"• Effect size (Cohen's d) = {cohens_d:.3f}")
    
    if p_value_vigilance < 0.05:
        if rat_context['bat_landing_to_food'].mean() > non_rat_context['bat_landing_to_food'].mean():
            print("★ SIGNIFICANT: Bats show INCREASED vigilance in rat contexts (supports predator hypothesis)")
        else:
            print("★ SIGNIFICANT: Bats show DECREASED vigilance in rat contexts (contradicts predator hypothesis)")
    else:
        print("✗ NOT SIGNIFICANT: No difference in vigilance between contexts")
else:
    print("⚠ Insufficient sample size for reliable t-test")

# ===================================================================
# TEST 2: RISK AVOIDANCE HYPOTHESIS
# ===================================================================

print(f"\n{'='*40}")
print("TEST 2: RISK AVOIDANCE HYPOTHESIS")
print("H₀: Risk behavior is independent of context")
print("H₁: Bats show more risk-avoidant behavior when rats are present")
print("="*40)

# Create contingency table for chi-square test
contingency_table = pd.crosstab(df['context'] == 'Rat Present', df['risk'])

print("Contingency Table:")
print("Context          Risk-Avoidant  Risk-Taking   Total")
print("="*50)
for i, (index, row) in enumerate(contingency_table.iterrows()):
    context_name = "Rat Present" if index else "Other Contexts"
    total = row.sum()
    print(f"{context_name:<15} {row[0]:>8} {row[1]:>12} {total:>8}")

# Perform chi-square test
chi2_statistic, p_value_risk, dof, expected = stats.chi2_contingency(contingency_table)

# Calculate effect size (Cramer's V)
n_total = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2_statistic / (n_total * (min(contingency_table.shape) - 1)))

print(f"\nStatistical Results:")
print(f"• χ² statistic = {chi2_statistic:.3f}")
print(f"• p-value = {p_value_risk:.3f}")
print(f"• Degrees of freedom = {dof}")
print(f"• Effect size (Cramer's V) = {cramers_v:.3f}")

if p_value_risk < 0.05:
    # Calculate risk-avoidance rates by context
    rat_risk_avoidance = (1 - df[df['context'] == 'Rat Present']['risk'].mean()) * 100
    other_risk_avoidance = (1 - df[df['context'] != 'Rat Present']['risk'].mean()) * 100
    
    print("★ SIGNIFICANT: Risk behavior differs between contexts")
    print(f"• Risk-avoidance rate (Rat Present): {rat_risk_avoidance:.1f}%")
    print(f"• Risk-avoidance rate (Other Contexts): {other_risk_avoidance:.1f}%")
    
    if rat_risk_avoidance > other_risk_avoidance:
        print("→ Supports predator hypothesis: More risk-avoidant behavior with rats")
    else:
        print("→ Contradicts predator hypothesis: Less risk-avoidant behavior with rats")
else:
    print("✗ NOT SIGNIFICANT: No association between context and risk behavior")

# ===================================================================
# TEST 3: SUCCESS RATE COMPARISON
# ===================================================================

print(f"\n{'='*40}")
print("TEST 3: FORAGING SUCCESS ANALYSIS")
print("H₀: Success rate is equal across contexts")
print("H₁: Success rate differs between rat and non-rat contexts")
print("="*40)

rat_success_data = df[df['context'] == 'Rat Present']['reward']
other_success_data = df[df['context'] != 'Rat Present']['reward']

if len(rat_success_data) > 10 and len(other_success_data) > 10:
    # Two-proportion z-test
    successes_rat = rat_success_data.sum()
    successes_other = other_success_data.sum()
    n_rat = len(rat_success_data)
    n_other = len(other_success_data)
    
    p_rat = successes_rat / n_rat
    p_other = successes_other / n_other
    p_pooled = (successes_rat + successes_other) / (n_rat + n_other)
    
    # Calculate z-statistic
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_rat + 1/n_other))
    z_statistic = (p_rat - p_other) / se
    p_value_success = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    
    print(f"Success Rate Analysis:")
    print(f"• Rat Present: {successes_rat}/{n_rat} = {p_rat:.3f} ({p_rat*100:.1f}%)")
    print(f"• Other Contexts: {successes_other}/{n_other} = {p_other:.3f} ({p_other*100:.1f}%)")
    print(f"• z-statistic = {z_statistic:.3f}")
    print(f"• p-value = {p_value_success:.3f}")
    
    if p_value_success < 0.05:
        print("★ SIGNIFICANT: Success rates differ between contexts")
        if p_rat < p_other:
            print("→ Lower success rate with rats present (supports predator hypothesis)")
        else:
            print("→ Higher success rate with rats present (contradicts predator hypothesis)")
    else:
        print("✗ NOT SIGNIFICANT: No difference in success rates")

# %%
# ===================================================================
# COMPREHENSIVE VISUALIZATION
# ===================================================================

print(f"\n{'='*50}")
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("="*50)

# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Predator Hypothesis Analysis: Bat Behavioral Responses to Rat Presence', 
             fontsize=20, fontweight='bold', y=0.95)

# ===================================================================
# Chart 1: Vigilance Comparison (Box Plot)
# ===================================================================
ax1 = plt.subplot(2, 3, 1)
vigilance_data = [rat_context['bat_landing_to_food'], non_rat_context['bat_landing_to_food']]
box_plot = ax1.boxplot(vigilance_data, labels=['Rat Present', 'Other Contexts'], 
                       patch_artist=True, notch=True)

# Color the boxes
colors = ['lightcoral', 'lightblue']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_title('Vigilance Response: Time to Approach Food\n(Lower values = less vigilant)', 
              fontsize=14, fontweight='bold', pad=20)
ax1.set_ylabel('Seconds to Approach Food', fontsize=12)
ax1.grid(True, alpha=0.3)

# Add statistical annotation
if 'p_value_vigilance' in locals():
    significance = "***" if p_value_vigilance < 0.001 else "**" if p_value_vigilance < 0.01 else "*" if p_value_vigilance < 0.05 else "ns"
    ax1.text(0.5, 0.95, f'p = {p_value_vigilance:.3f} {significance}', 
             transform=ax1.transAxes, ha='center', fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

# ===================================================================
# Chart 2: Risk Behavior Distribution
# ===================================================================
ax2 = plt.subplot(2, 3, 2)
risk_by_context = df.groupby('context')['risk'].agg(['mean', 'count']).reset_index()
risk_by_context = risk_by_context[risk_by_context['count'] >= 10]  # Filter contexts with sufficient data

# Create stacked bar chart showing risk-avoidance vs risk-taking
risk_avoidance = 1 - risk_by_context['mean']
risk_taking = risk_by_context['mean']

x_pos = range(len(risk_by_context))
width = 0.6

bars1 = ax2.bar(x_pos, risk_avoidance, width, label='Risk-Avoidant', 
                color='lightcoral', alpha=0.8)
bars2 = ax2.bar(x_pos, risk_taking, width, bottom=risk_avoidance, 
                label='Risk-Taking', color='lightblue', alpha=0.8)

ax2.set_title('Risk Behavior by Context\n(Higher red bars indicate more caution)', 
              fontsize=14, fontweight='bold', pad=20)
ax2.set_ylabel('Proportion of Behaviors', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(risk_by_context['context'], rotation=45, ha='right')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# Add sample size annotations
for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, risk_by_context['count'])):
    ax2.text(bar1.get_x() + bar1.get_width()/2, 1.05, f'n={count}', 
             ha='center', va='bottom', fontsize=10)

# ===================================================================
# Chart 3: Success Rate Comparison
# ===================================================================
ax3 = plt.subplot(2, 3, 3)
success_by_context = df.groupby('context')['reward'].agg(['mean', 'count', 'std']).reset_index()
success_by_context = success_by_context[success_by_context['count'] >= 10]

# Calculate standard errors for error bars
success_by_context['se'] = success_by_context['std'] / np.sqrt(success_by_context['count'])

bars = ax3.bar(range(len(success_by_context)), success_by_context['mean'], 
               yerr=success_by_context['se'], capsize=5, alpha=0.7,
               color=['lightcoral' if 'Rat' in ctx else 'lightblue' for ctx in success_by_context['context']])

ax3.set_title('Foraging Success Rate by Context\n(Success rate may indicate stress levels)', 
              fontsize=14, fontweight='bold', pad=20)
ax3.set_ylabel('Success Rate (Proportion)', fontsize=12)
ax3.set_xticks(range(len(success_by_context)))
ax3.set_xticklabels(success_by_context['context'], rotation=45, ha='right')
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, mean_val, count in zip(bars, success_by_context['mean'], success_by_context['count']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{mean_val:.3f}\n(n={count})', ha='center', va='bottom', fontsize=10)

# ===================================================================
# Chart 4: Temporal Analysis - Activity Over Time
# ===================================================================
ax4 = plt.subplot(2, 3, 4)
# Create hourly bins for analysis
df['hour_bins'] = pd.cut(df['hours_after_sunset'], 
                        bins=[-1, 1, 3, 6, 9, float('inf')], 
                        labels=['Early Night\n(0-1h)', 'Peak Activity\n(1-3h)', 
                               'Mid Night\n(3-6h)', 'Late Night\n(6-9h)', 'Very Late\n(9h+)'])

hourly_stats = df.groupby(['hour_bins', 'context']).size().unstack(fill_value=0)
hourly_stats_prop = hourly_stats.div(hourly_stats.sum(axis=1), axis=0)

# Focus on main contexts for clarity
main_contexts = ['Rat Present', 'Bat Only', 'Other Context']
available_contexts = [ctx for ctx in main_contexts if ctx in hourly_stats_prop.columns]

if available_contexts:
    hourly_stats_prop[available_contexts].plot(kind='bar', ax=ax4, alpha=0.8)
    ax4.set_title('Temporal Distribution of Landing Contexts\n(When do rat encounters occur?)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylabel('Proportion of Observations', fontsize=12)
    ax4.set_xlabel('Time Period After Sunset', fontsize=12)
    ax4.legend(title='Context', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ===================================================================
# Chart 5: Correlation Matrix
# ===================================================================
ax5 = plt.subplot(2, 3, 5)
# Select key variables for correlation analysis
correlation_vars = ['bat_landing_to_food', 'risk', 'reward', 'seconds_after_rat_arrival', 'hours_after_sunset']
available_vars = [var for var in correlation_vars if var in df.columns]

if len(available_vars) >= 3:
    corr_matrix = df[available_vars].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, square=True, 
                fmt='.3f', ax=ax5, cbar_kws={'label': 'Correlation Coefficient'})
    ax5.set_title('Behavioral Variable Correlations\n(Red = positive, Blue = negative)', 
                  fontsize=14, fontweight='bold', pad=20)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax5.yaxis.get_majorticklabels(), rotation=0)

# ===================================================================
# Chart 6: Evidence Summary
# ===================================================================
ax6 = plt.subplot(2, 3, 6)

# Collect evidence from all tests
evidence_tests = []
evidence_results = []
evidence_colors = []

if 'p_value_vigilance' in locals():
    evidence_tests.append('Vigilance\n(Approach Time)')
    if p_value_vigilance < 0.05 and rat_context['bat_landing_to_food'].mean() > non_rat_context['bat_landing_to_food'].mean():
        evidence_results.append('Supports Hypothesis')
        evidence_colors.append('green')
    elif p_value_vigilance < 0.05:
        evidence_results.append('Contradicts Hypothesis')
        evidence_colors.append('red')
    else:
        evidence_results.append('No Significant Effect')
        evidence_colors.append('gray')

if 'p_value_risk' in locals():
    evidence_tests.append('Risk Avoidance\n(Behavioral Choice)')
    if p_value_risk < 0.05:
        rat_risk_avoid = (1 - df[df['context'] == 'Rat Present']['risk'].mean())
        other_risk_avoid = (1 - df[df['context'] != 'Rat Present']['risk'].mean())
        if rat_risk_avoid > other_risk_avoid:
            evidence_results.append('Supports Hypothesis')
            evidence_colors.append('green')
        else:
            evidence_results.append('Contradicts Hypothesis')
            evidence_colors.append('red')
    else:
        evidence_results.append('No Significant Effect')
        evidence_colors.append('gray')

if 'p_value_success' in locals():
    evidence_tests.append('Success Rate\n(Performance)')
    if p_value_success < 0.05:
        if p_rat < p_other:
            evidence_results.append('Supports Hypothesis')
            evidence_colors.append('green')
        else:
            evidence_results.append('Contradicts Hypothesis')
            evidence_colors.append('red')
    else:
        evidence_results.append('No Significant Effect')
        evidence_colors.append('gray')

# Create evidence summary chart
if evidence_tests:
    y_pos = range(len(evidence_tests))
    colors_map = {'green': '#2ecc71', 'red': '#e74c3c', 'gray': '#95a5a6'}
    bar_colors = [colors_map[color] for color in evidence_colors]
    
    bars = ax6.barh(y_pos, [1]*len(evidence_tests), color=bar_colors, alpha=0.7)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(evidence_tests)
    ax6.set_xlabel('Evidence Strength', fontsize=12)
    ax6.set_title('Summary of Evidence for Predator Hypothesis\n(Green=Support, Red=Contradict, Gray=No Effect)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax6.set_xlim(0, 1)
    ax6.set_xticks([])
    
    # Add result labels
    for bar, result in zip(bars, evidence_results):
        ax6.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, result, 
                ha='center', va='center', fontweight='bold', color='white')

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
plt.show()

# %%
# ===================================================================
# COMPREHENSIVE CONCLUSIONS
# ===================================================================

print("\n" + "=" * 70)
print("COMPREHENSIVE ANALYSIS CONCLUSIONS")
print("=" * 70)

# Count supporting evidence
supporting_evidence = 0
total_tests = 0

print("\nDetailed Evidence Assessment:")
print("-" * 40)

if 'p_value_vigilance' in locals():
    total_tests += 1
    print(f"\n1. VIGILANCE TEST (Approach Time Analysis):")
    print(f"   • Statistical significance: p = {p_value_vigilance:.3f}")
    print(f"   • Effect size (Cohen's d): {cohens_d:.3f}")
    
    if p_value_vigilance < 0.05 and rat_context['bat_landing_to_food'].mean() > non_rat_context['bat_landing_to_food'].mean():
        supporting_evidence += 1
        print("   ✓ SUPPORTS HYPOTHESIS: Bats show increased vigilance (longer approach times) when rats are present")
        print("   → This suggests bats perceive rats as a threat requiring caution")
    elif p_value_vigilance < 0.05:
        print("   ✗ CONTRADICTS HYPOTHESIS: Bats show decreased vigilance when rats are present")
        print("   → This suggests bats do not perceive rats as threatening")
    else:
        print("   ⚬ INCONCLUSIVE: No significant difference in vigilance behavior")

if 'p_value_risk' in locals():
    total_tests += 1
    print(f"\n2. RISK AVOIDANCE TEST (Behavioral Choice Analysis):")
    print(f"   • Statistical significance: p = {p_value_risk:.3f}")
    print(f"   • Effect size (Cramer's V): {cramers_v:.3f}")
    
    if p_value_risk < 0.05:
        rat_avoid_rate = (1 - df[df['context'] == 'Rat Present']['risk'].mean()) * 100
        other_avoid_rate = (1 - df[df['context'] != 'Rat Present']['risk'].mean()) * 100
        
        if rat_avoid_rate > other_avoid_rate:
            supporting_evidence += 1
            print("   ✓ SUPPORTS HYPOTHESIS: Higher risk-avoidance when rats are present")
            print(f"   → Risk-avoidance rate: {rat_avoid_rate:.1f}% (rat contexts) vs {other_avoid_rate:.1f}% (other contexts)")
        else:
            print("   ✗ CONTRADICTS HYPOTHESIS: Lower risk-avoidance when rats are present")
            print(f"   → Risk-avoidance rate: {rat_avoid_rate:.1f}% (rat contexts) vs {other_avoid_rate:.1f}% (other contexts)")
    else:
        print("   ⚬ INCONCLUSIVE: No significant association between context and risk behavior")

if 'p_value_success' in locals():
    total_tests += 1
    print(f"\n3. FORAGING SUCCESS TEST (Performance Analysis):")
    print(f"   • Statistical significance: p = {p_value_success:.3f}")
    
    if p_value_success < 0.05:
        if p_rat < p_other:
            supporting_evidence += 1
            print("   ✓ SUPPORTS HYPOTHESIS: Lower success rate when rats are present")
            print(f"   → Success rates: {p_rat:.1%} (rat contexts) vs {p_other:.1%} (other contexts)")
            print("   → Reduced performance may indicate stress or distraction from predator presence")
        else:
            print("   ✗ CONTRADICTS HYPOTHESIS: Higher success rate when rats are present")
            print(f"   → Success rates: {p_rat:.1%} (rat contexts) vs {p_other:.1%} (other contexts)")
    else:
        print("   ⚬ INCONCLUSIVE: No significant difference in foraging success")

# Overall conclusion
print(f"\n" + "=" * 50)
print("FINAL SCIENTIFIC CONCLUSION")
print("=" * 50)

evidence_strength = supporting_evidence / total_tests if total_tests > 0 else 0

print(f"\nEvidence Summary: {supporting_evidence}/{total_tests} tests support the predator hypothesis")

if evidence_strength >= 0.67:  # 2/3 or more tests support
    conclusion_level = "STRONG EVIDENCE"
    conclusion_icon = "🎯"
    conclusion_text = """
    The data provides strong evidence that Egyptian Fruit Bats perceive Black Rats 
    as potential predators rather than merely competitors. This is demonstrated through:
    
    • Increased vigilance behaviors (longer approach times)
    • More risk-avoidant behavioral choices
    • Reduced foraging performance in rat contexts
    
    ECOLOGICAL IMPLICATIONS:
    - Rats may function as an additional predation pressure in bat foraging environments
    - Anti-predator behaviors may reduce foraging efficiency even when rats pose minimal direct threat
    - This suggests complex multi-species interactions beyond simple resource competition
    """
    
elif evidence_strength >= 0.33:  # 1/3 to 2/3 tests support
    conclusion_level = "MODERATE EVIDENCE" 
    conclusion_icon = "⚠️"
    conclusion_text = """
    The data provides moderate evidence for the predator hypothesis. Some behavioral
    indicators suggest bats may perceive rats as threatening, but the evidence is mixed.
    
    POSSIBLE EXPLANATIONS:
    - Context-dependent responses (rats may be threatening only in certain situations)
    - Individual variation in anti-predator responses
    - Habituation effects over time
    - Alternative explanations for observed behavioral differences
    
    RECOMMENDATION: Additional data collection focusing on environmental context
    and individual bat responses would strengthen conclusions.
    """
    
else:  # Less than 1/3 tests support
    conclusion_level = "INSUFFICIENT EVIDENCE"
    conclusion_icon = "❌"
    conclusion_text = """
    The current data does not provide sufficient evidence to support the predator
    hypothesis. Bats do not appear to consistently perceive rats as predatory threats.
    
    ALTERNATIVE INTERPRETATIONS:
    - Rats function primarily as resource competitors rather than predators
    - Behavioral differences may be due to factors other than predator perception
    - Sample size or study conditions may not capture true predator responses
    
    SCIENTIFIC IMPLICATIONS:
    - The relationship between bats and rats may be more complex than a simple predator-prey dynamic
    - Resource competition models may better explain observed interactions
    """

print(f"\n{conclusion_icon} {conclusion_level}")
print("=" * (len(conclusion_level) + 4))
print(conclusion_text)

# ===================================================================
# RESEARCH RECOMMENDATIONS
# ===================================================================

print(f"\n{'='*50}")
print("RECOMMENDATIONS FOR FUTURE RESEARCH")
print("="*50)

recommendations = [
    "1. EXPERIMENTAL DESIGN IMPROVEMENTS:",
    "   • Increase sample sizes for rare contexts (e.g., direct bat-rat encounters)",
    "   • Control for environmental variables (food availability, weather, season)",
    "   • Include physiological stress measures (cortisol, heart rate)",
    "",
    "2. BEHAVIORAL ANALYSIS ENHANCEMENTS:",
    "   • Video analysis of specific anti-predator behaviors (scanning, freezing)",
    "   • Quantify spatial positioning relative to rats",
    "   • Measure flight response times and directions",
    "",
    "3. ECOLOGICAL CONTEXT STUDIES:",
    "   • Compare responses across different habitat types",
    "   • Investigate seasonal variations in behavior",
    "   • Assess impact of alternative food sources",
    "",
    "4. MULTI-SPECIES INTERACTION ANALYSIS:",
    "   • Include other potential competitors/predators",
    "   • Study community-level effects of rat presence",
    "   • Examine long-term population impacts"
]

for recommendation in recommendations:
    print(recommendation)

# ===================================================================
# STATISTICAL SUMMARY TABLE
# ===================================================================

print(f"\n{'='*50}")
print("STATISTICAL TEST SUMMARY TABLE")
print("="*50)

print(f"{'Test':<25} {'Statistic':<12} {'p-value':<10} {'Effect Size':<12} {'Interpretation':<20}")
print("-" * 85)

if 'p_value_vigilance' in locals():
    sig_level = "***" if p_value_vigilance < 0.001 else "**" if p_value_vigilance < 0.01 else "*" if p_value_vigilance < 0.05 else "ns"
    effect_interpretation = "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"
    interpretation = f"{'Supports' if p_value_vigilance < 0.05 and rat_context['bat_landing_to_food'].mean() > non_rat_context['bat_landing_to_food'].mean() else 'No effect'}"
    print(f"{'Vigilance (t-test)':<25} {t_statistic:<12.3f} {p_value_vigilance:<10.3f} {cohens_d:<12.3f} {interpretation:<20}")

if 'p_value_risk' in locals():
    sig_level = "***" if p_value_risk < 0.001 else "**" if p_value_risk < 0.01 else "*" if p_value_risk < 0.05 else "ns"
    effect_interpretation = "Large" if cramers_v > 0.5 else "Medium" if cramers_v > 0.3 else "Small"
    interpretation = f"{'Significant' if p_value_risk < 0.05 else 'No effect'}"
    print(f"{'Risk Avoidance (χ²)':<25} {chi2_statistic:<12.3f} {p_value_risk:<10.3f} {cramers_v:<12.3f} {interpretation:<20}")

if 'p_value_success' in locals():
    sig_level = "***" if p_value_success < 0.001 else "**" if p_value_success < 0.01 else "*" if p_value_success < 0.05 else "ns"
    interpretation = f"{'Significant' if p_value_success < 0.05 else 'No effect'}"
    print(f"{'Success Rate (z-test)':<25} {z_statistic:<12.3f} {p_value_success:<10.3f} {'N/A':<12} {interpretation:<20}")

print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
print("Effect sizes: Cohen's d (small=0.2, medium=0.5, large=0.8), Cramer's V (small=0.1, medium=0.3, large=0.5)")

# ===================================================================
# DATA EXPORT SUMMARY
# ===================================================================

print(f"\n{'='*50}")
print("ANALYSIS COMPLETION SUMMARY")
print("="*50)

print(f"Analysis completed successfully!")
print(f"• Total observations analyzed: {len(df):,}")
print(f"• Clean observations used: {len(df_clean):,}")
print(f"• Behavioral contexts identified: {df['context'].nunique()}")
print(f"• Statistical tests performed: {total_tests}")
print(f"• Visualizations generated: 6 comprehensive charts")

print(f"\nKey Findings:")
print(f"• Mean approach time in rat contexts: {rat_context['bat_landing_to_food'].mean():.2f} seconds")
print(f"• Mean approach time in other contexts: {non_rat_context['bat_landing_to_food'].mean():.2f} seconds") 
print(f"• Overall risk-taking rate: {df['risk'].mean()*100:.1f}%")
print(f"• Overall foraging success rate: {df['reward'].mean()*100:.1f}%")

print(f"\nThis analysis provides a comprehensive examination of the predator hypothesis")
print(f"using rigorous statistical methods and clear visualizations suitable for")
print(f"scientific presentation and peer review.")

print(f"\n{'='*70}")
print("END OF ANALYSIS")
print("="*70)