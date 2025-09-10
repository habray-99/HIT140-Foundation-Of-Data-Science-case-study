# %%
"""
HIT140 Foundations of Data Science - Bat vs. Rat Foraging Analysis
Investigation A: Do bats perceive rats as predators?

This analysis examines whether Egyptian Fruit Bats exhibit predator avoidance behaviors
when Black Rats are present at feeding platforms, specifically looking for evidence of:
1. Increased vigilance (longer approach times)
2. Risk-avoidance behaviors
3. Reduced foraging success
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set visualization style for professional appearance
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %%
# =======================
# DATA LOADING & PREPROCESSING
# =======================

print("Loading and preprocessing dataset...")

# Load the dataset
df = pd.read_csv(r'/mnt/e/Foundation of Data Science/ass 1/dataset1.csv', engine='pyarrow')

# Convert datetime columns to proper datetime format
datetime_columns = ['rat_period_start', 'rat_period_end', 'sunset_time', 'start_time']
for col in datetime_columns:
    df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M')

# Handle missing values in the habit column using forward fill
df['habit'] = df['habit'].fillna(method='ffill')

print(f"Dataset loaded successfully with {len(df)} observations")
print(f"Date range: {df['start_time'].min().date()} to {df['start_time'].max().date()}")

# %%
# =======================
# DATA CLEANING & FEATURE ENGINEERING
# =======================

def categorize_landing_context(habit):
    """
    Clean and categorize the landing context for better analysis.
    
    This function simplifies the complex habit descriptions into meaningful categories
    that help us understand the behavioral context of each bat landing.
    """
    if pd.isna(habit):
        return 'unknown'
    
    habit_lower = str(habit).lower()
    
    # Identify key behavioral contexts
    if 'rat' in habit_lower and 'bat' not in habit_lower:
        return 'rat_present'
    elif 'bat' in habit_lower and 'rat' not in habit_lower:
        return 'bat_only'
    elif 'pick' in habit_lower:
        return 'picking_behavior'
    elif 'fast' in habit_lower:
        return 'fast_approach'
    elif any(char.isdigit() for char in habit_lower):  # Remove coordinate data
        return 'unknown'
    else:
        return 'other'

# Apply the cleaning function
df['context_category'] = df['habit'].apply(categorize_landing_context)

# Create binary indicators for easier analysis
df['rats_present'] = (df['context_category'] == 'rat_present').astype(int)

print("\n=== LANDING CONTEXT DISTRIBUTION ===")
context_counts = df['context_category'].value_counts()
for context, count in context_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{context}: {count} ({percentage:.1f}%)")

# %%
# =======================
# DESCRIPTIVE STATISTICS
# =======================

print("\n" + "="*60)
print("DESCRIPTIVE ANALYSIS: BAT FORAGING BEHAVIOR")
print("="*60)

# Overall behavioral metrics
print("\n🦇 OVERALL FORAGING BEHAVIOR SUMMARY:")
print("-" * 40)
print(f"Total bat landings observed: {len(df):,}")
print(f"Average approach time: {df['bat_landing_to_food'].mean():.2f} seconds")
print(f"Risk-taking incidents: {df['risk'].sum()} ({df['risk'].mean()*100:.1f}%)")
print(f"Successful foraging attempts: {df['reward'].sum()} ({df['reward'].mean()*100:.1f}%)")

# Context-specific analysis
print(f"\n🐀 RAT PRESENCE ANALYSIS:")
print("-" * 40)
rat_present_data = df[df['rats_present'] == 1]
no_rat_data = df[df['rats_present'] == 0]

print(f"Landings with rats present: {len(rat_present_data)} ({len(rat_present_data)/len(df)*100:.1f}%)")
print(f"Landings without rats: {len(no_rat_data)} ({len(no_rat_data)/len(df)*100:.1f}%)")

if len(rat_present_data) > 0:
    print(f"\nWhen rats are present:")
    print(f"  • Average approach time: {rat_present_data['bat_landing_to_food'].mean():.2f}s")
    print(f"  • Risk-taking rate: {rat_present_data['risk'].mean()*100:.1f}%")
    print(f"  • Success rate: {rat_present_data['reward'].mean()*100:.1f}%")

print(f"\nWhen rats are absent:")
print(f"  • Average approach time: {no_rat_data['bat_landing_to_food'].mean():.2f}s")
print(f"  • Risk-taking rate: {no_rat_data['risk'].mean()*100:.1f}%")
print(f"  • Success rate: {no_rat_data['reward'].mean()*100:.1f}%")

# %%
# =======================
# VISUALIZATION 1: APPROACH TIME ANALYSIS
# =======================

plt.figure(figsize=(12, 8))

# Remove extreme outliers for better visualization (keep 95% of data)
df_viz = df[df['bat_landing_to_food'] < df['bat_landing_to_food'].quantile(0.95)]

plt.subplot(2, 2, 1)
sns.boxplot(data=df_viz, x='context_category', y='bat_landing_to_food', palette='Set2')
plt.title('Vigilance Behavior: Time to Approach Food by Context', fontsize=12, fontweight='bold')
plt.xlabel('Landing Context')
plt.ylabel('Seconds to Approach Food')
plt.xticks(rotation=45)

# Add statistical annotation
rat_median = df_viz[df_viz['context_category'] == 'rat_present']['bat_landing_to_food'].median()
other_median = df_viz[df_viz['context_category'] != 'rat_present']['bat_landing_to_food'].median()
plt.text(0.02, 0.98, f'Rat context median: {rat_median:.1f}s', transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.subplot(2, 2, 2)
# Histogram comparison
plt.hist(df_viz[df_viz['rats_present'] == 1]['bat_landing_to_food'], 
         alpha=0.7, label='Rats Present', bins=20, color='coral')
plt.hist(df_viz[df_viz['rats_present'] == 0]['bat_landing_to_food'], 
         alpha=0.7, label='No Rats', bins=20, color='lightblue')
plt.xlabel('Seconds to Approach Food')
plt.ylabel('Frequency')
plt.title('Distribution of Approach Times', fontweight='bold')
plt.legend()

plt.subplot(2, 2, 3)
# Risk behavior by context
risk_by_context = df.groupby('context_category')['risk'].mean().sort_values(ascending=True)
bars = plt.barh(range(len(risk_by_context)), risk_by_context.values, color='lightgreen', alpha=0.8)
plt.yticks(range(len(risk_by_context)), risk_by_context.index)
plt.xlabel('Risk-Taking Rate (Proportion)')
plt.title('Risk-Taking Behavior by Context', fontweight='bold')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, risk_by_context.values)):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{value:.3f}', va='center', fontsize=10)

plt.subplot(2, 2, 4)
# Success rate comparison
success_comparison = df.groupby('rats_present')['reward'].mean()
bars = plt.bar(['Rats Absent', 'Rats Present'], success_comparison.values, 
               color=['lightblue', 'coral'], alpha=0.8)
plt.ylabel('Success Rate (Proportion)')
plt.title('Foraging Success: Rats Present vs Absent', fontweight='bold')
plt.ylim(0, 1)

# Add value labels
for bar, value in zip(bars, success_comparison.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.suptitle('Predator Avoidance Analysis: Bat Behavioral Responses to Rat Presence', 
             fontsize=14, fontweight='bold', y=1.02)
plt.show()

# %%
# =======================
# VISUALIZATION 2: BEHAVIORAL PATTERNS
# =======================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Chart 1: Risk vs Reward Analysis
risk_reward_crosstab = pd.crosstab(df['risk'], df['reward'], normalize='index')
risk_reward_crosstab.plot(kind='bar', ax=axes[0,0], color=['lightcoral', 'lightgreen'], alpha=0.8)
axes[0,0].set_title('Success Rate by Risk-Taking Behavior', fontweight='bold')
axes[0,0].set_xlabel('Risk Behavior (0=Avoidance, 1=Taking)')
axes[0,0].set_ylabel('Proportion of Outcomes')
axes[0,0].legend(['Failure', 'Success'])
axes[0,0].set_xticklabels(['Risk Avoidance', 'Risk Taking'], rotation=0)

# Chart 2: Temporal patterns
df['hour_after_sunset_rounded'] = df['hours_after_sunset'].round()
temporal_risk = df.groupby('hour_after_sunset_rounded')['risk'].mean()
axes[0,1].plot(temporal_risk.index, temporal_risk.values, marker='o', linewidth=2, markersize=6)
axes[0,1].set_title('Risk-Taking Behavior Throughout the Night', fontweight='bold')
axes[0,1].set_xlabel('Hours After Sunset')
axes[0,1].set_ylabel('Risk-Taking Rate')
axes[0,1].grid(True, alpha=0.3)

# Chart 3: Context distribution pie chart
context_counts = df['context_category'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(context_counts)))
axes[1,0].pie(context_counts.values, labels=context_counts.index, autopct='%1.1f%%', 
              colors=colors, startangle=90)
axes[1,0].set_title('Distribution of Landing Contexts', fontweight='bold')

# Chart 4: Correlation heatmap of key variables
correlation_vars = ['bat_landing_to_food', 'risk', 'reward', 'rats_present', 'hours_after_sunset']
correlation_matrix = df[correlation_vars].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', center=0, ax=axes[1,1])
axes[1,1].set_title('Correlation Matrix of Key Variables', fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# =======================
# STATISTICAL HYPOTHESIS TESTING
# =======================

print("\n" + "="*70)
print("STATISTICAL ANALYSIS: TESTING THE PREDATOR HYPOTHESIS")
print("="*70)

print("\n📊 Research Question: Do bats perceive rats as potential predators?")
print("Expected behaviors if rats are perceived as predators:")
print("  1. Increased vigilance (longer approach times)")
print("  2. More risk-avoidance behavior")
print("  3. Potentially reduced foraging success")

# Prepare data for analysis
rat_context_data = df[df['rats_present'] == 1]
no_rat_context_data = df[df['rats_present'] == 0]

print(f"\nSample sizes:")
print(f"  • Rat present: {len(rat_context_data)} observations")
print(f"  • No rats: {len(no_rat_context_data)} observations")

# =======================
# TEST 1: VIGILANCE HYPOTHESIS
# =======================
print(f"\n{'='*50}")
print("TEST 1: VIGILANCE BEHAVIOR (Approach Time)")
print('='*50)
print("H₀: No difference in approach time between rat and no-rat contexts")
print("H₁: Bats take longer to approach food when rats are present")

# Remove extreme outliers for robust testing
approach_rat = rat_context_data['bat_landing_to_food'].dropna()
approach_no_rat = no_rat_context_data['bat_landing_to_food'].dropna()

# Filter out extreme outliers (beyond 95th percentile)
approach_rat = approach_rat[approach_rat < approach_rat.quantile(0.95)]
approach_no_rat = approach_no_rat[approach_no_rat < approach_no_rat.quantile(0.95)]

if len(approach_rat) > 0 and len(approach_no_rat) > 0:
    # Perform independent t-test
    t_statistic, p_value_t = stats.ttest_ind(approach_rat, approach_no_rat)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(approach_rat) - 1) * approach_rat.var() + 
                         (len(approach_no_rat) - 1) * approach_no_rat.var()) / 
                        (len(approach_rat) + len(approach_no_rat) - 2))
    cohens_d = (approach_rat.mean() - approach_no_rat.mean()) / pooled_std
    
    print(f"\nDescriptive Statistics:")
    print(f"  Rat present: μ = {approach_rat.mean():.2f}s, σ = {approach_rat.std():.2f}s, n = {len(approach_rat)}")
    print(f"  No rats:     μ = {approach_no_rat.mean():.2f}s, σ = {approach_no_rat.std():.2f}s, n = {len(approach_no_rat)}")
    print(f"  Mean difference: {approach_rat.mean() - approach_no_rat.mean():.2f}s")
    
    print(f"\nStatistical Test Results:")
    print(f"  t-statistic: {t_statistic:.3f}")
    print(f"  p-value: {p_value_t:.3f}")
    print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
    
    # Interpret results
    if p_value_t < 0.05:
        significance = "SIGNIFICANT ✓"
        interpretation = "Bats show significantly different approach times when rats are present"
    else:
        significance = "NOT SIGNIFICANT ✗"
        interpretation = "No significant difference in approach times"
        
    print(f"  Result: {significance}")
    print(f"  Interpretation: {interpretation}")
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size_interp = "small"
    elif abs(cohens_d) < 0.8:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    print(f"  Effect size: {effect_size_interp}")

# =======================
# TEST 2: RISK-AVOIDANCE HYPOTHESIS
# =======================
print(f"\n{'='*50}")
print("TEST 2: RISK-AVOIDANCE BEHAVIOR")
print('='*50)
print("H₀: Risk behavior is independent of rat presence")
print("H₁: Bats show more risk-avoidance when rats are present")

# Create contingency table
contingency_table = pd.crosstab(df['rats_present'], df['risk'], margins=True)
print(f"\nContingency Table:")
print("                 Risk-Avoid  Risk-Take   Total")
print(f"No Rats             {contingency_table.iloc[0,0]:8d}   {contingency_table.iloc[0,1]:8d}   {contingency_table.iloc[0,2]:8d}")
print(f"Rats Present        {contingency_table.iloc[1,0]:8d}   {contingency_table.iloc[1,1]:8d}   {contingency_table.iloc[1,2]:8d}")
print(f"Total               {contingency_table.iloc[2,0]:8d}   {contingency_table.iloc[2,1]:8d}   {contingency_table.iloc[2,2]:8d}")

# Perform Chi-square test
chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table.iloc[:2, :2])

# Calculate effect size (Cramer's V)
n_total = contingency_table.iloc[2,2]
cramers_v = np.sqrt(chi2_stat / (n_total * (min(contingency_table.shape[:2]) - 1)))

print(f"\nRisk-taking rates:")
risk_rate_no_rats = df[df['rats_present'] == 0]['risk'].mean()
risk_rate_rats = df[df['rats_present'] == 1]['risk'].mean()
print(f"  No rats: {risk_rate_no_rats:.3f} ({risk_rate_no_rats*100:.1f}%)")
print(f"  Rats present: {risk_rate_rats:.3f} ({risk_rate_rats*100:.1f}%)")

print(f"\nStatistical Test Results:")
print(f"  χ² statistic: {chi2_stat:.3f}")
print(f"  p-value: {p_value_chi2:.3f}")
print(f"  Degrees of freedom: {dof}")
print(f"  Effect size (Cramer's V): {cramers_v:.3f}")

if p_value_chi2 < 0.05:
    significance = "SIGNIFICANT ✓"
    interpretation = "Risk behavior is significantly associated with rat presence"
else:
    significance = "NOT SIGNIFICANT ✗"
    interpretation = "No significant association between rat presence and risk behavior"

print(f"  Result: {significance}")
print(f"  Interpretation: {interpretation}")

# =======================
# TEST 3: FORAGING SUCCESS HYPOTHESIS
# =======================
print(f"\n{'='*50}")
print("TEST 3: FORAGING SUCCESS")
print('='*50)
print("H₀: Success rate is the same regardless of rat presence")
print("H₁: Success rate differs when rats are present")

# Two-proportion z-test
success_no_rats = df[df['rats_present'] == 0]['reward']
success_rats = df[df['rats_present'] == 1]['reward']

count_success_no_rats = success_no_rats.sum()
count_success_rats = success_rats.sum()
n_no_rats = len(success_no_rats)
n_rats = len(success_rats)

p1 = count_success_no_rats / n_no_rats
p2 = count_success_rats / n_rats
p_pooled = (count_success_no_rats + count_success_rats) / (n_no_rats + n_rats)

# Calculate z-statistic
z_statistic = (p1 - p2) / np.sqrt(p_pooled * (1 - p_pooled) * (1/n_no_rats + 1/n_rats))
p_value_z = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

print(f"\nSuccess Rates:")
print(f"  No rats: {p1:.3f} ({count_success_no_rats}/{n_no_rats})")
print(f"  Rats present: {p2:.3f} ({count_success_rats}/{n_rats})")
print(f"  Difference: {p1 - p2:.3f}")

print(f"\nStatistical Test Results:")
print(f"  z-statistic: {z_statistic:.3f}")
print(f"  p-value: {p_value_z:.3f}")

if p_value_z < 0.05:
    significance = "SIGNIFICANT ✓"
    interpretation = "Success rates differ significantly between contexts"
else:
    significance = "NOT SIGNIFICANT ✗"
    interpretation = "No significant difference in success rates"

print(f"  Result: {significance}")
print(f"  Interpretation: {interpretation}")

# =======================
# OVERALL CONCLUSION
# =======================
print(f"\n{'='*70}")
print("FINAL CONCLUSION: EVIDENCE FOR PREDATOR HYPOTHESIS")
print('='*70)

# Count significant results
significant_tests = 0
total_tests = 3

evidence_summary = []

if 'p_value_t' in locals() and p_value_t < 0.05:
    evidence_summary.append("✓ VIGILANCE: Significant difference in approach times")
    significant_tests += 1
else:
    evidence_summary.append("✗ VIGILANCE: No significant difference in approach times")

if p_value_chi2 < 0.05:
    evidence_summary.append("✓ RISK-AVOIDANCE: Significant association with rat presence")
    significant_tests += 1
else:
    evidence_summary.append("✗ RISK-AVOIDANCE: No significant association with rat presence")

if p_value_z < 0.05:
    evidence_summary.append("✓ SUCCESS RATE: Significant difference between contexts")
    significant_tests += 1
else:
    evidence_summary.append("✗ SUCCESS RATE: No significant difference between contexts")

print(f"\nEvidence Summary ({significant_tests}/{total_tests} tests significant):")
for evidence in evidence_summary:
    print(f"  {evidence}")

# Overall interpretation
if significant_tests >= 2:
    overall_conclusion = "🎯 STRONG EVIDENCE: Bats likely perceive rats as predators"
    recommendation = "The data strongly supports the predator hypothesis. Multiple behavioral indicators suggest bats modify their foraging behavior in response to rat presence."
elif significant_tests == 1:
    overall_conclusion = "⚠️  MODERATE EVIDENCE: Some support for predator hypothesis"
    recommendation = "There is some evidence supporting the predator hypothesis, but additional data or analysis may be needed for stronger conclusions."
else:
    overall_conclusion = "❌ LIMITED EVIDENCE: Weak support for predator hypothesis"
    recommendation = "The current data does not provide strong evidence that bats perceive rats as predators. Other explanations for behavioral differences should be considered."

print(f"\n{overall_conclusion}")
print(f"\nRecommendation: {recommendation}")

print(f"\n📝 Note: All statistical tests used α = 0.05 significance level.")
print(f"Analysis completed on {len(df)} total observations spanning {df['start_time'].dt.date.nunique()} days.")

# %%
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("This analysis provides evidence for Investigation A of the Bat vs. Rat study.")
print("For Investigation B (seasonal analysis), additional temporal analysis would be required.")
print("Consider examining monthly/seasonal patterns in the observed behaviors.")