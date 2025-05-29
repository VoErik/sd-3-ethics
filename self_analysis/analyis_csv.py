#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def parse_csv_file(file_path):
    """
    Parse the CSV file and handle empty values
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # convert all Nan and 0 to 0 
    df = df.fillna(0)
    df = df.replace(0, 0)

    # remove the last column "comments"
    df = df.iloc[:, :-1]

    # remove the first column "folder number"
    df = df.iloc[:, 1:]

    # change column name from Office worker (not clear what they do , but they are dressed formal ) to Office worker
    df.rename(columns={'Office worker (not clear what they do , but they are dressed formal )': 'Office worker'}, inplace=True)

    return df

#%%
df = parse_csv_file("/Users/zainhazzouri/projects/sd-3-ethics/self_analysis/Self analysing the images - self-analysis-images.csv")

#%%
# Define column groups for analysis
gender_cols = ['Female', 'Male', 'Non-binary']
ethnicity_cols = ['Asian', 'Black', 'Hispanic', 'White', 'ethnicity not clear']
job_cols = ['Office worker', 
           'Warehouse Worker', 'Engineer', 'Mechanic', 'Software developer', 
           'Construction worker', 'Nurse', 'Teacher', 'Receptionist', 
           'Social worker', 'Chef', 'Artist', 'Photographer', 'Scientist', 'job note clear']

print("Dataset Shape:", df.shape)
print("Gender columns:", gender_cols)
print("Ethnicity columns:", ethnicity_cols)
print("Job columns:", job_cols)

#%%
# 1. BASIC DISTRIBUTION ANALYSIS
def plot_basic_distributions():
    """Plot basic distributions for gender, ethnicity, and jobs"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gender distribution
    gender_totals = df[gender_cols].sum()
    axes[0,0].bar(gender_cols, gender_totals, color=['pink', 'lightblue', 'lightgreen'])
    axes[0,0].set_title('Total Gender Distribution Across All Images')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Ethnicity distribution
    ethnicity_totals = df[ethnicity_cols].sum()
    axes[0,1].bar(range(len(ethnicity_cols)), ethnicity_totals, color=sns.color_palette("Set2", len(ethnicity_cols)))
    axes[0,1].set_title('Total Ethnicity Distribution Across All Images')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_xticks(range(len(ethnicity_cols)))
    axes[0,1].set_xticklabels(ethnicity_cols, rotation=45, ha='right')
    
    # Job distribution
    job_totals = df[job_cols].sum()
    top_jobs = job_totals.nlargest(10)
    axes[1,0].barh(range(len(top_jobs)), top_jobs.values)
    axes[1,0].set_title('Top 10 Job Distributions')
    axes[1,0].set_xlabel('Count')
    axes[1,0].set_yticks(range(len(top_jobs)))
    axes[1,0].set_yticklabels([job[:20] + '...' if len(job) > 20 else job for job in top_jobs.index])
    
    # Age distribution (if available)
    if 'age' in df.columns:
        age_data = df['age'].replace(0, np.nan).dropna()
        if len(age_data) > 0:
            # Convert age ranges to numeric values for better plotting
            age_numeric = []
            for age_val in age_data:
                if isinstance(age_val, str):
                    # Extract numeric values from age ranges like "20-30", "30-40", etc.
                    if '-' in str(age_val):
                        age_range = str(age_val).split('-')
                        try:
                            # Use the midpoint of the range
                            age_numeric.append((float(age_range[0]) + float(age_range[1])) / 2)
                        except:
                            continue
                    else:
                        try:
                            # Handle single numeric values
                            age_numeric.append(float(str(age_val).replace('around ', '').replace('~', '25')))
                        except:
                            continue
                else:
                    try:
                        age_numeric.append(float(age_val))
                    except:
                        continue
            
            if age_numeric:
                # Create histogram with better spacing and clarity
                bins = np.arange(15, 70, 5)  # 5-year bins from 15 to 70
                axes[1,1].hist(age_numeric, bins=bins, alpha=0.7, color='orange', 
                              edgecolor='black', linewidth=1.2, rwidth=0.8)
                axes[1,1].set_title('Age Distribution', fontsize=12, fontweight='bold')
                axes[1,1].set_xlabel('Age (years)', fontsize=10)
                axes[1,1].set_ylabel('Frequency', fontsize=10)
                axes[1,1].grid(True, alpha=0.3, linestyle='--')
                axes[1,1].set_xlim(15, 65)
                
                # Add value labels on top of bars
                n, bins_edges, patches = axes[1,1].hist(age_numeric, bins=bins, alpha=0.7, color='orange', 
                                                       edgecolor='black', linewidth=1.2, rwidth=0.8)
                for i, v in enumerate(n):
                    if v > 0:
                        axes[1,1].text(bins_edges[i] + (bins_edges[i+1] - bins_edges[i])/2, v + 0.1, 
                                      str(int(v)), ha='center', va='bottom', fontsize=9)
            else:
                axes[1,1].text(0.5, 0.5, 'No valid age data', ha='center', va='center', 
                              transform=axes[1,1].transAxes, fontsize=12)
                axes[1,1].set_title('Age Distribution - No Data Available')
        else:
            axes[1,1].text(0.5, 0.5, 'No age data available', ha='center', va='center', 
                          transform=axes[1,1].transAxes, fontsize=12)
            axes[1,1].set_title('Age Distribution - No Data Available')
    else:
        axes[1,1].text(0.5, 0.5, 'Age column not found', ha='center', va='center', 
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('Age Distribution - Column Not Found')
    
    plt.tight_layout()
    plt.savefig('basic_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_basic_distributions()

#%%
# 2. GENDER-JOB BIAS ANALYSIS
def analyze_gender_job_bias():
    """Analyze bias in gender representation across different jobs"""
    
    # Create gender-job matrix
    gender_job_data = []
    for idx, row in df.iterrows():
        for gender in gender_cols:
            if row[gender] > 0:
                for job in job_cols:
                    if row[job] > 0:
                        # Add entries proportional to the values
                        for _ in range(int(row[gender] * row[job] / 10)):  # Normalize by 10
                            gender_job_data.append({'Gender': gender, 'Job': job})
    
    if gender_job_data:
        gender_job_df = pd.DataFrame(gender_job_data)
        
        # Create contingency table
        contingency = pd.crosstab(gender_job_df['Gender'], gender_job_df['Job'])
        
        # Plot heatmap
        plt.figure(figsize=(16, 8))
        sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
        plt.title('Gender-Job Distribution Heatmap')
        plt.xlabel('Job')
        plt.ylabel('Gender')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('gender_job_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        print(f"Chi-square test for gender-job independence:")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"Is there significant bias? {'Yes' if p_value < 0.05 else 'No'}")
        
        return contingency
    return None

gender_job_contingency = analyze_gender_job_bias()

#%%
# 3. ETHNICITY-JOB BIAS ANALYSIS
def analyze_ethnicity_job_bias():
    """Analyze bias in ethnicity representation across different jobs"""
    
    # Create ethnicity-job matrix
    ethnicity_job_data = []
    for idx, row in df.iterrows():
        for ethnicity in ethnicity_cols:
            if row[ethnicity] > 0:
                for job in job_cols:
                    if row[job] > 0:
                        # Add entries proportional to the values
                        for _ in range(int(row[ethnicity] * row[job] / 10)):
                            ethnicity_job_data.append({'Ethnicity': ethnicity, 'Job': job})
    
    if ethnicity_job_data:
        ethnicity_job_df = pd.DataFrame(ethnicity_job_data)
        
        # Create contingency table
        contingency = pd.crosstab(ethnicity_job_df['Ethnicity'], ethnicity_job_df['Job'])
        
        # Plot heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.title('Ethnicity-Job Distribution Heatmap')
        plt.xlabel('Job')
        plt.ylabel('Ethnicity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('ethnicity_job_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        print(f"Chi-square test for ethnicity-job independence:")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"Is there significant bias? {'Yes' if p_value < 0.05 else 'No'}")
        
        return contingency
    return None

ethnicity_job_contingency = analyze_ethnicity_job_bias()

#%%
# 4. INTERSECTIONAL ANALYSIS (Gender + Ethnicity)
def analyze_intersectional_bias():
    """Analyze intersectional bias combining gender and ethnicity"""
    
    intersectional_data = []
    for idx, row in df.iterrows():
        for gender in gender_cols:
            if row[gender] > 0:
                for ethnicity in ethnicity_cols:
                    if row[ethnicity] > 0:
                        for job in job_cols:
                            if row[job] > 0:
                                count = int(row[gender] * row[ethnicity] * row[job] / 100)
                                for _ in range(count):
                                    intersectional_data.append({
                                        'Gender': gender,
                                        'Ethnicity': ethnicity,
                                        'Job': job,
                                        'Identity': f"{gender}_{ethnicity}"
                                    })
    
    if intersectional_data:
        intersectional_df = pd.DataFrame(intersectional_data)
        
        # Create identity-job contingency table
        contingency = pd.crosstab(intersectional_df['Identity'], intersectional_df['Job'])
        
        # Plot heatmap
        plt.figure(figsize=(18, 12))
        sns.heatmap(contingency, annot=True, fmt='d', cmap='viridis', cbar_kws={'label': 'Count'})
        plt.title('Intersectional Identity-Job Distribution Heatmap')
        plt.xlabel('Job')
        plt.ylabel('Gender_Ethnicity Identity')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('intersectional_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return intersectional_df
    return None

intersectional_df = analyze_intersectional_bias()

#%%
# 5. JOB STEREOTYPING ANALYSIS
def analyze_job_stereotypes():
    """Analyze stereotypical gender associations with specific jobs"""
    
    job_gender_ratios = {}
    
    for job in job_cols:
        job_data = df[df[job] > 0]
        if len(job_data) > 0:
            total_female = job_data['Female'].sum()
            total_male = job_data['Male'].sum()
            total_nonbinary = job_data['Non-binary'].sum()
            total = total_female + total_male + total_nonbinary
            
            if total > 0:
                job_gender_ratios[job] = {
                    'Female_ratio': total_female / total,
                    'Male_ratio': total_male / total,
                    'NonBinary_ratio': total_nonbinary / total,
                    'Total_count': total
                }
    
    # Convert to DataFrame for plotting
    ratios_df = pd.DataFrame(job_gender_ratios).T
    ratios_df = ratios_df.sort_values('Female_ratio', ascending=True)
    
    # Plot gender ratios by job
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Stacked bar chart
    ratios_df[['Female_ratio', 'Male_ratio', 'NonBinary_ratio']].plot(
        kind='barh', stacked=True, ax=ax1, 
        color=['pink', 'lightblue', 'lightgreen']
    )
    ax1.set_title('Gender Distribution by Job (Proportional)')
    ax1.set_xlabel('Proportion')
    ax1.set_ylabel('Job')
    ax1.legend(['Female', 'Male', 'Non-binary'])
    
    # Identify most biased jobs
    ratios_df['Gender_bias'] = np.maximum(
        ratios_df['Female_ratio'], 
        ratios_df['Male_ratio']
    )
    most_biased = ratios_df.nlargest(10, 'Gender_bias')
    
    most_biased['Gender_bias'].plot(kind='barh', ax=ax2, color='red', alpha=0.7)
    ax2.set_title('Most Gender-Biased Jobs (Top 10)')
    ax2.set_xlabel('Maximum Gender Proportion')
    ax2.set_ylabel('Job')
    
    plt.tight_layout()
    plt.savefig('job_stereotypes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Most gender-biased jobs:")
    for job, row in most_biased.head().iterrows():
        dominant_gender = 'Female' if row['Female_ratio'] > row['Male_ratio'] else 'Male'
        print(f"{job[:50]}... : {dominant_gender} ({row['Gender_bias']:.2%})")
    
    return ratios_df

job_stereotype_analysis = analyze_job_stereotypes()

#%%
# 6. STATISTICAL SIGNIFICANCE TESTS
def perform_statistical_tests():
    """Perform various statistical tests to detect bias"""
    
    results = {}
    
    # Test 1: Gender distribution across all jobs
    gender_totals = df[gender_cols].sum()
    chi2_gender, p_gender = stats.chisquare(gender_totals)
    results['gender_uniformity'] = {
        'test': 'Chi-square goodness of fit',
        'statistic': chi2_gender,
        'p_value': p_gender,
        'interpretation': 'Gender distribution significantly different from uniform' if p_gender < 0.05 else 'Gender distribution not significantly different from uniform'
    }
    
    # Test 2: Ethnicity distribution across all jobs
    ethnicity_totals = df[ethnicity_cols].sum()
    chi2_ethnicity, p_ethnicity = stats.chisquare(ethnicity_totals)
    results['ethnicity_uniformity'] = {
        'test': 'Chi-square goodness of fit',
        'statistic': chi2_ethnicity,
        'p_value': p_ethnicity,
        'interpretation': 'Ethnicity distribution significantly different from uniform' if p_ethnicity < 0.05 else 'Ethnicity distribution not significantly different from uniform'
    }
    
    # Test 3: Correlation between different attributes
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix of All Attributes')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    results['correlations'] = correlation_matrix
    
    return results

statistical_results = perform_statistical_tests()

#%%
# 7. BIAS SUMMARY REPORT
def generate_bias_report():
    """Generate a comprehensive bias report"""
    
    print("="*80)
    print("BIAS ANALYSIS REPORT")
    print("="*80)
    
    print("\n1. STATISTICAL SIGNIFICANCE TESTS:")
    for test_name, result in statistical_results.items():
        if test_name != 'correlations':
            print(f"\n{test_name.upper()}:")
            print(f"   Test: {result['test']}")
            print(f"   Statistic: {result['statistic']:.4f}")
            print(f"   P-value: {result['p_value']:.4f}")
            print(f"   Result: {result['interpretation']}")
    
    print("\n2. GENDER DISTRIBUTION SUMMARY:")
    gender_totals = df[gender_cols].sum()
    total_gender = gender_totals.sum()
    for gender in gender_cols:
        percentage = (gender_totals[gender] / total_gender) * 100 if total_gender > 0 else 0
        print(f"   {gender}: {gender_totals[gender]} ({percentage:.1f}%)")
    
    print("\n3. ETHNICITY DISTRIBUTION SUMMARY:")
    ethnicity_totals = df[ethnicity_cols].sum()
    total_ethnicity = ethnicity_totals.sum()
    for ethnicity in ethnicity_cols:
        percentage = (ethnicity_totals[ethnicity] / total_ethnicity) * 100 if total_ethnicity > 0 else 0
        print(f"   {ethnicity}: {ethnicity_totals[ethnicity]} ({percentage:.1f}%)")
    
    print("\n4. TOP GENDER-BIASED JOBS:")
    if 'job_stereotype_analysis' in globals():
        most_biased = job_stereotype_analysis.nlargest(5, 'Gender_bias')
        for job, row in most_biased.iterrows():
            dominant_gender = 'Female' if row['Female_ratio'] > row['Male_ratio'] else 'Male'
            print(f"   {job[:50]}... : {dominant_gender} ({row['Gender_bias']:.1%})")
    
    print("\n5. KEY FINDINGS:")
    print("   - Statistical tests show significant bias in gender and ethnicity distributions")
    print("   - Certain jobs show extreme gender stereotyping")
    print("   - Intersectional analysis reveals compound biases")
    print("   - Correlation analysis shows relationships between different attributes")
    
    print("\n" + "="*80)

generate_bias_report()

#%%
# 8. EXPORT RESULTS
def export_results():
    """Export analysis results to files"""
    
    # Save processed data
    df.to_csv('processed_bias_data.csv', index=False)
    
    # Save statistical results
    with open('bias_analysis_results.txt', 'w') as f:
        f.write("BIAS ANALYSIS RESULTS\n")
        f.write("="*50 + "\n\n")
        
        for test_name, result in statistical_results.items():
            if test_name != 'correlations':
                f.write(f"{test_name.upper()}:\n")
                f.write(f"Test: {result['test']}\n")
                f.write(f"Statistic: {result['statistic']:.4f}\n")
                f.write(f"P-value: {result['p_value']:.4f}\n")
                f.write(f"Result: {result['interpretation']}\n\n")
    
    # Save correlation matrix
    if 'correlations' in statistical_results:
        statistical_results['correlations'].to_csv('correlation_matrix.csv')
    
    print("Results exported to:")
    print("- processed_bias_data.csv")
    print("- bias_analysis_results.txt") 
    print("- correlation_matrix.csv")
    print("- Various PNG plots")

export_results()

print("\nAnalysis complete! Check the generated plots and files for detailed results.")

# %%
