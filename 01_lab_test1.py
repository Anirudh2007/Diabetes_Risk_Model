# Importing necessary libraries
import pandas as pd  # Data manipulation ke liye
import numpy as np   # Numerical calculations ke liye
import matplotlib.pyplot as plt  # Graphs aur plots banane ke liye
from scipy.stats import multivariate_normal  # Multivariate normal distribution ke liye
from mpl_toolkits.mplot3d import Axes3D  # 3D plots banane ke liye
import warnings
warnings.filterwarnings('ignore')  # Warning messages hide karne ke liye

# Dataset ko load karna
# Raw string use kiya hai taki Windows path mein backslash issue na ho
df = pd.read_csv(r'C:\Users\Toani\College\4th Sem\DSC 10\Project testing\archive\diabetes.csv')

# Dataset ki basic information print karna
print("Dataset shape:", df.shape)  # Kitne rows aur columns hain
print("Columns:", df.columns.tolist())  # Column names dekhene ke liye
print("\nFirst few rows:")  # Pehle kuch rows dikhane ke liye
print(df.head())

# Zero values wale columns identify karna
# Ye columns medical measurements hain jahan zero values missing values hain
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Noise ke liye standard deviation set karna
sigma_small = 0.1  # Chota standard deviation liya hai

print("\n=== DATA CLEANING ===")  # Data cleaning process start

# Har column ko process karna
for column in zero_columns:
    print(f"Processing {column}:")  # Konsa column process ho raha hai
    print(f"  Zeros before: {(df[column] == 0).sum()}")  # Pehle kitne zero the
    
    # Zero values ko NaN se replace karna (missing values mark karne ke liye)
    df[column].replace(0, np.nan, inplace=True)
    
    # Non-missing values ka mean calculate karna
    mu = df[column].mean()
    print(f"  Mean: {mu:.2f}")  # Mean value print karna
    print(f"  Missing values: {df[column].isnull().sum()}")  # Kitne missing values hain
    
    # Agar missing values hain toh unhe impute karna
    if df[column].isnull().sum() > 0:
        # Normal distribution se random noise generate karna
        epsilon = np.random.normal(0, sigma_small, size=df[column].isnull().sum())
        
        # Missing values ko impute karna: μ + ε formula use karke
        df.loc[df[column].isnull(), column] = mu + epsilon
        
        # Imputed values ko meaningful banane ke liye (negative values avoid karne)
        df[column] = df[column].clip(lower=0)
    
    print(f"  Zeros after: {(df[column] == 0).sum()}")  # Baad mein kitne zero hain
    print()

print("=== BIVARIATE NORMAL ANALYSIS ===")  # Bivariate analysis start

# Analysis ke liye features select karna aur NaN values remove karna
X = df[['Glucose', 'BMI']].dropna().values

print(f"Data points for analysis: {len(X)}")  # Kitne data points available hain

# Mean vector aur covariance matrix calculate karna
mean_vector = np.mean(X, axis=0)  # Dono features ka mean
cov_matrix = np.cov(X, rowvar=False)  # Covariance matrix

print("Mean Vector:", mean_vector)  # Mean vector print karna
print("Covariance Matrix:\n", cov_matrix)  # Covariance matrix print karna

# Check karna ki covariance matrix positive definite hai ya nahi
if np.any(np.linalg.eigvals(cov_matrix) <= 0):
    print("Warning: Covariance matrix is not positive definite. Adding small regularization.")
    cov_matrix += np.eye(2) * 1e-6  # Thoda regularization add karna

# Bivariate normal distribution create karna
try:
    bvn = multivariate_normal(mean=mean_vector, cov=cov_matrix)
except:
    # Agar error aaye toh identity covariance use karna
    cov_matrix = np.eye(2) * np.var(X, axis=0)
    bvn = multivariate_normal(mean=mean_vector, cov=cov_matrix)

# Scatter plot with contours banana
plt.figure(figsize=(12, 8))  # Figure size set karna
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, s=10, label='Data points', color='blue')  # Scatter plot

# Contour plot ke liye grid create karna
x = np.linspace(df['Glucose'].min(), df['Glucose'].max(), 100)  # Glucose range
y = np.linspace(df['BMI'].min(), df['BMI'].max(), 100)  # BMI range
X_grid, Y_grid = np.meshgrid(x, y)  # Grid points create karna
pos = np.dstack((X_grid, Y_grid))  # 3D positions banane ke liye
Z = bvn.pdf(pos)  # Probability density calculate karna

# Contour lines plot karna
contour = plt.contour(X_grid, Y_grid, Z, levels=5, colors='red', alpha=0.7, linewidths=2)
plt.clabel(contour, inline=True, fontsize=10)  # Contour labels add karna

# Plot ko customize karna
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title('Bivariate Normal Distribution: Glucose vs BMI')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3D Surface Plot banana
fig = plt.figure(figsize=(14, 10))  # Figure size
ax = fig.add_subplot(111, projection='3d')  # 3D subplot create karna

# 3D surface plot banana
surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', 
                      alpha=0.7, linewidth=0, antialiased=True)

# Peak point mark karna (mean vector)
peak_z = bvn.pdf(mean_vector)  # Peak ki height calculate karna
ax.scatter(mean_vector[0], mean_vector[1], peak_z, color='red', s=200, label='Mean Vector', marker='*')
ax.text(mean_vector[0], mean_vector[1], peak_z, " Mean Vector", color='red', fontsize=12)

# Spread points mark karna (mean ± standard deviation)
std_dev = np.sqrt(np.diag(cov_matrix))  # Standard deviation calculate karna
spread_points = [
    (mean_vector[0] + std_dev[0], mean_vector[1], bvn.pdf([mean_vector[0] + std_dev[0], mean_vector[1]])),
    (mean_vector[0] - std_dev[0], mean_vector[1], bvn.pdf([mean_vector[0] - std_dev[0], mean_vector[1]])),
    (mean_vector[0], mean_vector[1] + std_dev[1], bvn.pdf([mean_vector[0], mean_vector[1] + std_dev[1]])),
    (mean_vector[0], mean_vector[1] - std_dev[1], bvn.pdf([mean_vector[0], mean_vector[1] - std_dev[1]]))
]

# Different colors aur labels spread points ke liye
colors = ['orange', 'orange', 'purple', 'purple']
labels = ['μ+σ_X', 'μ-σ_X', 'μ+σ_Y', 'μ-σ_Y']

# Har spread point ko plot karna
for i, point in enumerate(spread_points):
    ax.scatter(point[0], point[1], point[2], color=colors[i], s=100, marker='o')
    ax.text(point[0], point[1], point[2], f" {labels[i]}", color=colors[i])

# Eigenvectors calculate karna tilt direction ke liye
eigvals, eigvecs = np.linalg.eig(cov_matrix)  # Eigen decomposition
largest_index = np.argmax(eigvals)  # Sabse bada eigenvalue
smallest_index = np.argmin(eigvals)  # Sabse chota eigenvalue
largest_eigvec = eigvecs[:, largest_index]  # Major axis direction
smallest_eigvec = eigvecs[:, smallest_index]  # Minor axis direction

print(f"Eigenvalues: {eigvals}")  # Eigenvalues print karna
print(f"Major axis eigenvector: {largest_eigvec}")  # Major axis print karna
print(f"Minor axis eigenvector: {smallest_eigvec}")  # Minor axis print karna

# Major aur minor axes ke liye arrows draw karna
scale = 30  # Arrow length ke liye scaling factor
# Major axis arrow
ax.quiver(mean_vector[0], mean_vector[1], peak_z, 
          largest_eigvec[0]*scale, largest_eigvec[1]*scale, 0, 
          color='red', label='Major Axis', linewidth=3, arrow_length_ratio=0.1)
# Minor axis arrow
ax.quiver(mean_vector[0], mean_vector[1], peak_z, 
          smallest_eigvec[0]*scale, smallest_eigvec[1]*scale, 0, 
          color='green', label='Minor Axis', linewidth=3, arrow_length_ratio=0.1)

# 3D plot ko customize karna
ax.set_xlabel('Glucose', fontsize=12)
ax.set_ylabel('BMI', fontsize=12)
ax.set_zlabel('Probability Density', fontsize=12)
ax.set_title('3D Bivariate Normal Distribution', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()

print("\n=== CONDITIONAL PROBABILITY ANALYSIS ===")  # Conditional probability analysis start

# Data ko outcome ke hisaab se split karna
non_diabetic = df[df['Outcome'] == 0][['Glucose', 'BMI']].dropna()  # Non-diabetic patients
diabetic = df[df['Outcome'] == 1][['Glucose', 'BMI']].dropna()  # Diabetic patients

print(f"Non-diabetic samples: {len(non_diabetic)}")  # Non-diabetic count
print(f"Diabetic samples: {len(diabetic)}")  # Diabetic count

# Prior probabilities calculate karna
total_patients = len(non_diabetic) + len(diabetic)  # Total patients
p_non_diabetic = len(non_diabetic) / total_patients  # P(non-diabetic)
p_diabetic = len(diabetic) / total_patients  # P(diabetic)

print(f"Prior P(Non-diabetic): {p_non_diabetic:.3f}")  # Prior probability print
print(f"Prior P(Diabetic): {p_diabetic:.3f}")  # Prior probability print

# Dono groups ke liye mean vectors aur covariance matrices calculate karna
mu_non_diabetic = np.mean(non_diabetic.values, axis=0)  # Non-diabetic mean
mu_diabetic = np.mean(diabetic.values, axis=0)  # Diabetic mean

cov_non_diabetic = np.cov(non_diabetic.values, rowvar=False)  # Non-diabetic covariance
cov_diabetic = np.cov(diabetic.values, rowvar=False)  # Diabetic covariance

print("\nNon-diabetic group:")  # Non-diabetic group statistics
print(f"  Mean: {mu_non_diabetic}")
print(f"  Covariance:\n{cov_non_diabetic}")

print("\nDiabetic group:")  # Diabetic group statistics
print(f"  Mean: {mu_diabetic}")
print(f"  Covariance:\n{cov_diabetic}")

# Covariance matrices ko positive definite banane ke liye regularization add karna
cov_non_diabetic += np.eye(2) * 1e-6
cov_diabetic += np.eye(2) * 1e-6

# Dono groups ke liye bivariate normal distributions create karna
bvn_non_diabetic = multivariate_normal(mean=mu_non_diabetic, cov=cov_non_diabetic)
bvn_diabetic = multivariate_normal(mean=mu_diabetic, cov=cov_diabetic)

# Diabetes probability calculate karne wala function
def probability_diabetic(glucose, bmi):
    """
    P(Outcome = 1 | Glucose = glucose, BMI = bmi) calculate karta hai
    Bayes' theorem use karke
    """
    try:
        # Likelihoods calculate karna
        f0 = bvn_non_diabetic.pdf([glucose, bmi])  # P(features | non-diabetic)
        f1 = bvn_diabetic.pdf([glucose, bmi])      # P(features | diabetic)
        
        # Numerical issues avoid karne ke liye
        if f0 == 0 and f1 == 0:
            return 0.5  # Neutral probability return karna
        
        # Bayes' theorem use karke posterior probability calculate karna
        numerator = f1 * p_diabetic  # Numerator
        denominator = f0 * p_non_diabetic + f1 * p_diabetic  # Denominator
        
        return numerator / denominator  # Final probability
    except:
        return 0.5  # Error case mein neutral probability return karna

# 2D grid create karna probability calculation ke liye
glucose_range = np.linspace(df['Glucose'].min(), df['Glucose'].max(), 80)  # Glucose range
bmi_range = np.linspace(df['BMI'].min(), df['BMI'].max(), 80)  # BMI range
G, B = np.meshgrid(glucose_range, bmi_range)  # Grid points

# Har grid point ke liye probability calculate karna
print("Computing probability grid...")  # Progress message
prob_grid = np.zeros_like(G)  # Empty grid initialize karna
for i in range(G.shape[0]):  # Rows traverse karna
    for j in range(G.shape[1]):  # Columns traverse karna
        prob_grid[i, j] = probability_diabetic(G[i, j], B[i, j])  # Probability calculate karna

# Heatmap create karna
plt.figure(figsize=(14, 10))  # Figure size
heatmap = plt.contourf(G, B, prob_grid, levels=50, cmap='RdBu_r', alpha=0.9)  # Filled contours
plt.colorbar(heatmap, label='P(Diabetic | Glucose, BMI)')  # Colorbar add karna

# Specific probability levels ke liye contour lines add karna
contour_lines = plt.contour(G, B, prob_grid, levels=[0.2, 0.5, 0.8], 
                           colors=['blue', 'green', 'red'], linewidths=3, linestyles='--')
plt.clabel(contour_lines, inline=True, fontsize=12, fmt='%.1f')  # Contour labels

# Actual data points ko plot karna outcome ke hisaab se color karke
plt.scatter(non_diabetic['Glucose'], non_diabetic['BMI'], 
           c='blue', alpha=0.7, s=30, label='Non-diabetic', edgecolors='black', linewidth=0.5)
plt.scatter(diabetic['Glucose'], diabetic['BMI'], 
           c='red', alpha=0.7, s=30, label='Diabetic', edgecolors='black', linewidth=0.5)

# Plot ko finalize karna
plt.xlabel('Glucose', fontsize=12)
plt.ylabel('BMI', fontsize=12)
plt.title('Diabetes Risk Heatmap: P(Outcome=1 | Glucose, BMI)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Final summary print karna
print("\n=== ANALYSIS COMPLETE ===")
print("Key Findings:")
print(f"- Glucose mean: {mean_vector[0]:.2f}, BMI mean: {mean_vector[1]:.2f}")
print(f"- Correlation between Glucose and BMI: {cov_matrix[0,1]/(std_dev[0]*std_dev[1]):.3f}")
print(f"- Diabetes prevalence in dataset: {p_diabetic:.3f}")

# Data Cleaning: Zero values ko realistic values se replace kiya hai

# Bivariate Analysis: Glucose aur BMI ka joint distribution study kiya hai

# 3D Visualization: Bell-shaped curve properly dikhaya hai

# Risk Prediction: Bayes theorem use karke diabetes risk calculate kiya hai

# Heatmap: Red color high risk, blue color low risk dikhata hai