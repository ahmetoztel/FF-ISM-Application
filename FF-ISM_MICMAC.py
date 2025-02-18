import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font
import matplotlib.patches as patches

# Define the Fermatian Fuzzy number type
class FF:
    def __init__(self):
        self.Mf = 0.0
        self.NMf = 0.0

# Start file selection dialog
Tk().withdraw()  # Hide main window
file_path = askopenfilename(title="Select the expert opinions file", filetypes=[("Excel files", "*.xlsx *.xls")])

# Read the Excel file
data = pd.read_excel(file_path, header=None)

# Identify non-empty columns and rows
non_empty_columns = data.dropna(axis=1, how='all')
non_empty_rows = data.dropna(axis=0, how='all')

# Calculate the number of factors and experts
factor_count = non_empty_columns.shape[1]
expert_count = non_empty_rows.shape[0] // factor_count

# Convert expert opinions to a 3D numpy array
expert_opinions = np.array(non_empty_columns).reshape((expert_count, factor_count, factor_count))

# Initialize FFExpert array
FFExpert = np.empty((expert_count, factor_count, factor_count), dtype=object)

for exp in range(expert_count):
    for i in range(factor_count):
        for j in range(factor_count):
            FFExpert[exp, i, j] = FF()

# Convert expert opinions to Fermatian Fuzzy format
for exp in range(expert_count):
    for i in range(factor_count):
        for j in range(factor_count):
            value = expert_opinions[exp, i, j]
            if value == 0:
                FFExpert[exp, i, j].Mf = 0.0
                FFExpert[exp, i, j].NMf = 1.0
            elif value == 1:
                FFExpert[exp, i, j].Mf = 0.1
                FFExpert[exp, i, j].NMf = 0.8
            elif value == 2:
                FFExpert[exp, i, j].Mf = 0.4
                FFExpert[exp, i, j].NMf = 0.5
            elif value == 3:
                FFExpert[exp, i, j].Mf = 0.7
                FFExpert[exp, i, j].NMf = 0.2
            elif value == 4:
                FFExpert[exp, i, j].Mf = 0.9
                FFExpert[exp, i, j].NMf = 0.1

# Create Fermatian Fuzzy Decision Matrix (FFDec)
FFDec = np.empty((factor_count, factor_count), dtype=object)

for i in range(factor_count):
    for j in range(factor_count):
        FFDec[i, j] = FF()
        mf_sum = 0.0
        nmf_sum = 0.0

        for exp in range(expert_count):
            mf_sum += FFExpert[exp, i, j].Mf
            nmf_sum += FFExpert[exp, i, j].NMf

        FFDec[i, j].Mf = mf_sum / expert_count
        FFDec[i, j].NMf = nmf_sum / expert_count

# Create Crisp Decision Matrix
CrispDec = np.empty((factor_count, factor_count))

for i in range(factor_count):
    for j in range(factor_count):
        CrispDec[i, j] = (1 + 2 * (FFDec[i, j].Mf) ** 3 - (FFDec[i, j].NMf) ** 3) / 3

# Calculate threshold value
threshold_value = np.mean(CrispDec)

# Create Initial Reachability Matrix (IRM)
IRM = np.zeros((factor_count, factor_count))

for i in range(factor_count):
    for j in range(factor_count):
        if CrispDec[i, j] > threshold_value or i == j:
            IRM[i, j] = 1

FRM_backup = IRM.copy()  # Backup before transitive closure
# Create Final Reachability Matrix (FRM)  # Backup of FRM before any modifications
FRM = IRM.copy()

# Compute transitive closure using Warshall's algorithm
for k in range(factor_count):
    for i in range(factor_count):
        for j in range(factor_count):
            FRM[i, j] = max(FRM[i, j], min(FRM[i, k], FRM[k, j]))

# Backup FRM after transitive closure
FRM_backup = FRM.copy()

# MICMAC Analysis: Calculate Driving and Dependence Powers
DRIVING_POWER = FRM.sum(axis=1)  # Row sum (driving power)
DEPENDENCE_POWER = FRM.sum(axis=0)  # Column sum (dependence power)

# Create MICMAC Matrix
micmac_df = pd.DataFrame({
    'Factor': [f'Factor {i+1}' for i in range(factor_count)],
    'Driving Power': DRIVING_POWER,
    'Dependence Power': DEPENDENCE_POWER
})

# Determine Factor Types and Assign Colors
factor_colors = {
    'Driving': 'FF0000',     # Red
    'Linkage': '0000FF',     # Blue
    'Dependent': '008000',   # Green
    'Autonomous': 'FFD700'   # Yellow
}

micmac_df['Factor Type'] = micmac_df.apply(lambda row: (
    'Driving' if row['Driving Power'] > factor_count/2 and row['Dependence Power'] <= factor_count/2 else
    'Linkage' if row['Driving Power'] > factor_count/2 and row['Dependence Power'] > factor_count/2 else
    'Dependent' if row['Driving Power'] <= factor_count/2 and row['Dependence Power'] > factor_count/2 else
    'Autonomous'
), axis=1)

micmac_df['Color'] = micmac_df['Factor Type'].map(factor_colors)

# Factor Levels Determination
levels = []
remaining_factors = set(range(factor_count))
level = 1

while remaining_factors:
    current_level_factors = []
    for factor in remaining_factors:
        reachability_set = set(np.where(FRM[factor] == 1)[0])
        antecedent_set = set(np.where(FRM[:, factor] == 1)[0])
        intersection_set = reachability_set & antecedent_set

        if reachability_set == intersection_set:
            current_level_factors.append(factor)

    if not current_level_factors:
        break

    for factor in current_level_factors:
        levels.append({
            'Factor': factor + 1,
            'Level': level,
            'Reachability Set': list(map(int, np.where(FRM[factor] == 1)[0] + 1)),
            'Antecedent Set': list(map(int, np.where(FRM[:, factor] == 1)[0] + 1)),
            'Intersection Set': list(map(int, [x + 1 for x in (set(np.where(FRM[factor] == 1)[0]) & set(np.where(FRM[:, factor] == 1)[0]))]))
        })
        remaining_factors.remove(factor)
        FRM[factor, :] = 0
        FRM[:, factor] = 0

    level += 1

levels_df = pd.DataFrame(levels)

# Plot MICMAC Analysis Results
plt.figure(figsize=(10, 8))

# Group factors with the same coordinates
coordinate_groups = {}
for i in range(factor_count):
    coord = (DRIVING_POWER[i], DEPENDENCE_POWER[i])
    if coord not in coordinate_groups:
        coordinate_groups[coord] = []
    coordinate_groups[coord].append(f'{i+1}')

# Plot grouped factors
for coord, factors in coordinate_groups.items():
    color = f'#{micmac_df[micmac_df["Factor"] == f"Factor {factors[0]}"]["Color"].values[0]}'
    label = r"$Cha_{" + ",".join(factors) + r"}$"
    plt.scatter(coord[0], coord[1], color=color)
    plt.text(coord[0] + 0.1, coord[1] + 0.1, label, fontsize=12, color=color, fontweight='bold')

# Add quadrant lines
mid_x = factor_count / 2
mid_y = factor_count / 2
plt.axvline(x=mid_x, color='black', linestyle='--')
plt.axhline(y=mid_y, color='black', linestyle='--')

# Add quadrant labels at the top-right corner of each region
plt.text(factor_count * 1.2, factor_count * 0.95, "Linkage", fontsize=24, color='blue')
plt.text(factor_count * 0.2, factor_count * 0.95, "Dependent", fontsize=24, color='green')
plt.text(factor_count * 1.2, factor_count * 0.4, "Driving", fontsize=24, color='red')
plt.text(factor_count * 0.2, factor_count * 0.4, "Autonomous", fontsize=24, color='orange')

plt.xlabel('Driving Power', color='black', fontsize=16)
plt.ylabel('Dependence Power', color='black', fontsize=16)
plt.title('MICMAC Analysis Results', color='black', fontsize=24)
plt.grid(True)
plt.xlim(0, factor_count)
plt.ylim(0, factor_count)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.show()



# FFDec nesnelerini string formatına çevirme ve ondalık basamak düzenleme
def format_ff(x):
    mf_formatted = f"{x.Mf:.2f}".rstrip('0').rstrip('.')  # Ondalık kısmı ayarla
    nmf_formatted = f"{x.NMf:.2f}".rstrip('0').rstrip('.')  # Aynı işlemi NMf için de yap
    return f"({mf_formatted}, {nmf_formatted})"

FFDec_str = np.vectorize(format_ff)(FFDec)

# FRM_backup matrisindeki değişen 0 -> 1 elemanlarını belirleme ve formatlama
FRM_formatted = FRM_backup.astype(str)  # Matris elemanlarını string'e çevir

# 0'dan 1'e dönüşen elemanları bul ve "1*" olarak değiştir
for i in range(FRM_backup.shape[0]):
    for j in range(FRM_backup.shape[1]):
        if IRM[i, j] == 0 and FRM_backup[i, j] == 1:
            FRM_formatted[i, j] = "1*"

# Save All Results to a Single Excel File
with pd.ExcelWriter('MICMAC_Factor_Analysis.xlsx', engine='openpyxl') as writer:
    micmac_df.drop(columns=['Color']).to_excel(writer, sheet_name='MICMAC Results', index=False)
    pd.DataFrame(CrispDec).to_excel(writer, sheet_name='Crisp Decision Matrix', index=False)
    pd.DataFrame(IRM).to_excel(writer, sheet_name='Initial Reachability', index=False)
    pd.DataFrame(FRM_formatted).to_excel(writer, sheet_name='Final Reachability', index=False)  # Güncellenmiş FRM_backup
    levels_df.to_excel(writer, sheet_name='Factor Levels', index=False)
    pd.DataFrame(FFDec_str).to_excel(writer, sheet_name='Fermatean Fuzzy Decision Matrix', index=False)
# Plot Factor Levels
unique_levels = sorted(levels_df['Level'].unique())
num_levels = len(unique_levels)

fig, ax = plt.subplots(figsize=(8, num_levels * 2))  # Adjusted for proper portrait layout  # Adjusted for portrait layout  # Set for vertical layout

# Define colors for levels
level_colors = plt.get_cmap('tab20')

# Draw level sections
for idx, level in enumerate(unique_levels):
    y_bottom = idx
    y_top = idx + 1
    ax.add_patch(patches.Rectangle((0, y_bottom), 8, 1, color=level_colors(idx % 20), alpha=0.3))  # Ensure color cycles
    ax.text(8.5, y_bottom + 0.5, f'Level {level}', fontsize=14, verticalalignment='center')

# Faktör pozisyonlarını saklamak için bir sözlük oluşturuyoruz
factor_positions = {}
factor_levels_dict = {}  # Faktörlerin seviyelerini kaydetmek için bir sözlük

# Faktör seviyelerini çizme ve pozisyonları kaydetme
for level in unique_levels:
    level_factors = levels_df[levels_df['Level'] == level]['Factor'].tolist()
    num_factors = len(level_factors)
    start_x = 4 - (num_factors - 1) / 2

    for i, factor in enumerate(level_factors):
        x_pos = start_x + i
        y_pos = level - 0.5  # Level 1 en üstte olacak şekilde yerleştirildi
        size = 0.5  # Kutucuk boyutu
        ax.add_patch(
            patches.Rectangle((x_pos - size / 2, y_pos - size / 2), size, size, edgecolor='black', facecolor='white'))
        ax.text(x_pos, y_pos, f'{factor}', ha='center', va='center', fontsize=10)

        # Faktör pozisyonlarını ve seviyelerini kaydediyoruz
        factor_positions[factor] = (x_pos, y_pos)
        factor_levels_dict[factor] = level

# Okları çiziyoruz: FRM_backup üzerinden aynı seviye ve ardıl seviye ilişkilerini kontrol ederek
ok_sayisi = 0  # Çizilen okları takip etmek için

# Öncelikle her seviyedeki faktörleri sıralı olarak alacağız
for level in unique_levels:
    level_factors = sorted(levels_df[levels_df['Level'] == level]['Factor'].tolist())

    # Yatay oklar: Aynı seviyedeki yan yana faktörler için
    for idx in range(len(level_factors) - 1):
        factor_i = level_factors[idx]
        factor_j = level_factors[idx + 1]  # Sıradaki faktör yan yana

        start_pos = factor_positions[factor_i]
        end_pos = factor_positions[factor_j]

        # 1. Soldan sağa ilişki varsa mavi ok çiz
        if FRM_backup[factor_i - 1, factor_j - 1] == 1:
            ax.annotate("",
                        xy=(end_pos[0] - 0.3, end_pos[1]),  # Ok ucu
                        xytext=(start_pos[0] + 0.3, start_pos[1]),  # Ok başlangıcı
                        arrowprops=dict(arrowstyle="->", color='blue', lw=1))  # Mavi ok
            ok_sayisi += 1

        # 2. Sağdan sola ilişki varsa kırmızı ok çiz
        if FRM_backup[factor_j - 1, factor_i - 1] == 1:
            ax.annotate("",
                        xy=(start_pos[0] + 0.3, start_pos[1]),  # Ok ucu
                        xytext=(end_pos[0] - 0.3, end_pos[1]),  # Ok başlangıcı
                        arrowprops=dict(arrowstyle="->", color='red', lw=1))  # Kırmızı ok
            ok_sayisi += 1

# Dikey oklar: Ardıl seviyedeki faktörler için
for i in range(factor_count):
    for j in range(factor_count):
        if FRM_backup[i, j] == 1 and i != j:
            factor_i = i + 1
            factor_j = j + 1

            level_i = factor_levels_dict[factor_i]
            level_j = factor_levels_dict[factor_j]

            # Sadece ardıl seviyedeki faktörler arasında ok çiz
            if abs(level_j - level_i) == 1:
                start_pos = factor_positions[factor_i]
                end_pos = factor_positions[factor_j]

                ax.annotate("",
                            xy=(end_pos[0], end_pos[1] + 0.3),  # Ok ucu
                            xytext=(start_pos[0], start_pos[1] - 0.3),  # Ok başlangıcı
                            arrowprops=dict(arrowstyle="->", color='black', lw=1))  # Siyah ok
                ok_sayisi += 1

print(f"\nToplam çizilen ok sayısı: {ok_sayisi}")

# Grafik sınırlarını ayarlıyoruz
ax.set_xlim(0, 9)
ax.set_ylim(0, num_levels + 1)
ax.set_aspect('equal')
ax.axis('off')

# Başlık ve kaydetme
plt.title('Factor Levels', fontsize=16)
plt.savefig('Factor_Levels.pdf', format='pdf', bbox_inches='tight')
plt.show()

print("\nAll analysis results saved to 'MICMAC_Factor_Analysis.xlsx'.")
