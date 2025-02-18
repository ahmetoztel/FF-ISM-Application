# FF-ISM-Application
```markdown
# MICMAC Factor Analysis with Fermatean Fuzzy Numbers

This Python script performs MICMAC (Matrix Cross-Impact Matrix Multiplication Applied to Classification) analysis using Fermatean Fuzzy numbers to analyze factor relationships in decision-making processes.

## Features

- 🎯 **Fermatean Fuzzy Number Conversion**: Converts expert opinions into fuzzy numbers
- 📊 **MICMAC Analysis**: Calculates driving/dependence powers and classifies factors
- 📈 **Interactive Visualization**: Generates MICMAC chart and factor hierarchy diagram
- 📥 **Excel Integration**: Processes Excel input and exports comprehensive results
- 🧩 **Automatic Level Detection**: Identifies factor hierarchy levels using reachability matrices

## Requirements

- Python 3.6+
- Required libraries:
  ```bash
  pip install pandas numpy matplotlib openpyxl tk
  ```

## Usage

1. **Prepare Input File**:
   - Create Excel file with expert opinions matrix
   - Use 0-4 scale for relationships (0 = No influence, 4 = Very strong influence)
   - Example structure:
     ```
     | Expert 1 Matrix | Expert 2 Matrix | ... |
     |-----------------|-----------------|-----|
     | 0 3 1 ...       | 2 1 4 ...       |     |
     ```

2. **Run the Script**:
   ```bash
   python micmac_analysis.py
   ```
   - A file dialog will appear to select your Excel file
   - Analysis runs automatically after file selection

3. **Outputs**:
   - 📄 `MICMAC_Factor_Analysis.xlsx` with 6 sheets:
     1. MICMAC Results
     2. Crisp Decision Matrix
     3. Reachability Matrices
     4. Factor Levels
     5. Fermatean Fuzzy Matrix
   - 📈 Two interactive matplotlib windows showing:
     - MICMAC Analysis Chart
     - Factor Hierarchy Diagram

## Key Algorithms

1. **Fermatean Fuzzy Conversion**:
   ```python
   Value | Mf  | NMf
   ------|-----|-----
   0     | 0.0 | 1.0
   1     | 0.1 | 0.8
   2     | 0.4 | 0.5
   3     | 0.7 | 0.2
   4     | 0.9 | 0.1
   ```

2. **Crisp Matrix Calculation**:
   ```python
   CrispDec[i,j] = (1 + 2*(Mf³) - (NMf³) / 3
   ```

3. **Warshall's Algorithm**: For transitive closure in reachability matrix

## Visualization Examples

![MICMAC Analysis](https://via.placeholder.com/400x300.png?text=MICMAC+Chart)
![Factor Hierarchy](https://via.placeholder.com/300x500.png?text=Factor+Levels)

## Code Structure

```text
1. File Selection Dialog
2. Data Loading & Validation
3. Expert Opinion Processing
4. Fermatean Fuzzy Conversion
5. Decision Matrix Creation
6. Threshold Calculation
7. Reachability Matrices
8. MICMAC Analysis
9. Factor Level Determination
10. Visualization & Export
```

## Contributing

Contributions welcome! Please fork the repository and submit pull requests for:
- Additional fuzzy number types
- Improved visualization capabilities
- Alternative threshold calculation methods

## License

MIT License - Free for academic and commercial use
```

**Note:** Replace placeholder images with actual screenshots from your analysis results.
