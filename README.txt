================================================================================
                             DOOIT - DATA VISUALIZATION TOOL
================================================================================

DOOIT (Data Observation & Overlay Insights Tool) is an interactive Python GUI 
application for exploratory data analysis (EDA) and advanced visualization of 
CSV datasets. It provides powerful pair plotting, overlay plotting, and 
statistical analysis capabilities.

================================================================================
                                  HOW TO USE
================================================================================

INSTALLATION:
-------------
1. Install required dependencies:
   pip install -r requirements.txt

2. Run the application:
   python DOOIT.py

BASIC WORKFLOW:
--------------
1. Click "Load CSV" to select your dataset
2. Follow the interactive prompts to:
   - Impute missing values (optional)
   - Convert binary columns to integers (optional)
   - Filter by non-numeric columns (optional)
   - Create a SUM column from numeric features (optional)
   - Create custom calculated columns (optional)
   - Delete unwanted columns (optional)
   - Load threshold values for comparison (optional)
   - Highlight specific features in pair plot (optional)

3. View the automatic pair plot visualization
4. Select X and Y features for overlay plotting
5. Configure plot options and generate overlay plots
6. Export processed data and configuration as needed

================================================================================
                            FUNCTION BREAKDOWN
================================================================================

CORE DATA PROCESSING FUNCTIONS:
-------------------------------

clean_numeric_df(df)
    Cleans and prepares DataFrame for analysis:
    - Replaces '?' with NaN
    - Converts columns to numeric where possible
    - Preserves datetime columns
    - Drops non-numeric columns and rows with NaN values
    - Returns cleaned DataFrame ready for plotting

plot_pairgrid(df, highlight_feature=None, thresholds=None)
    Generates comprehensive pairwise relationship visualization:
    - Diagonal: Histograms with KDE and detailed statistics (mean, median, 
      min, max, Q1, Q3)
    - Lower triangle: Scatter plots showing relationships between feature pairs
    - Upper triangle: Pearson correlation coefficients and p-values with 
      color-coded significance
    - Optional feature highlighting with dark background and cyan markers
    - Optional threshold lines (vertical/horizontal) on relevant plots
    - Automatically displays after loading data

plot_overlay(df, x_col, y_cols, x_intercepts=None, x_names=None, 
             y_intercepts=None, y_names=None, df_comparison=None)
    Creates advanced overlay plots with multiple visualization modes:
    
    BASIC MODES:
    - Single Y-axis: One Y feature vs X feature
    - Multiple Y-axes: Up to 5 Y features with independent twin axes
    
    ADVANCED MODES:
    - Stack Mode: Normalizes Y features to [0,1] and stacks them vertically
    - Trendline Mode: Adds linear regression lines with slope annotations
    - Residuals Mode: Plots deviations from trendline instead of raw values
    - Bootstrap Mode: Generates 10,000 resampled data points with coefficient 
      variation from nearby points
    - Alpha Mode: Adjusts point transparency based on data density (reduces 
      overplotting)
    - Annotation Mode: Adds statistical summary boxes (mean, median, min, max, 
      slope)
    - Comparison Mode: Overlays a second dataset with "+" prefix markers
    
    FEATURES:
    - Custom vertical lines (x_intercepts) with labels for event markers
    - Custom horizontal lines (y_intercepts) with labels for thresholds
    - Automatic axis alignment for fair comparison
    - Dark theme with vibrant colors for better visibility
    - Scatter or line+marker plot styles

GUI CLASSES:
-----------

CustomColumnDialog(df, parent=None)
    Dialog for creating calculated columns:
    - Assigns DataFrame columns to variables (a-j)
    - Supports mathematical formulas using numpy and math functions
    - Available functions include: abs, sqrt, log, exp, sin, cos, tan, 
      arcsin, arccos, arctan, arctan2, round, floor, ceil, clip, pow, 
      degrees, radians, and pi constant
    - Example: Create a column using formula "sqrt(a**2 + b**2)" where 
      a and b are selected columns
    - Returns column name and calculated values

DataFrameExportDialog(df, parent=None)
    Non-modal dialog for viewing current DataFrame:
    - Displays full DataFrame in scrollable table
    - Shows all rows and columns with current values
    - Allows visual inspection of data after processing
    - Appears automatically after loading data

PairPlotApp
    Main application window and GUI controller:
    - Manages all user interactions
    - Coordinates data loading and processing pipeline
    - Handles feature selection for overlay plots
    - Controls plot configuration options
    - Exports processed data and configuration

    MAJOR METHODS:
    
    load_csv()
        Primary data loading function that orchestrates:
        1. CSV file selection and reading
        2. Missing value imputation (forward fill)
        3. Timestamp conversion to datetime
        4. Binary column detection and integer mapping
        5. Multi-column filtering with AND/OR logic
        6. SUM column creation from selected features
        7. Custom column creation with formulas
        8. Column deletion
        9. Data cleaning and validation
        10. Threshold loading (latest value or min/max range)
        11. Feature highlighting selection
        12. Automatic pair plot generation
        13. DataFrame preview display
        14. Feature list population for overlay plotting
        
    load_comparison_dataset()
        Loads a second dataset for comparison:
        - Reads CSV file
        - Applies same cleaning process
        - Stores separately for overlay comparison mode
        - Updates GUI to show loaded filename
    
    show_overlay()
        Generates overlay plots based on user selections:
        - Validates feature selections
        - Removes X feature from Y features if selected
        - Applies all toggle settings (stack, trendline, residuals, etc.)
        - Parses intercept values and names
        - Calls plot_overlay() with configured parameters
    
    export_csv()
        Exports current processed DataFrame to CSV:
        - Saves all transformations and calculations
        - Preserves column names and data types
        - Does not include row indices
    
    export_dooit_inputs()
        Exports complete analysis configuration:
        - All file paths (dataset, comparison, thresholds)
        - Data processing steps in order
        - Binary column mappings
        - Filter conditions and logic
        - Custom column formulas and variable assignments
        - Feature selections for both plots
        - All plot configuration toggles
        - Intercept values and names
        - Enables exact recreation of analysis

HELPER DIALOG CLASSES:
---------------------

MultiFilterDialog
    Advanced filtering interface:
    - Toggle up to 20 simultaneous filters
    - Select column, value, and combining logic (AND/OR) for each
    - Dynamic value dropdowns based on selected column
    - Builds complex query strings for DataFrame filtering

SumDialog
    Sum column creator:
    - Checkbox selection of numeric features to sum
    - Custom name input for resulting column
    - Validates selections before creation

BinaryValueDialog
    Binary value mapper:
    - Appears when 2-value columns detected
    - User selects which value represents 0 (other becomes 1)
    - Converts categorical binary data to numeric

ThresholdMethodDialog
    Threshold interpretation selector:
    - Latest value: Uses last row from threshold CSV
    - Min/Max range: Uses minimum and maximum values from threshold CSV
    - Applied to pair plot visualization

HighlightDialog
    Feature highlighter:
    - Dropdown of numeric features
    - Selected feature shown with dark background and red/cyan markers
    - Helps identify important relationships in pair plot

DeleteColumnsDialog
    Column deletion interface:
    - Checkbox list of all columns
    - Removes selected columns from analysis
    - Applied after all other processing

================================================================================
                              GUI FEATURES
================================================================================

MAIN WINDOW SECTIONS:
--------------------

1. File Loading
   - "Load CSV" button
   - "Load Comparison Dataset" button
   - Status labels showing loaded files

2. Feature Selection
   - X-axis feature selector (single selection)
   - Y-axis feature selector (multi-selection, up to 5)

3. Plot Type
   - Scatter plot (default)
   - Line with markers

4. Data Toggles (6 options)
   - Stack Y features: Normalize and offset vertically
   - Bootstrap data: Generate 10k resampled points
   - Annotate with stats: Show summary boxes
   - Plot residuals: Show deviations from trendline
   - Show trendline: Add linear regression lines
   - Vary transparency: Adjust alpha by data density

5. Comparison Dataset
   - Load button and status
   - "Plot Comparison" checkbox to overlay second dataset

6. Event/Threshold Markers
   - X-intercepts input (comma-separated values)
   - X-intercept names input (comma-separated labels)
   - Y-intercepts input (comma-separated values)
   - Y-intercept names input (comma-separated labels)

7. Action Buttons
   - "Show Overlay Plot" - Generate configured plot
   - "Export DataFrame as CSV" - Save processed data
   - "Export DOOIT Inputs" - Save full configuration

================================================================================
                              DATA REQUIREMENTS
================================================================================

INPUT DATA:
----------
- CSV format with header row
- At least 2 numeric columns for visualization
- Optional: 'timestamp' column (Unix timestamp in seconds)
- Optional: Non-numeric columns for filtering
- Optional: Binary columns (will be detected and converted)

COMPARISON DATASET:
------------------
- Same format as main dataset
- Should contain matching column names for overlay
- Can have different number of rows

THRESHOLD CSV:
-------------
- Columns should match feature names in main dataset
- Each row represents threshold values
- "Latest" method uses last row
- "Min/Max" method uses column min/max values

================================================================================
                              TIPS & TRICKS
================================================================================

PERFORMANCE:
-----------
- Large datasets (>10,000 rows): Use alpha mode to reduce overplotting
- Many features: Select subset for overlay plots (max 5 Y features)
- Bootstrap mode significantly increases processing time

VISUALIZATION:
-------------
- Use stack mode to compare trends of features with different scales
- Residuals mode helps identify patterns in model errors
- Comparison mode works best with aligned datasets (same X range)
- Threshold lines appear as orange (min) and red (max) dashed lines

DATA PREPARATION:
----------------
- Remove or impute missing values before complex analysis
- Create SUM columns to analyze combined effects
- Use custom columns for derived metrics (e.g., ratios, differences)
- Filter by non-numeric columns to focus on specific subsets

WORKFLOW OPTIMIZATION:
---------------------
- Export DOOIT Inputs after configuring a useful analysis
- Use this export to document your analysis pipeline
- Compare multiple datasets by loading different comparison files
- Highlight features in pair plot to identify key relationships

================================================================================
                              TROUBLESHOOTING
================================================================================

COMMON ISSUES:
-------------
Q: "Not enough numeric columns for plotting" error?
A: Ensure your CSV has at least 2 columns with numeric data. Check for 
   non-numeric characters in data columns.

Q: Pair plot is too crowded?
A: Delete unnecessary columns before generating the plot, or highlight a 
   specific feature to focus attention.

Q: Overlay plot axes don't align for comparison?
A: The tool automatically aligns axes for multi-Y and residuals modes. Ensure 
   both datasets have reasonable value ranges.

Q: Custom column formula fails?
A: Check formula syntax. Use numpy/math functions as shown in examples 
   (e.g., "np.sqrt(a)" not "sqrt(a)"). Ensure all variables are assigned.

Q: Threshold lines not appearing?
A: Verify threshold CSV column names exactly match feature names. Check that 
   threshold method is selected.

================================================================================
                            EXAMPLE USE CASES
================================================================================

1. TIME SERIES ANALYSIS:
   - Load CSV with 'timestamp' column
   - Select timestamp as X-axis
   - Select multiple metrics as Y-features
   - Enable trendline to see trends over time
   - Add threshold lines for acceptable ranges

2. MACHINE LEARNING FEATURE EXPLORATION:
   - Load training data CSV
   - Create SUM column for combined features
   - Use pair plot to identify correlations
   - Highlight target variable
   - Export filtered data for modeling

3. A/B TEST COMPARISON:
   - Load control group data
   - Load treatment group as comparison dataset
   - Filter by test segment
   - Enable comparison mode
   - Overlay metrics to visualize differences

4. QUALITY CONTROL:
   - Load measurement data
   - Load specification limits as thresholds
   - Use residuals mode to detect systematic errors
   - Bootstrap to understand variation
   - Export flagged data points

================================================================================
                              VERSION INFO
================================================================================

Dependencies:
- Python 3.x
- numpy: Array operations and mathematical functions
- pandas: Data manipulation and CSV handling
- matplotlib: Plotting backend
- seaborn: Statistical visualizations
- scipy: Statistical tests (Pearson correlation)
- PyQt5: GUI framework
- scikit-learn: Data normalization (MinMaxScaler)

For questions, issues, or contributions, please refer to the project 
documentation or contact the development team.

================================================================================
                                END OF README
================================================================================
