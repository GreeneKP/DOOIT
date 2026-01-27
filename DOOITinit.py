import sys
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
from scipy.stats import pearsonr
from numpy.polynomial.polynomial import Polynomial
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QRadioButton, QButtonGroup,
    QDialog, QComboBox, QCheckBox, QHBoxLayout, QLineEdit, QFrame, QTableWidget, QTableWidgetItem, QAbstractScrollArea
)
from sklearn.preprocessing import MinMaxScaler
from numpy import abs, sqrt, log, exp, sin, cos, tan, arcsin, arccos, arctan, round, floor, ceil, clip
from math import atan2, pow, degrees, radians, pi

def clean_numeric_df(df):
    """
    Cleans the input DataFrame by:
    - Replacing '?' with np.nan for easier numeric conversion.
    - Attempting to convert all columns to float. If conversion fails, drops the column.
    - Returns a DataFrame containing only numeric columns, with rows containing NaNs dropped.
    This ensures the data is ready for numeric analysis and plotting.
    """
    df = df.replace('?', np.nan)
    # Preserve datetime columns (datetime64[ns]) and numeric columns; drop others
    cols_to_keep = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            cols_to_keep.append(col)
            continue
        # Try numeric conversion for other columns
        try:
            df[col] = df[col].astype(float)
            cols_to_keep.append(col)
        except Exception:
            # Not numeric and not datetime -> drop
            pass
    df_clean = df[cols_to_keep]
    # Only drop rows that have NaNs in numeric columns (keep rows if only datetime is NaN)
    numeric_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
    if numeric_cols:
        df_clean = df_clean.dropna(subset=numeric_cols)
    return df_clean

def plot_pairgrid(df, highlight_feature=None, thresholds=None):
    """
    Generates a seaborn PairGrid for the given DataFrame, visualizing pairwise relationships between numeric features.
    - Diagonal: Histogram and KDE for each feature.
    - Lower triangle: Scatter plots for feature pairs.
    - Upper triangle: Annotates Pearson correlation coefficient and p-value for each feature pair.
      If p >= 0.05, annotation is black with white font (not significant); otherwise, uses a colormap.
    - Adds a title and tight layout for clarity.
    """
    import matplotlib.cm as cm
    from scipy.stats import pearsonr
    def annotate_corr(x, y, **kwargs):
        """
        Annotates the upper triangle of the PairGrid with Pearson correlation and p-value.
        Uses color mapping for significant correlations, black/white for non-significant.
        """
        ax = plt.gca()
        if len(x) > 1 and len(y) > 1:
            r, p = pearsonr(x, y)
            if p >= 0.05:
                facecolor = 'black'
                fontcolor = 'white'
            else:
                norm = plt.Normalize(-1, 1)
                cmap = cm.coolwarm
                facecolor = cmap(norm(r))
                fontcolor = 'black'
            ax.annotate(f'r={r:.2f}\np={p:.2g}', xy=(0.5, 0.5), xycoords='axes fraction',
                        ha='center', va='center', fontsize=14, color=fontcolor,
                        bbox={'facecolor': facecolor, 'alpha': 0.7, 'pad': 20})
            ax.set_axis_off()
    def diag_hist_with_stats(x, color='magenta', **kwargs):
        ax = plt.gca()
        # Plot histogram and KDE
        sns.histplot(x, kde=True, color=color, ax=ax)
        # Calculate stats
        mean = np.mean(x)
        median = np.median(x)
        min_val = np.min(x)
        max_val = np.max(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
    # Format annotation text for upper right (mean, median, Q1, Q3)
        stats_text = (f"Q1: {q1:.2f}\nMean: {mean:.2f}\nQ3: {q3:.2f}\n")
        # Format annotation text for upper left (min, max)
        minmax_text = (f"Max: {max_val:.2f}\nMedian: {median:.2f}\nMin: {min_val:.2f}")
        # Place annotation in upper right
        ax.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=8,
                bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 0})
        # Place annotation in upper left
        ax.annotate(minmax_text, xy=(0.02, 0.98), xycoords='axes fraction',
                ha='left', va='top', fontsize=8,
                bbox={'facecolor': 'white', 'alpha': 0.0, 'pad': 0})
    import inspect
    def custom_scatter(x, y, **kwargs):
        ax = plt.gca()
        hf = highlight_feature
        # Use PairGrid's x_vars and y_vars to determine which axes are being plotted
        # kwargs['label'] and kwargs['label2'] are not always reliable, so use ax.get_title() as fallback
        x_var = getattr(ax, '_x_var', None)
        y_var = getattr(ax, '_y_var', None)
        # If not set, try to infer from axis labels
        if x_var is None or y_var is None:
            x_label = ax.get_xlabel()
            y_label = ax.get_ylabel()
            x_var = x_label if x_label in df.columns else None
            y_var = y_label if y_label in df.columns else None
        highlight = hf and (x_var == hf or y_var == hf)
        if highlight:
            ax.set_facecolor('#222222')
            sns.scatterplot(x=x, y=y, color='cyan', alpha=0.7, ax=ax)
        else:
            sns.scatterplot(x=x, y=y, color='blue', alpha=0.5, ax=ax)
    g = sns.PairGrid(df)
    g.map_diag(diag_hist_with_stats, color='magenta')
    # Custom lower triangle mapping to ensure all relevant plots are highlighted and threshold lines are drawn
    for i, x_var in enumerate(g.x_vars):
        for j, y_var in enumerate(g.y_vars):
            if j > i:  # lower triangle only
                ax = g.axes[j, i]
                ax._x_var = x_var
                ax._y_var = y_var
                x = df[x_var]
                y = df[y_var]
                highlight = highlight_feature and (x_var == highlight_feature or y_var == highlight_feature)
                if highlight:
                    ax.set_facecolor('#222222')
                    sns.scatterplot(x=x, y=y, color='red', alpha=0.7, ax=ax)
                else:
                    sns.scatterplot(x=x, y=y, color='blue', alpha=0.5, ax=ax)
                # Plot threshold lines if available
                if thresholds:
                    # If x_var has a threshold, plot vertical line
                    if x_var in thresholds:
                        ax.axvline(thresholds[x_var], color='orange', linestyle='--', linewidth=2, label=f'{x_var} threshold')
                    # If y_var has a threshold, plot horizontal line
                    if y_var in thresholds:
                        ax.axhline(thresholds[y_var], color='orange', linestyle='--', linewidth=2, label=f'{y_var} threshold')
    g.map_upper(annotate_corr)
    plt.suptitle('DOOIT!', y=1.08, fontsize=26,color='black')
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('DOOIT! Pair Plot')
    plt.show()
    plt.tight_layout()

def plot_overlay(df, x_col, y_cols, x_intercepts=None, x_names=None, y_intercepts=None, y_names=None):
    # ...existing code...
    """
    Generates an overlay plot for selected features.
    - x_col: Feature for the x-axis.
    - y_cols: List of features for the y-axis (up to 5).
    - If multiple y features, normalizes them to [0, 1] for comparison.
    - Plots each y feature against x_col as a scatter plot or line+marker plot based on plot_type.
    - If stacking is enabled, each normalized feature is offset by +1 on the Y axis.
    - If trendline is enabled, plots a linear fit for each Y feature.
    - If residuals is enabled, plots absolute residuals from the trendline (sqrt(residual^2)).
    - If bootstrap is enabled, plots 10,000 bootstrapped samples from the data.
    - Plots vertical lines for x_intercepts and horizontal lines for y_intercepts, with labels.
    - Adds axis labels, legend, title, and tight layout for clarity.
    """
    import numpy as np
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('#222222')  # dark grey background
    # Bootstrap logic
    if getattr(plot_overlay, 'bootstrap', False):
        # Sample 10,000 rows with replacement
        boot_idx = np.random.choice(df.index, size=10000, replace=True)
        df_boot = df.loc[boot_idx].reset_index(drop=True)
        # For each Y feature, apply random coefficient from adjacent points in original data
    x_vals = df[x_col]
    if getattr(plot_overlay, 'residuals', False):
        # Plot residuals for each Y-feature
        for i, y in enumerate(y_cols):
            y_vals = df[y]
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            residuals = y_vals - p(x_vals)
            color = f'C{i}'
            if plot_overlay.plot_type == 'scatter':
                plt.scatter(x_vals, residuals, label=f'{y} residuals', color=color)
            else:
                sort_idx = np.argsort(x_vals)
                plt.plot(np.array(x_vals)[sort_idx], np.array(residuals)[sort_idx], marker='o', label=f'{y} residuals', color=color)
            # Trendline slope annotation for residuals (far right)
            if getattr(plot_overlay, 'trendline', False):
                slope = z[0]
                x_sorted = np.array(x_vals)[np.argsort(x_vals)]
                residuals_sorted = np.array(residuals)[np.argsort(x_vals)]
                x_right = max(x_sorted)
                y_right = residuals_sorted[-1]
                x_annot = x_right - 0.03 * (max(x_sorted) - min(x_sorted))
                y_annot = y_right + 0.05 * (max(residuals_sorted) - min(residuals_sorted))
                plt.text(x_annot, y_annot, f'Slope: {slope:.3f}', color='white', ha='right', va='bottom', fontsize=10, fontweight='bold', bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2})
        plt.ylabel('Residuals')
    elif getattr(plot_overlay, 'stack', False):
        # Normalize all features to [0, 1], then offset each by +1
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        y_norm = pd.DataFrame(scaler.fit_transform(df[y_cols]), columns=y_cols)
        offset = 0
        for i, y in enumerate(y_cols):
            vals = y_norm[y] + offset
            if plot_overlay.plot_type == 'scatter':
                plt.scatter(x_vals, vals, label=f'{y} (stack {i+1})')
            else:
                sort_idx = np.argsort(x_vals)
                plt.plot(np.array(x_vals)[sort_idx], np.array(vals)[sort_idx], marker='o', label=f'{y} (stack {i+1})')
            # Trendline
            if getattr(plot_overlay, 'trendline', False):
                z = np.polyfit(x_vals, vals, 1)
                p = np.poly1d(z)
                x_sorted = np.array(x_vals)[np.argsort(x_vals)]
                plt.plot(x_sorted, p(x_sorted), linestyle='--', color='white', alpha=0.9, label=f'{y} trendline')
                # Annotate slope just above and to the left of the trendline at far right
                slope = z[0]
                x_right = max(x_sorted)
                y_right = p(x_right)
                x_annot = x_right - 0.03 * (max(x_sorted) - min(x_sorted))
                y_annot = y_right + 0.05 * (max(vals) - min(vals))
                plt.text(x_annot, y_annot, f'Slope: {slope:.3f}', color='white', ha='right', va='bottom', fontsize=10, fontweight='bold', bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2})
            offset += 1
        plt.ylabel('Stacked Normalized Y Features')
    elif len(y_cols) > 1:
        # Dual y-axes for multiple features
        ax = plt.gca()
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        lines = []
        labels = []
        twin_axes = [ax]
        for i, y in enumerate(y_cols):
            color = colors[i % len(colors)]
            if i == 0:
                if plot_overlay.plot_type == 'scatter':
                    l = ax.scatter(x_vals, df[y], color=color, label=y)
                else:
                    sort_idx = np.argsort(x_vals)
                    l, = ax.plot(np.array(x_vals)[sort_idx], np.array(df[y])[sort_idx], marker='o', color=color, label=y)
                ax.set_ylabel(y)
                lines.append(l)
                labels.append(y)
                # Trendline
                if getattr(plot_overlay, 'trendline', False):
                    z = np.polyfit(x_vals, df[y], 1)
                    p = np.poly1d(z)
                    x_sorted = np.array(x_vals)[np.argsort(x_vals)]
                    l2, = ax.plot(x_sorted, p(x_sorted), linestyle='--', color='white', alpha=0.9, label=f'{y} trendline')
                    lines.append(l2)
                    labels.append(f'{y} trendline')
                    # Annotate slope just above and to the left of the trendline at far right
                    slope = z[0]
                    x_right = max(x_sorted)
                    y_right = p(x_right)
                    x_annot = x_right - 0.03 * (max(x_sorted) - min(x_sorted))
                    y_annot = y_right + 0.05 * (max(df[y]) - min(df[y]))
                    ax.text(x_annot, y_annot, f'Slope: {slope:.3f}', color='white', ha='right', va='bottom', fontsize=10, fontweight='bold', bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2})
            else:
                ax2 = ax.twinx()
                twin_axes.append(ax2)
                if plot_overlay.plot_type == 'scatter':
                    l = ax2.scatter(x_vals, df[y], color=color, label='_nolegend_')
                else:
                    sort_idx = np.argsort(x_vals)
                    l, = ax2.plot(np.array(x_vals)[sort_idx], np.array(df[y])[sort_idx], marker='o', color=color, label='_nolegend_')
                ax2.set_ylabel(y)
                ax2.spines['right'].set_position(('outward', 60 * (i-1)))
                lines.append(l)
                labels.append(y)
                # Trendline
                if getattr(plot_overlay, 'trendline', False):
                    z = np.polyfit(x_vals, df[y], 1)
                    p = np.poly1d(z)
                    x_sorted = np.array(x_vals)[np.argsort(x_vals)]
                    l2, = ax2.plot(x_sorted, p(x_sorted), linestyle='--', color='white', alpha=0.9, label=f'{y} trendline')
                    lines.append(l2)
                    labels.append(f'{y} trendline')
                    # Annotate slope just above and to the left of the trendline at far right
                    slope = z[0]
                    x_right = max(x_sorted)
                    y_right = p(x_right)
                    x_annot = x_right - 0.03 * (max(x_sorted) - min(x_sorted))
                    y_annot = y_right + 0.05 * (max(df[y]) - min(df[y]))
                    ax2.text(x_annot, y_annot, f'Slope: {slope:.3f}', color='white', ha='right', va='bottom', fontsize=10, fontweight='bold', bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2})
                # Explicitly remove any legend from this twin axis
                # Remove any legend from this twin axis (matplotlib may auto-generate)
                if hasattr(ax2, 'legend_') and ax2.legend_ is not None:
                    ax2.legend_.remove()
        # Always set x-axis label on the main axis
        ax.set_xlabel(x_col)
        # Only show one legend with all features, but remove trendlines
        filtered = [(h, l) for h, l in zip(lines, labels) if 'trendline' not in l.lower()]
        if filtered:
            handles, legend_labels = zip(*filtered)
        else:
            handles, legend_labels = [], []
        # Draw legend above data points
        ax.legend(handles, legend_labels, title='Y-Features')
    # Single-y overlay plot logic
    if len(y_cols) == 1:
        y = y_cols[0]
        if plot_overlay.plot_type == 'scatter':
            plt.scatter(x_vals, df[y], label=y)
        else:
            sort_idx = np.argsort(x_vals)
            plt.plot(np.array(x_vals)[sort_idx], np.array(df[y])[sort_idx], marker='o', label=y)
        # Trendline
        if getattr(plot_overlay, 'trendline', False):
            z = np.polyfit(x_vals, df[y], 1)
            p = np.poly1d(z)
            x_sorted = np.array(x_vals)[np.argsort(x_vals)]
            plt.plot(x_sorted, p(x_sorted), linestyle='--', color='white', alpha=0.9, label=f'{y} trendline')
            # Annotate slope just above and to the left of the trendline at far right
            slope = z[0]
            x_right = max(x_sorted)
            y_right = p(x_right)
            x_annot = x_right - 0.03 * (max(x_sorted) - min(x_sorted))
            y_annot = y_right + 0.05 * (max(df[y]) - min(df[y]))
            plt.text(x_annot, y_annot, f'Slope: {slope:.3f}', color='white', ha='right', va='bottom', fontsize=10, fontweight='bold', bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2})
        plt.ylabel(y)
    # Annotate overlay plot with Y-feature stats if requested (draw after all plotting)
    if getattr(plot_overlay, 'annotation', False):
        stats_text = ''
        for y in y_cols:
            mean = df[y].mean()
            median = df[y].median()
            min_val = df[y].min()
            max_val = df[y].max()
            # Calculate trendline slope
            try:
                z = np.polyfit(df[x_col], df[y], 1)
                slope = z[0]
            except Exception:
                slope = float('nan')
            stats_text += f'{y}:\n  Mean: {mean:.3f}\n  Median: {median:.3f}\n  Min: {min_val:.3f}\n  Max: {max_val:.3f}\n  Slope: {slope:.3f}\n\n'
        ax = plt.gca()
        ax.text(0.01, 0.99, stats_text, transform=ax.transAxes, fontsize=11, color='white', va='top', ha='left', bbox=dict(facecolor='black', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.5'))
    plt.xlabel(f'{x_col}')
    # Remove trendlines from legend (single-y only)
    if len(y_cols) == 1:
        handles, labels = plt.gca().get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if 'trendline' not in l.lower()]
        if filtered:
            handles, labels = zip(*filtered)
            plt.legend(handles, labels, title='Y-Features')
        else:
            plt.legend(title='Y-Features')
    # Set dynamic plot title
    if len(y_cols) == 1:
        plt.title(f'{x_col} vs {y_cols[0]}')
    else:
        plt.title(f'{x_col} vs Y-Features')
    # Plot vertical lines for x_intercepts
    if x_intercepts:
        for i, x in enumerate(x_intercepts):
            name = x_names[i] if x_names and i < len(x_names) else f'x={x}'
            plt.axvline(x, color='yellow', linestyle='-', linewidth=2, zorder=10)
            # Place label at top, inside plot face, left of line
            y_top = ax.get_ylim()[1] - 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            x_left = x - 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
            plt.text(x_left, y_top, name, color='yellow', ha='right', va='top', fontsize=10, zorder=11)
    # Plot horizontal lines for y_intercepts
    if y_intercepts:
        for i, y in enumerate(y_intercepts):
            name = y_names[i] if y_names and i < len(y_names) else f'y={y}'
            plt.axhline(y, color='cyan', linestyle='-', linewidth=2, zorder=10)
            # Place label at far right, barely above line
            x_right = ax.get_xlim()[1]
            plt.text(x_right, y + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]), name, color='cyan', ha='right', va='bottom', fontsize=10, zorder=11)
    plt.tight_layout()
    plt.show()
    plt.tight_layout()

plot_overlay.plot_type = 'scatter'  # default
plot_overlay.stack = False  # default
plot_overlay.trendline = False  # default
plot_overlay.residuals = False  # default
plot_overlay.bootstrap = False  # default
    

class CustomColumnDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Create Custom Value Column')
        self.df = df
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.name_label = QLabel('Custom column name:')
        self.name_edit = QLineEdit()
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.name_edit)
        self.var_combos = {}
        colnames = list(df.select_dtypes(include=[np.number]).columns)
        for i, var in enumerate('abcdefghij'):
            hbox = QHBoxLayout()
            label = QLabel(f'Assign column to variable "{var}":')
            combo = QComboBox()
            combo.addItems([''] + colnames)
            hbox.addWidget(label)
            hbox.addWidget(combo)
            self.layout.addLayout(hbox)
            self.var_combos[var] = combo
        self.formula_label = QLabel('Formula Input (use a-j):')
        self.formula_edit = QLineEdit()
        self.layout.addWidget(self.formula_label)
        self.layout.addWidget(self.formula_edit)
        self.btn_ok = QPushButton('Create Column')
        self.btn_ok.clicked.connect(self.accept)
        self.layout.addWidget(self.btn_ok)
        # Add numpy and math function dropdown (no duplicates)
    # np and math are imported globally and available for formula evaluation
        np_funcs = [
            ('abs', 'Absolute value', 'np.abs(a)'),
            ('sqrt', 'Square root', 'np.sqrt(a)'),
            ('log', 'Natural logarithm', 'np.log(a)'),
            ('exp', 'Exponential', 'np.exp(a)'),
            ('sin', 'Sine', 'np.sin(a)'),
            ('cos', 'Cosine', 'np.cos(a)'),
            ('tan', 'Tangent', 'np.tan(a)'),
            ('arcsin', 'Inverse sine', 'np.arcsin(a)'),
            ('arccos', 'Inverse cosine', 'np.arccos(a)'),
            ('arctan', 'Inverse tangent', 'np.arctan(a)'),
            ('arctan2', 'Arctangent of y/x', 'np.arctan2(a, b)'),
            ('round', 'Round to nearest integer', 'np.round(a)'),
            ('floor', 'Floor (round down)', 'np.floor(a)'),
            ('ceil', 'Ceil (round up)', 'np.ceil(a)'),
            ('clip', 'Limit values to a range', 'np.clip(a, 0, 1)'),
        ]
        math_funcs = [
            # Only include unique math functions not duplicated in numpy
            ('pow', 'Power', 'math.pow(a, b)'),
            ('degrees', 'Radians to degrees', 'math.degrees(a)'),
            ('radians', 'Degrees to radians', 'math.radians(a)'),
            ('pi', 'Pi constant', 'math.pi'),
        ]
        self.func_label = QLabel('Available functions (numpy, math):')
        self.layout.addWidget(self.func_label)
        self.func_combo = QComboBox()
        for func, desc, example in np_funcs:
            self.func_combo.addItem(f'np.{func} - {desc} | Example: {example}')
        for func, desc, example in math_funcs:
            self.func_combo.addItem(f'math.{func} - {desc} | Example: {example}')
        self.layout.addWidget(self.func_combo)
        self.func_desc_label = QLabel('Select a function to see its name, description, and example. You can use any listed function in your formula input as np.function_name(...), math.function_name(...), or math.pi.')
        self.layout.addWidget(self.func_desc_label)

    def get_custom_column(self):
        name = self.name_edit.text().strip()
        formula = self.formula_edit.text().strip()
        var_map = {var: self.var_combos[var].currentText() for var in self.var_combos}
        # Build local dict for eval
        local_vars = {'np': np, 'math': math}
        for var, col in var_map.items():
            if col:
                local_vars[var] = self.df[col]
        try:
            result = eval(formula, {"__builtins__": None}, local_vars)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error in formula: {e}')
            return None, None
        return name, result

class DataFrameExportDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle('View DataFrame')
        self.df = df
        self.setWindowModality(False)  # Make dialog non-modal
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.df_label = QLabel('DataFrame Preview:')
        self.layout.addWidget(self.df_label)
        # Show DataFrame in a scrollable table
        table = QTableWidget(df.shape[0], df.shape[1])
        table.setHorizontalHeaderLabels([str(col) for col in df.columns])
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        self.layout.addWidget(table)
    def export_csv(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Export CSV', '', 'CSV Files (*.csv)')
        if fname:
            self.df.to_csv(fname, index=False)
            QMessageBox.information(self, 'Exported', f'DataFrame exported to {fname}')

class PairPlotApp(QWidget):
    """
    Main GUI application for interactive pair plotting and overlay visualization.
    Allows users to:
    - Load a CSV dataset
    - Optionally create a SUM column from numeric features
    - Filter the dataframe by multiple non-numeric features
    - Select features for pair plotting and overlay plotting
    """
    def __init__(self):
        """
        Initializes the GUI, sets up widgets for file loading, feature selection, and plotting.
        """
        super().__init__()
        self.setWindowTitle('Pair Plot & Overlay GUI')
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel('Select a CSV file to begin.')
        self.layout.addWidget(self.label)

        self.btn_load = QPushButton('Load CSV')
        self.btn_load.clicked.connect(self.load_csv)
        self.layout.addWidget(self.btn_load)

        self.list_x = QListWidget()
        self.list_x.setSelectionMode(QListWidget.SingleSelection)
        self.layout.addWidget(QLabel('Select X axis feature:'))
        self.layout.addWidget(self.list_x)

        self.list_y = QListWidget()
        self.list_y.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(QLabel('Select Y axis features:'))
        self.layout.addWidget(self.list_y)
        # Move radio buttons after feature selection
        self.plot_type_label = QLabel('Select Overlay Plot Type:')
        self.radio_scatter = QRadioButton('Scatter')
        self.radio_line = QRadioButton('Line with Markers')
        self.radio_scatter.setChecked(True)
        self.plot_type_group = QButtonGroup(self)
        self.plot_type_group.addButton(self.radio_scatter)
        self.plot_type_group.addButton(self.radio_line)
        self.layout.addWidget(self.plot_type_label)
        self.layout.addWidget(self.radio_scatter)
        self.layout.addWidget(self.radio_line)

        # Add stacking, trendline, residuals, bootstrap, alpha, annotation checkboxes
        self.stack_checkbox = QCheckBox('Stack Y features on Y axis')
        self.trendline_checkbox = QCheckBox('Show trendline for each Y feature')
        self.residuals_checkbox = QCheckBox('Plot residuals (absolute distance from trendline)')
        self.bootstrap_checkbox = QCheckBox('Bootstrap data (10,000 samples)')
        self.alpha_checkbox = QCheckBox('Vary Data Transparency by Data Density (Scatter Only)')
        self.annotation_checkbox = QCheckBox('Annotate Overlay Plot with Y-Axes stats')

        # Add horizontal line above Data Toggles section
        self.data_toggles_hr = QFrame()
        self.data_toggles_hr.setFrameShape(QFrame.HLine)
        self.data_toggles_hr.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.data_toggles_hr)
        self.data_toggles_label = QLabel('Select Data Toggles:')
        self.layout.addWidget(self.data_toggles_label)
        # Arrange toggles in three rows of two
        toggles_row1 = QHBoxLayout()
        toggles_row1.addWidget(self.stack_checkbox)
        toggles_row1.addWidget(self.bootstrap_checkbox)
        self.layout.addLayout(toggles_row1)
        toggles_row2 = QHBoxLayout()
        toggles_row2.addWidget(self.annotation_checkbox)
        toggles_row2.addWidget(self.residuals_checkbox)
        self.layout.addLayout(toggles_row2)
        toggles_row3 = QHBoxLayout()
        toggles_row3.addWidget(self.trendline_checkbox)
        toggles_row3.addWidget(self.alpha_checkbox)
        self.layout.addLayout(toggles_row3)

        # Add horizontal line above Event/Threshold Comparison section
        self.event_threshold_hr = QFrame()
        self.event_threshold_hr.setFrameShape(QFrame.HLine)
        self.event_threshold_hr.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.event_threshold_hr)
        self.event_threshold_label = QLabel('Enter Values for Event/Threshold Comparison:')
        self.layout.addWidget(self.event_threshold_label)
        # Add controls for vertical and horizontal lines (side-by-side)
        x_hbox = QHBoxLayout()
        x_hbox.addWidget(QLabel('X-intercepts (comma-separated):'))
        self.x_intercepts_edit = QLineEdit()
        x_hbox.addWidget(self.x_intercepts_edit)
        x_hbox.addWidget(QLabel('Names for X-intercepts (comma-separated):'))
        self.x_intercepts_names_edit = QLineEdit()
        x_hbox.addWidget(self.x_intercepts_names_edit)
        self.layout.addLayout(x_hbox)
        y_hbox = QHBoxLayout()
        y_hbox.addWidget(QLabel('Y-intercepts (comma-separated):'))
        self.y_intercepts_edit = QLineEdit()
        y_hbox.addWidget(self.y_intercepts_edit)
        y_hbox.addWidget(QLabel('Names for Y-intercepts (comma-separated):'))
        self.y_intercepts_names_edit = QLineEdit()
        y_hbox.addWidget(self.y_intercepts_names_edit)
        self.layout.addLayout(y_hbox)

        self.btn_overlay = QPushButton('Show Overlay Plot')
        self.btn_overlay.clicked.connect(self.show_overlay)
        self.layout.addWidget(self.btn_overlay)

        self.export_btn = QPushButton('Export DataFrame as CSV')
        self.export_btn.clicked.connect(self.export_csv)
        self.layout.addWidget(self.export_btn)
        self.layout.addStretch(1)  # Push export button to bottom

        self.df = None

    def load_csv(self):
        """
        Loads a CSV file, optionally creates a SUM column, allows filtering by non-numeric features,
    plot_type = 'Scatter' if self.radio_scatter.isChecked() else 'Line with Markers'
    plot_overlay(self.df, x_col, y_cols, plot_type)
        """
        fname, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '', 'CSV Files (*.csv)')
        if fname:
            try:
                df = pd.read_csv(fname)
                df = df[1:]
                for col in df.columns:
                    if col == 'timestamp':
                        df[col] = datetime.datetime(year=1970, month=1, day=1) + pd.to_timedelta(df[col], unit='s')
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox, QHBoxLayout, QLineEdit
                reply = QMessageBox.question(self, 'Create SUM Column?', 'Would you like to create a SUM column from numeric features?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes and len(numeric_cols) > 1:
                    class SumDialog(QDialog):
                        def __init__(self, columns, parent=None):
                            super().__init__(parent)
                            self.setWindowTitle('Select Features to SUM')
                            self.layout = QVBoxLayout(self)
                            self.checkboxes = []
                            self.layout.addWidget(QLabel('Select features to SUM:'))
                            for col in columns:
                                cb = QCheckBox(col)
                                self.layout.addWidget(cb)
                                self.checkboxes.append(cb)
                            self.layout.addWidget(QLabel('Name for SUM feature:'))
                            self.name_edit = QLineEdit()
                            self.layout.addWidget(self.name_edit)
                            self.btn_ok = QPushButton('OK')
                            self.btn_ok.clicked.connect(self.accept)
                            self.layout.addWidget(self.btn_ok)
                        def get_sum_features(self):
                            features = [cb.text() for cb in self.checkboxes if cb.isChecked()]
                            name = self.name_edit.text().strip()
                            return features, name
                    sum_dialog = SumDialog(numeric_cols, self)
                    if sum_dialog.exec_() == QDialog.Accepted:
                        sum_features, sum_name = sum_dialog.get_sum_features()
                        if sum_features and sum_name:
                            df[sum_name] = df[sum_features].sum(axis=1)
                # Prompt for custom column after SUM column dialog
                reply_custom = QMessageBox.question(self, 'Create Custom Column?', 'Would you like to create a column with custom values?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply_custom == QMessageBox.Yes:
                    custom_dialog = CustomColumnDialog(df, self)
                    if custom_dialog.exec_() == QDialog.Accepted:
                        col_name, col_vals = custom_dialog.get_custom_column()
                        if col_name and col_vals is not None:
                            df[col_name] = col_vals
                # Identify non-numeric columns
                non_numeric_cols = []
                for col in df.columns:
                    try:
                        df[col].astype(float)
                    except Exception:
                        non_numeric_cols.append(col)
                # If there are non-numeric columns, prompt user before showing filter GUI
                class MultiFilterDialog(QDialog):
                    def __init__(self, columns, df, parent=None):
                        super().__init__(parent)
                        self.setWindowTitle('Advanced Filter by Non-numeric Features')
                        self.layout = QVBoxLayout(self)
                        self.filters = []
                        self.df = df
                        self.columns = columns
                        self.layout.addWidget(QLabel('Toggle up to 20 filters. For each filter, select a column, then a value, then AND/OR logic.'))
                        header_hbox = QHBoxLayout()
                        header_hbox.addWidget(QLabel('Filter Toggle'))
                        header_hbox.addWidget(QLabel('Column to Filter By'))
                        header_hbox.addWidget(QLabel('Value to Filter By'))
                        header_hbox.addWidget(QLabel('Combining Logic (AND/OR)'))
                        self.layout.addLayout(header_hbox)
                        for i in range(20):
                            hbox = QHBoxLayout()
                            cb = QCheckBox(f'Filter {i+1}')
                            hbox.addWidget(cb)
                            col_combo = QComboBox()
                            col_combo.addItems([''] + columns)
                            hbox.addWidget(col_combo)
                            val_combo = QComboBox()
                            hbox.addWidget(val_combo)
                            logic_combo = QComboBox()
                            logic_combo.addItems(['', 'AND', 'OR'])
                            hbox.addWidget(logic_combo)
                            self.filters.append((cb, col_combo, val_combo, logic_combo))
                            self.layout.addLayout(hbox)
                            # Connect column dropdown to update value dropdown for this filter only
                            def make_update_func(val_combo, col_combo):
                                def update_values():
                                    col = col_combo.currentText()
                                    val_combo.clear()
                                    if col and col in df.columns:
                                        val_combo.addItems([str(v) for v in df[col].unique()])
                                return update_values
                            col_combo.currentTextChanged.connect(make_update_func(val_combo, col_combo))
                        self.btn_ok = QPushButton('OK')
                        self.btn_ok.clicked.connect(self.accept)
                        self.layout.addWidget(self.btn_ok)
                    def get_filters(self):
                        filters = []
                        logic_ops = []
                        for i, (cb, col_combo, val_combo, logic_combo) in enumerate(self.filters):
                            if cb.isChecked():
                                col = col_combo.currentText()
                                val = val_combo.currentText()
                                if col and val:
                                    filters.append((col, val))
                                    logic_ops.append(logic_combo.currentText())
                        return filters, logic_ops
                if non_numeric_cols:
                    reply_filter = QMessageBox.question(self, 'Filter by Non-numeric Columns?', 'Would you like to filter by non-numeric columns?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply_filter == QMessageBox.Yes:
                        # Show dialog for multi-filter selection
                        mf_dialog = MultiFilterDialog(non_numeric_cols, df, self)
                        if mf_dialog.exec_() == QDialog.Accepted:
                            filters, logic_ops = mf_dialog.get_filters()
                            # Build query string
                            query = ''
                            for i, (col, val) in enumerate(filters):
                                cond = f'`{col}` == @val'
                                if i == 0:
                                    query += cond
                                else:
                                    op = logic_ops[i-1].lower() if i-1 < len(logic_ops) else 'and'
                                    if op == 'and':
                                        query += f' & {cond}'
                                    else:
                                        query += f' | {cond}'
                            if query:
                                df = df.query(query)
                # Now clean and plot
                df = clean_numeric_df(df)
                if df.shape[1] < 2:
                    QMessageBox.warning(self, 'Error', 'Not enough numeric columns for plotting.')
                    return
                self.df = df
                # Prompt for thresholds before pair plot
                thresholds = None
                reply_thresholds = QMessageBox.question(self, 'Load Thresholds?', 'Would you like to load thresholds for any feature?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply_thresholds == QMessageBox.Yes:
                    fname_thresh, _ = QFileDialog.getOpenFileName(self, 'Open Thresholds CSV', '', 'CSV Files (*.csv)')
                    if fname_thresh:
                        try:
                            df_thresh = pd.read_csv(fname_thresh)
                            if not df_thresh.empty:
                                last_row = df_thresh.iloc[-1]
                                thresholds = {col: last_row[col] for col in df_thresh.columns}
                        except Exception as e:
                            QMessageBox.warning(self, 'Error', f'Failed to load thresholds CSV: {e}')
                # Prompt for feature highlight before pair plot
                highlight_feature = None
                reply_highlight = QMessageBox.question(self, 'Highlight Feature?', 'Would you like to highlight a feature in the pair plot?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply_highlight == QMessageBox.Yes:
                    # Only show numeric columns
                    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
                    class HighlightDialog(QDialog):
                        def __init__(self, columns, parent=None):
                            super().__init__(parent)
                            self.setWindowTitle('Select Feature to Highlight')
                            self.layout = QVBoxLayout(self)
                            self.layout.addWidget(QLabel('Select a feature to highlight:'))
                            self.combo = QComboBox()
                            self.combo.addItems(columns)
                            self.layout.addWidget(self.combo)
                            self.btn_ok = QPushButton('OK')
                            self.btn_ok.clicked.connect(self.accept)
                            self.layout.addWidget(self.btn_ok)
                        def get_feature(self):
                            return self.combo.currentText()
                    hd = HighlightDialog(numeric_cols, self)
                    if hd.exec_() == QDialog.Accepted:
                        highlight_feature = hd.get_feature()
                plot_pairgrid(df, highlight_feature=highlight_feature, thresholds=thresholds)
                # Show DataFrame pop-up after pair plot (non-blocking)
                self.df_dialog = DataFrameExportDialog(df, self)
                self.df_dialog.show()
                self.list_x.clear()
                self.list_y.clear()
                for col in df.columns:
                    self.list_x.addItem(QListWidgetItem(col))
                    self.list_y.addItem(QListWidgetItem(col))
                self.label.setText(f'Loaded: {fname}')
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to load CSV: {e}')

    def show_overlay(self):
        """
        Generates the overlay plot for selected features using plot_overlay().
        Ensures valid selections and removes x_col from y_cols if present.
        Uses radio button to select plot type (scatter or line+marker).
        Uses checkbox to select stacking mode for Y features.
        Uses checkbox to select trendline mode for Y features.
        Uses checkbox to select residuals mode for Y features.
        Uses checkbox to select bootstrap mode for Y features.
        Passes x/y intercepts and names to plot_overlay.
        """
        if self.df is None:
            QMessageBox.warning(self, 'Error', 'No data loaded.')
            return
        x_items = self.list_x.selectedItems()
        y_items = self.list_y.selectedItems()
        if not x_items or not y_items:
            QMessageBox.warning(self, 'Error', 'Select one X and one or more Y features.')
            return
        x_col = x_items[0].text()
        y_cols = [item.text() for item in y_items][:5]
        if x_col in y_cols:
            y_cols.remove(x_col)
        # Set plot type based on radio button
        if self.radio_scatter.isChecked():
            plot_overlay.plot_type = 'scatter'
        else:
            plot_overlay.plot_type = 'line'
        # Set stacking mode based on checkbox
        plot_overlay.stack = self.stack_checkbox.isChecked()
        # Set trendline mode based on checkbox
        plot_overlay.trendline = self.trendline_checkbox.isChecked()
        # Set residuals mode based on checkbox
        plot_overlay.residuals = self.residuals_checkbox.isChecked()
        # Set bootstrap mode based on checkbox
        plot_overlay.bootstrap = self.bootstrap_checkbox.isChecked()
        # Set annotation mode based on checkbox
        plot_overlay.annotation = self.annotation_checkbox.isChecked()
        # Parse x/y intercepts and names
        x_intercepts = [float(x.strip()) for x in self.x_intercepts_edit.text().split(',') if x.strip()] if self.x_intercepts_edit.text().strip() else None
        x_names = [n.strip() for n in self.x_intercepts_names_edit.text().split(',') if n.strip()] if self.x_intercepts_names_edit.text().strip() else None
        y_intercepts = [float(y.strip()) for y in self.y_intercepts_edit.text().split(',') if y.strip()] if self.y_intercepts_edit.text().strip() else None
        y_names = [n.strip() for n in self.y_intercepts_names_edit.text().split(',') if n.strip()] if self.y_intercepts_names_edit.text().strip() else None
        plot_overlay(self.df, x_col, y_cols, x_intercepts=x_intercepts, x_names=x_names, y_intercepts=y_intercepts, y_names=y_names)
        # Add Export CSV button to overlay plot GUI
        # No need to add export_btn here; it's always present at the bottom

    def export_csv(self):
        """
        Exports the current DataFrame to a CSV file.
        """
        if self.df is None:
            QMessageBox.warning(self, 'Error', 'No data to export.')
            return
        fname, _ = QFileDialog.getSaveFileName(self, 'Export CSV', '', 'CSV Files (*.csv)')
        if fname:
            self.df.to_csv(fname, index=False)
            QMessageBox.information(self, 'Exported', f'DataFrame exported to {fname}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PairPlotApp()
    window.show()
    sys.exit(app.exec_())