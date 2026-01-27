import sys
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    - Preserves datetime columns even though they're not numeric.
    - Returns a DataFrame containing numeric and datetime columns, with rows containing NaNs dropped.
    This ensures the data is ready for numeric analysis and plotting.
    """
    df = df.replace('?', np.nan)
    # Track datetime columns before conversion attempts
    datetime_cols = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols[col] = df[col].copy()
    
    for col in df.columns:
        # Skip datetime columns from float conversion
        if col in datetime_cols:
            continue
        try:
            df[col] = df[col].astype(float)
        except Exception:
            df = df.drop(col, axis=1)
    
    # Get numeric columns
    df_clean = df.select_dtypes(include=[np.number]).dropna()
    
    # Add datetime columns back at the beginning
    for col, values in datetime_cols.items():
        if col in df.columns:  # Only add if column wasn't dropped
            # Insert at the beginning of the dataframe
            df_clean.insert(0, col, values.loc[df_clean.index])
    
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
                        thresh_val = thresholds[x_var]
                        if isinstance(thresh_val, list):
                            # Plot min and max lines
                            ax.axvline(thresh_val[0], color='orange', linestyle='--', linewidth=2, label=f'{x_var} min threshold')
                            ax.axvline(thresh_val[1], color='red', linestyle='--', linewidth=2, label=f'{x_var} max threshold')
                        else:
                            # Plot single threshold line
                            ax.axvline(thresh_val, color='orange', linestyle='--', linewidth=2, label=f'{x_var} threshold')
                    # If y_var has a threshold, plot horizontal line
                    if y_var in thresholds:
                        thresh_val = thresholds[y_var]
                        if isinstance(thresh_val, list):
                            # Plot min and max lines
                            ax.axhline(thresh_val[0], color='orange', linestyle='--', linewidth=2, label=f'{y_var} min threshold')
                            ax.axhline(thresh_val[1], color='red', linestyle='--', linewidth=2, label=f'{y_var} max threshold')
                        else:
                            # Plot single threshold line
                            ax.axhline(thresh_val, color='orange', linestyle='--', linewidth=2, label=f'{y_var} threshold')
    g.map_upper(annotate_corr)
    plt.suptitle('DOOIT!', y=1.08, fontsize=26,color='black')
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('DOOIT! Pair Plot')
    plt.show()
    plt.tight_layout()

def plot_overlay(df, x_col, y_cols, x_intercepts=None, x_names=None, y_intercepts=None, y_names=None, df_comparison=None):
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
    - If df_comparison is provided and plot_overlay.comparison is True, plots comparison data with "+" prefix in legend.
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
        # For each feature (X and Y), apply random coefficient from adjacent points in original data
        all_cols = [x_col] + y_cols
        for col in all_cols:
            coefficients = []
            for idx in boot_idx:
                # Get random offset above (1 to 10 indexes)
                offset_above = np.random.randint(1, 11)
                # Get random offset below (1 to 10 indexes)
                offset_below = np.random.randint(1, 11)
                
                # Find the position of this index in the dataframe
                try:
                    pos = df.index.get_loc(idx)
                except KeyError:
                    # If index not found, use first position
                    pos = 0
                
                # Calculate positions with bounds checking
                pos_above = min(pos + offset_above, len(df) - 1)
                pos_below = max(pos - offset_below, 0)
                
                # Get values from original dataframe using positional indexing
                val_above = df.iloc[pos_above][col]
                val_below = df.iloc[pos_below][col]
                
                # Calculate coefficient (average above / average below)
                # Use mean of the value above, and avoid division by zero
                if val_below != 0:
                    coef = val_above / val_below
                else:
                    coef = 1.0  # Default coefficient if division by zero
                coefficients.append(coef)
            
            # Apply coefficients to bootstrapped values
            df_boot[col] = df_boot[col] * coefficients
        
        # Use bootstrapped dataframe for plotting
        df = df_boot
    # Calculate alpha based on dataset size if alpha mode is enabled
    if getattr(plot_overlay, 'alpha', False):
        # Scale alpha inversely with dataset size, but keep it visible (min 0.01, max 1.0)
        alpha_value = max(0.01, min(1.0, 100.0 / len(df)))
    else:
        alpha_value = 1.0
    
    # Pre-calculate y-limits if comparison mode is enabled (for all plot modes)
    y_limits = {}
    twin_axes = []  # Initialize for all plot modes, will be populated by multi-y mode
    aligned_ranges = []  # Initialize for residuals mode with twin axes
    if getattr(plot_overlay, 'comparison', False) and df_comparison is not None:
        for y in y_cols:
            if y in df.columns and y in df_comparison.columns:
                combined_min = min(df[y].min(), df_comparison[y].min())
                combined_max = max(df[y].max(), df_comparison[y].max())
                padding = (combined_max - combined_min) * 0.05
                y_limits[y] = (combined_min - padding, combined_max + padding)
    
    x_vals = df[x_col]
    if getattr(plot_overlay, 'residuals', False) and getattr(plot_overlay, 'stack', False):
        # Plot stacked residuals: calculate residuals first, then normalize and stack
        from sklearn.preprocessing import MinMaxScaler
        residuals_dict = {}
        # Calculate residuals for each Y feature
        for y in y_cols:
            y_vals = df[y]
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            residuals_dict[y] = y_vals - p(x_vals)
        # Normalize residuals to [0, 1] and stack them
        residuals_df = pd.DataFrame(residuals_dict)
        scaler = MinMaxScaler()
        residuals_norm = pd.DataFrame(scaler.fit_transform(residuals_df), columns=y_cols)
        offset = 0
        for i, y in enumerate(y_cols):
            vals = residuals_norm[y] + offset
            # Use brighter colors
            bright_colors = ['#00D4FF', '#FF6B35', '#00FF88', '#FF3366', '#FFDD00', '#BB77FF', '#00FFDD']
            color = bright_colors[i % len(bright_colors)]
            if plot_overlay.plot_type == 'scatter':
                plt.scatter(x_vals, vals, label=f'{y} residuals (stack {i+1})', color=color, alpha=alpha_value)
            else:
                sort_idx = np.argsort(x_vals)
                plt.plot(np.array(x_vals)[sort_idx], np.array(vals)[sort_idx], marker='o', label=f'{y} residuals (stack {i+1})', color=color)
            # Trendline slope annotation for stacked residuals (far right)
            if getattr(plot_overlay, 'trendline', False):
                z = np.polyfit(x_vals, vals, 1)
                p = np.poly1d(z)
                x_sorted = np.array(x_vals)[np.argsort(x_vals)]
                plt.plot(x_sorted, p(x_sorted), linestyle='--', color='white', alpha=0.9, label=f'{y} trendline')
                slope = z[0]
                x_right = max(x_sorted)
                y_right = p(x_right)
                x_annot = x_right - 0.03 * (max(x_sorted) - min(x_sorted))
                y_annot = y_right + 0.05 * (max(vals) - min(vals))
                plt.text(x_annot, y_annot, f'Slope: {slope:.3f}', color='white', ha='right', va='bottom', fontsize=10, fontweight='bold', bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2})
            offset += 1
        plt.ylabel('Stacked Normalized Residuals')
        # Remove y-axis labels and add horizontal lines at integer values
        ax = plt.gca()
        ax.set_yticks([])  # Remove y-axis tick labels
        # Add horizontal lines at each integer to differentiate stacked features
        for i in range(len(y_cols)):
            ax.axhline(y=i, color='white', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    elif getattr(plot_overlay, 'residuals', False):
        # Plot residuals for each Y-feature using twin axes for multiple features
        if len(y_cols) > 1:
            # Use twin axes for multiple residual features (like multi-y mode)
            ax = plt.gca()
            bright_colors = ['#00D4FF', '#FF6B35', '#00FF88', '#FF3366', '#FFDD00', '#BB77FF', '#00FFDD']
            lines = []
            labels = []
            twin_axes = [ax]
            
            # First pass: calculate all residuals (BOTH original and comparison together)
            all_residuals = []
            all_residuals_comp = []
            
            for y in y_cols:
                y_vals = df[y]
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                residuals = y_vals - p(x_vals)
                all_residuals.append(residuals)
            
            # Calculate comparison residuals if comparison mode is enabled
            if getattr(plot_overlay, 'comparison', False) and df_comparison is not None and x_col in df_comparison.columns:
                for y in y_cols:
                    if y in df_comparison.columns:
                        y_vals_comp = df_comparison[y]
                        x_vals_comp = df_comparison[x_col]
                        z_comp = np.polyfit(x_vals_comp, y_vals_comp, 1)
                        p_comp = np.poly1d(z_comp)
                        residuals_comp = y_vals_comp - p_comp(x_vals_comp)
                        all_residuals_comp.append(residuals_comp)
                    else:
                        all_residuals_comp.append(None)
            
            # Calculate ranges INCLUDING both original and comparison data from the start
            residual_ranges = []
            for i, residuals in enumerate(all_residuals):
                res_min = residuals.min()
                res_max = residuals.max()
                
                # Include comparison data in the INITIAL range calculation
                if all_residuals_comp and i < len(all_residuals_comp) and all_residuals_comp[i] is not None:
                    res_min = min(res_min, all_residuals_comp[i].min())
                    res_max = max(res_max, all_residuals_comp[i].max())
                
                # Expand range to include zero
                res_min = min(res_min, 0)
                res_max = max(res_max, 0)
                # Add 15% padding
                padding = (res_max - res_min) * 0.15
                res_min -= padding
                res_max += padding
                residual_ranges.append((res_min, res_max))
            
            # Find the zero position ratio from the first axis
            y0_min, y0_max = residual_ranges[0]
            zero_ratio = (0 - y0_min) / (y0_max - y0_min) if (y0_max - y0_min) != 0 else 0.5
            
            # Adjust all other axes to have the same zero ratio
            # For each axis, we need: (0 - new_min) / (new_max - new_min) = zero_ratio
            # This means: -new_min = zero_ratio * (new_max - new_min)
            # Therefore: new_min = -zero_ratio * new_range and new_max = (1 - zero_ratio) * new_range
            aligned_ranges = [residual_ranges[0]]
            for i in range(1, len(residual_ranges)):
                res_min, res_max = residual_ranges[i]
                
                # Calculate the minimum range needed to fit the data
                # Data must fit: res_min <= new_min and res_max <= new_max
                # where new_min = -zero_ratio * R and new_max = (1 - zero_ratio) * R
                
                # From res_min <= -zero_ratio * R, we get: R >= -res_min / zero_ratio (if zero_ratio > 0)
                # From res_max <= (1 - zero_ratio) * R, we get: R >= res_max / (1 - zero_ratio) (if zero_ratio < 1)
                
                if zero_ratio > 0 and zero_ratio < 1:
                    # Calculate required range from below (negative values)
                    if res_min < 0:
                        required_range_neg = -res_min / zero_ratio
                    else:
                        required_range_neg = 0
                    
                    # Calculate required range from above (positive values)
                    if res_max > 0:
                        required_range_pos = res_max / (1 - zero_ratio)
                    else:
                        required_range_pos = 0
                    
                    # Use the larger required range
                    new_range = max(required_range_neg, required_range_pos, res_max - res_min)
                    new_min = -zero_ratio * new_range
                    new_max = (1 - zero_ratio) * new_range
                elif zero_ratio <= 0:
                    # Zero is below the visible range
                    new_min = res_min
                    new_max = res_max
                else:  # zero_ratio >= 1
                    # Zero is above the visible range
                    new_min = res_min
                    new_max = res_max
                
                aligned_ranges.append((new_min, new_max))
            
            # Second pass: plot with aligned axes
            for i, y in enumerate(y_cols):
                residuals = all_residuals[i]
                color = bright_colors[i % len(bright_colors)]
                
                if i == 0:
                    if plot_overlay.plot_type == 'scatter':
                        l = ax.scatter(x_vals, residuals, color=color, label=f'{y} residuals', alpha=alpha_value)
                    else:
                        sort_idx = np.argsort(x_vals)
                        l, = ax.plot(np.array(x_vals)[sort_idx], np.array(residuals)[sort_idx], marker='o', color=color, label=f'{y} residuals')
                    ax.set_ylabel(f'{y} residuals')
                    ax.set_ylim(aligned_ranges[0])
                    ax.autoscale(enable=False, axis='y')
                    lines.append(l)
                    labels.append(f'{y} residuals')
                else:
                    ax2 = ax.twinx()
                    twin_axes.append(ax2)
                    if plot_overlay.plot_type == 'scatter':
                        l = ax2.scatter(x_vals, residuals, color=color, label=f'{y} residuals', alpha=alpha_value)
                    else:
                        sort_idx = np.argsort(x_vals)
                        l, = ax2.plot(np.array(x_vals)[sort_idx], np.array(residuals)[sort_idx], marker='o', color=color, label=f'{y} residuals')
                    ax2.set_ylabel(f'{y} residuals')
                    ax2.set_ylim(aligned_ranges[i])
                    ax2.autoscale(enable=False, axis='y')
                    ax2.spines['right'].set_position(('outward', 60 * (i-1)))
                    lines.append(l)
                    labels.append(f'{y} residuals')
                    # Remove any legend from twin axis
                    if hasattr(ax2, 'legend_') and ax2.legend_ is not None:
                        ax2.legend_.remove()
                
                # Trendline for residuals
                if getattr(plot_overlay, 'trendline', False):
                    z_trend = np.polyfit(x_vals, residuals, 1)
                    p_trend = np.poly1d(z_trend)
                    x_sorted = np.array(x_vals)[np.argsort(x_vals)]
                    if i == 0:
                        l2, = ax.plot(x_sorted, p_trend(x_sorted), linestyle='--', color='white', alpha=0.9, label=f'{y} trendline')
                    else:
                        l2, = twin_axes[i].plot(x_sorted, p_trend(x_sorted), linestyle='--', color='white', alpha=0.9, label=f'{y} trendline')
                    lines.append(l2)
                    labels.append(f'{y} trendline')
                    # Trendline slope annotation
                    slope = z_trend[0]
                    residuals_sorted = np.array(residuals)[np.argsort(x_vals)]
                    x_right = max(x_sorted)
                    y_right = residuals_sorted[-1]
                    x_annot = x_right - 0.03 * (max(x_sorted) - min(x_sorted))
                    y_annot = y_right + 0.05 * (max(residuals_sorted) - min(residuals_sorted))
                    if i == 0:
                        ax.text(x_annot, y_annot, f'Slope: {slope:.3f}', color='white', ha='right', va='bottom', fontsize=10, fontweight='bold', bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2})
                    else:
                        twin_axes[i].text(x_annot, y_annot, f'Slope: {slope:.3f}', color='white', ha='right', va='bottom', fontsize=10, fontweight='bold', bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2})
            ax.set_xlabel(x_col)
        else:
            # Single Y feature residuals (original simple plotting)
            y = y_cols[0]
            y_vals = df[y]
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            residuals = y_vals - p(x_vals)
            bright_colors = ['#00D4FF', '#FF6B35', '#00FF88', '#FF3366', '#FFDD00', '#BB77FF', '#00FFDD']
            color = bright_colors[0]
            if plot_overlay.plot_type == 'scatter':
                plt.scatter(x_vals, residuals, label=f'{y} residuals', color=color, alpha=alpha_value)
            else:
                sort_idx = np.argsort(x_vals)
                plt.plot(np.array(x_vals)[sort_idx], np.array(residuals)[sort_idx], marker='o', label=f'{y} residuals', color=color)
            # Trendline for residuals
            if getattr(plot_overlay, 'trendline', False):
                z_trend = np.polyfit(x_vals, residuals, 1)
                p_trend = np.poly1d(z_trend)
                x_sorted = np.array(x_vals)[np.argsort(x_vals)]
                plt.plot(x_sorted, p_trend(x_sorted), linestyle='--', color='white', alpha=0.9, label=f'{y} trendline')
                slope = z_trend[0]
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
        # Reverse the order so first feature is on top
        offset = len(y_cols) - 1
        for i, y in enumerate(y_cols):
            vals = y_norm[y] + offset
            # Use brighter colors
            bright_colors = ['#00D4FF', '#FF6B35', '#00FF88', '#FF3366', '#FFDD00', '#BB77FF', '#00FFDD']
            color = bright_colors[i % len(bright_colors)]
            if plot_overlay.plot_type == 'scatter':
                plt.scatter(x_vals, vals, label=f'{y} (stack {len(y_cols)-i})', color=color, alpha=alpha_value)
            else:
                sort_idx = np.argsort(x_vals)
                plt.plot(np.array(x_vals)[sort_idx], np.array(vals)[sort_idx], marker='o', label=f'{y} (stack {len(y_cols)-i})', color=color)
            # Trendline
            if getattr(plot_overlay, 'trendline', False):
                z = np.polyfit(x_vals, vals, 1)
                p = np.poly1d(z)
                x_sorted = np.array(x_vals)[np.argsort(x_vals)]
                plt.plot(x_sorted, p(x_sorted), linestyle='--', color='white', alpha=0.9, label=f'{y} trendline')
                # Annotate slope of the original data (not normalized/stacked)
                z_original = np.polyfit(x_vals, df[y], 1)
                slope = z_original[0]
                x_right = max(x_sorted)
                y_right = p(x_right)
                x_annot = x_right - 0.03 * (max(x_sorted) - min(x_sorted))
                y_annot = y_right + 0.05 * (max(vals) - min(vals))
                plt.text(x_annot, y_annot, f'Slope: {slope:.3f}', color='white', ha='right', va='bottom', fontsize=10, fontweight='bold', bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2})
            offset -= 1
        plt.ylabel('Stacked Normalized Y Features')
        # Remove y-axis labels and add horizontal lines at integer values
        ax = plt.gca()
        ax.set_yticks([])  # Remove y-axis tick labels
        # Add horizontal lines at each integer to differentiate stacked features
        for i in range(len(y_cols)):
            ax.axhline(y=i, color='white', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    elif len(y_cols) > 1:
        # Dual y-axes for multiple features
        ax = plt.gca()
        # Use brighter, more vibrant colors
        colors = ['#00D4FF', '#FF6B35', '#00FF88', '#FF3366', '#FFDD00', '#BB77FF', '#00FFDD']
        
        lines = []
        labels = []
        twin_axes = [ax]
        
        # Apply y-limits to main axis BEFORE plotting if available
        if len(y_cols) > 0 and y_cols[0] in y_limits:
            ax.set_ylim(y_limits[y_cols[0]])
            ax.autoscale(enable=False, axis='y')  # Disable autoscaling
        
        for i, y in enumerate(y_cols):
            color = colors[i % len(colors)]
            if i == 0:
                if plot_overlay.plot_type == 'scatter':
                    l = ax.scatter(x_vals, df[y], color=color, label=y, alpha=alpha_value)
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
                # Apply pre-calculated y-limits BEFORE plotting if available
                if y in y_limits:
                    ax2.set_ylim(y_limits[y])
                    ax2.autoscale(enable=False, axis='y')  # Disable autoscaling
                if plot_overlay.plot_type == 'scatter':
                    l = ax2.scatter(x_vals, df[y], color=color, label=y, alpha=alpha_value)
                else:
                    sort_idx = np.argsort(x_vals)
                    l, = ax2.plot(np.array(x_vals)[sort_idx], np.array(df[y])[sort_idx], marker='o', color=color, label=y)
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
        # Store lines and labels for legend creation later (after comparison data is added)
        if 'lines' not in dir():
            lines = []
            labels = []
    # Single-y overlay plot logic
    elif len(y_cols) == 1:
        y = y_cols[0]
        # Use first bright color
        bright_colors = ['#00D4FF', '#FF6B35', '#00FF88', '#FF3366', '#FFDD00', '#BB77FF', '#00FFDD']
        color = bright_colors[0]
        if plot_overlay.plot_type == 'scatter':
            plt.scatter(x_vals, df[y], label=y, color=color, alpha=alpha_value)
        else:
            sort_idx = np.argsort(x_vals)
            plt.plot(np.array(x_vals)[sort_idx], np.array(df[y])[sort_idx], marker='o', label=y, color=color)
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
    
    # Plot comparison dataset if enabled
    if getattr(plot_overlay, 'comparison', False) and df_comparison is not None:
        # Check if required columns exist in comparison dataset
        if x_col not in df_comparison.columns:
            print(f"Warning: {x_col} not found in comparison dataset. Skipping comparison plot.")
        else:
            # Filter y_cols to only those that exist in comparison dataset
            y_cols_comp = [y for y in y_cols if y in df_comparison.columns]
            if not y_cols_comp:
                print(f"Warning: None of the selected Y columns found in comparison dataset. Skipping comparison plot.")
            else:
                df_comp = df_comparison.copy()
                x_vals_comp = df_comp[x_col]
                
                # Calculate alpha for comparison dataset
                if getattr(plot_overlay, 'alpha', False):
                    alpha_value_comp = max(0.01, min(1.0, 100.0 / len(df_comp)))
                else:
                    alpha_value_comp = 1.0
                
                # Bright colors for comparison (same palette)
                bright_colors = ['#00D4FF', '#FF6B35', '#00FF88', '#FF3366', '#FFDD00', '#BB77FF', '#00FFDD']
                
                # Handle different plot modes
                if getattr(plot_overlay, 'residuals', False) and getattr(plot_overlay, 'stack', False):
                    # Stacked residuals comparison
                    from sklearn.preprocessing import MinMaxScaler
                    residuals_dict_comp = {}
                    for y in y_cols_comp:
                        y_vals = df_comp[y]
                        z = np.polyfit(x_vals_comp, y_vals, 1)
                        p = np.poly1d(z)
                        residuals_dict_comp[y] = y_vals - p(x_vals_comp)
                    residuals_df_comp = pd.DataFrame(residuals_dict_comp)
                    scaler_comp = MinMaxScaler()
                    residuals_norm_comp = pd.DataFrame(scaler_comp.fit_transform(residuals_df_comp), columns=y_cols_comp)
                    offset = 0
                    for i, y in enumerate(y_cols_comp):
                        vals = residuals_norm_comp[y] + offset
                        color = bright_colors[i % len(bright_colors)]
                        if plot_overlay.plot_type == 'scatter':
                            plt.scatter(x_vals_comp, vals, label=f'+{y} residuals (stack {i+1})', color=color, alpha=alpha_value_comp, marker='x', s=30)
                        else:
                            sort_idx = np.argsort(x_vals_comp)
                            plt.plot(np.array(x_vals_comp)[sort_idx], np.array(vals)[sort_idx], marker='x', label=f'+{y} residuals (stack {i+1})', color=color, linestyle=':')
                        offset += 1
                elif getattr(plot_overlay, 'residuals', False):
                    # Regular residuals comparison - plot on the same axes as original residuals
                    for i, y in enumerate(y_cols_comp):
                        # Only plot if this feature exists in both datasets
                        if y not in y_cols:
                            continue
                        # Find the index in original y_cols to use same axis and color
                        orig_idx = y_cols.index(y)
                        color = bright_colors[orig_idx % len(bright_colors)]
                        
                        # Calculate comparison residuals
                        y_vals = df_comp[y]
                        z = np.polyfit(x_vals_comp, y_vals, 1)
                        p = np.poly1d(z)
                        residuals = y_vals - p(x_vals_comp)
                        
                        # Use the corresponding twin axis (for multiple features) or main axis (for single feature)
                        if len(y_cols) > 1 and 'twin_axes' in locals() and orig_idx < len(twin_axes):
                            ax_comp = twin_axes[orig_idx]
                            if plot_overlay.plot_type == 'scatter':
                                l = ax_comp.scatter(x_vals_comp, residuals, color=color, label=f'+{y} residuals', alpha=alpha_value_comp, marker='x', s=30)
                            else:
                                sort_idx = np.argsort(x_vals_comp)
                                l, = ax_comp.plot(np.array(x_vals_comp)[sort_idx], np.array(residuals)[sort_idx], marker='x', color=color, label=f'+{y} residuals', linestyle=':')
                            lines.append(l)
                            labels.append(f'+{y} residuals')
                        else:
                            # Single feature residuals - use main axis
                            if plot_overlay.plot_type == 'scatter':
                                plt.scatter(x_vals_comp, residuals, label=f'+{y} residuals', color=color, alpha=alpha_value_comp, marker='x', s=30)
                            else:
                                sort_idx = np.argsort(x_vals_comp)
                                plt.plot(np.array(x_vals_comp)[sort_idx], np.array(residuals)[sort_idx], marker='x', label=f'+{y} residuals', color=color, linestyle=':')
                    
                    # CRITICAL FIX: After all comparison residuals are plotted, re-enforce limits on ALL axes
                    # This prevents matplotlib from auto-rescaling any axis
                    if len(y_cols) > 1 and 'twin_axes' in locals() and aligned_ranges:
                        for idx, ax_to_fix in enumerate(twin_axes):
                            if idx < len(aligned_ranges):
                                ax_to_fix.set_ylim(aligned_ranges[idx])
                                ax_to_fix.autoscale(enable=False, axis='y')
                
                elif getattr(plot_overlay, 'stack', False):
                    # Stacked comparison
                    from sklearn.preprocessing import MinMaxScaler
                    scaler_comp = MinMaxScaler()
                    y_norm_comp = pd.DataFrame(scaler_comp.fit_transform(df_comp[y_cols_comp]), columns=y_cols_comp)
                    # Reverse the order so first feature is on top
                    offset = len(y_cols_comp) - 1
                    for i, y in enumerate(y_cols_comp):
                        vals = y_norm_comp[y] + offset
                        color = bright_colors[i % len(bright_colors)]
                        if plot_overlay.plot_type == 'scatter':
                            plt.scatter(x_vals_comp, vals, label=f'+{y} (stack {len(y_cols_comp)-i})', color=color, alpha=alpha_value_comp, marker='x', s=30)
                        else:
                            sort_idx = np.argsort(x_vals_comp)
                            plt.plot(np.array(x_vals_comp)[sort_idx], np.array(vals)[sort_idx], marker='x', label=f'+{y} (stack {len(y_cols_comp)-i})', color=color, linestyle=':')
                        offset -= 1
                elif len(y_cols_comp) > 1:
                    # Multi-y comparison - plot on the same axes as original
                    # Plot the comparison data on matching axes
                    for i, y in enumerate(y_cols_comp):
                        # Only plot if this feature exists in both datasets
                        if y not in y_cols:
                            continue
                        # Find the index in original y_cols to use same axis and color
                        orig_idx = y_cols.index(y)
                        color = bright_colors[orig_idx % len(bright_colors)]
                        
                        if orig_idx == 0:
                            # Use twin_axes[0] which is the main axis, not plt.gca()
                            main_ax = twin_axes[0]
                            if plot_overlay.plot_type == 'scatter':
                                l = main_ax.scatter(x_vals_comp, df_comp[y], color=color, label=f'+{y}', alpha=alpha_value_comp, marker='x', s=30)
                            else:
                                sort_idx = np.argsort(x_vals_comp)
                                l, = main_ax.plot(np.array(x_vals_comp)[sort_idx], np.array(df_comp[y])[sort_idx], marker='x', color=color, label=f'+{y}', linestyle=':')
                            lines.append(l)
                            labels.append(f'+{y}')
                        else:
                            # Use the corresponding twin axis
                            if 'twin_axes' in locals() and orig_idx < len(twin_axes):
                                ax_comp = twin_axes[orig_idx]
                                if plot_overlay.plot_type == 'scatter':
                                    l = ax_comp.scatter(x_vals_comp, df_comp[y], color=color, label=f'+{y}', alpha=alpha_value_comp, marker='x', s=30)
                                else:
                                    sort_idx = np.argsort(x_vals_comp)
                                    l, = ax_comp.plot(np.array(x_vals_comp)[sort_idx], np.array(df_comp[y])[sort_idx], marker='x', color=color, label=f'+{y}', linestyle=':')
                                lines.append(l)
                                labels.append(f'+{y}')
                
                # Re-apply y-limits after all comparison data is plotted to ensure correct scaling
                if y_limits and twin_axes:
                    for idx, y_col in enumerate(y_cols):
                        if y_col in y_limits:
                            if idx == 0 and len(twin_axes) > 0:
                                twin_axes[0].set_ylim(y_limits[y_col])
                            elif idx < len(twin_axes):
                                twin_axes[idx].set_ylim(y_limits[y_col])
                
                elif len(y_cols_comp) == 1:
                    # Single-y comparison
                    y = y_cols_comp[0]
                    color = bright_colors[0]
                    if plot_overlay.plot_type == 'scatter':
                        plt.scatter(x_vals_comp, df_comp[y], label=f'+{y}', color=color, alpha=alpha_value_comp, marker='x', s=30)
                    else:
                        sort_idx = np.argsort(x_vals_comp)
                        plt.plot(np.array(x_vals_comp)[sort_idx], np.array(df_comp[y])[sort_idx], marker='x', label=f'+{y}', color=color, linestyle=':')
                
                # Add comparison dataset annotations if annotation mode is enabled
                if getattr(plot_overlay, 'annotation', False):
                    stats_text_comp = 'Comparison Dataset:\n\n'
                    for y in y_cols_comp:
                        mean = df_comp[y].mean()
                        median = df_comp[y].median()
                        min_val = df_comp[y].min()
                        max_val = df_comp[y].max()
                        # Calculate trendline slope
                        try:
                            z = np.polyfit(df_comp[x_col], df_comp[y], 1)
                            slope = z[0]
                        except Exception:
                            slope = float('nan')
                        stats_text_comp += f'+{y}:\n  Mean: {mean:.3f}\n  Median: {median:.3f}\n  Min: {min_val:.3f}\n  Max: {max_val:.3f}\n  Slope: {slope:.3f}\n\n'
                    ax = plt.gca()
                    # Place comparison annotations in bottom right (0.99, 0.01)
                    ax.text(0.99, 0.01, stats_text_comp, transform=ax.transAxes, fontsize=11, color='cyan', va='bottom', ha='right', bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan', boxstyle='round,pad=0.5'))
    
    # Create comprehensive legend at the end (after all plotting including comparison)
    # This ensures all features are captured in the legend
    if len(y_cols) > 1 and not getattr(plot_overlay, 'stack', False) and (not getattr(plot_overlay, 'residuals', False) or getattr(plot_overlay, 'residuals', False)):
        # For multi-y or residuals with dual axes, use the stored lines and labels from plotting
        # Filter out trendlines
        filtered = [(h, l) for h, l in zip(lines, labels) if 'trendline' not in l.lower() and not l.startswith('_')]
        if filtered:
            handles, labels_filtered = zip(*filtered)
        else:
            handles, labels_filtered = [], []
        
        ax = plt.gca()
        legend = ax.legend(handles, labels_filtered, title='Y-Features', loc='upper right')
        legend.set_zorder(1000)
        if getattr(plot_overlay, 'alpha', False):
            for lh in legend.legend_handles:
                lh.set_alpha(1.0)
    else:
        # For other modes (stacked, single-y), create unified legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out trendlines
        filtered = [(h, l) for h, l in zip(handles, labels) if 'trendline' not in l.lower()]
        if filtered:
            handles, labels = zip(*filtered)
            legend = plt.legend(handles, labels, title='Y-Features', loc='upper right')
        else:
            legend = plt.legend(title='Y-Features', loc='upper right')
        legend.set_zorder(1000)
        if getattr(plot_overlay, 'alpha', False):
            for lh in legend.legend_handles:
                lh.set_alpha(1.0)
    
    plt.tight_layout()
    plt.show()
    plt.tight_layout()

plot_overlay.plot_type = 'scatter'  # default
plot_overlay.stack = False  # default
plot_overlay.trendline = False  # default
plot_overlay.residuals = False  # default
plot_overlay.bootstrap = False  # default
plot_overlay.comparison = False  # default
    

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
        self.formula_label = QLabel('Formula Input (use a-j in expression format):')
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
        
        # Add Comparison Dataset section
        self.comparison_hr = QFrame()
        self.comparison_hr.setFrameShape(QFrame.HLine)
        self.comparison_hr.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.comparison_hr)
        self.comparison_label = QLabel('Comparison Dataset:')
        self.layout.addWidget(self.comparison_label)
        
        self.btn_load_comparison = QPushButton('Load Comparison Dataset')
        self.btn_load_comparison.clicked.connect(self.load_comparison_dataset)
        self.layout.addWidget(self.btn_load_comparison)
        
        self.comparison_file_label = QLabel('No comparison dataset loaded')
        self.comparison_file_label.setStyleSheet('color: gray; font-style: italic;')
        self.layout.addWidget(self.comparison_file_label)
        
        self.comparison_checkbox = QCheckBox('Plot Comparison')
        self.layout.addWidget(self.comparison_checkbox)

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
        
        self.export_inputs_btn = QPushButton('Export DOOIT Inputs')
        self.export_inputs_btn.clicked.connect(self.export_dooit_inputs)
        self.layout.addWidget(self.export_inputs_btn)
        self.layout.addStretch(1)  # Push export buttons to bottom

        self.df = None
        self.df_comparison = None  # Store comparison dataset
        self.comparison_filename = None  # Store comparison filename
        # Track file loading inputs for export
        self.loaded_filename = None
        self.imputation_applied = False
        self.columns_deleted = []
        self.binary_columns_detected = {}
        self.threshold_method = None
        self.sum_column_created = False
        self.sum_features = []
        self.sum_column_name = ''
        self.custom_column_created = False
        self.custom_column_name = ''
        self.custom_column_formula = ''
        self.custom_column_var_map = {}
        self.filters_applied = []
        self.filter_logic = []
        self.thresholds_file = None
        self.highlight_feature = None

    def load_comparison_dataset(self):
        """
        Loads a comparison dataset CSV file and cleans it to contain only numeric and datetime columns.
        """
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Comparison CSV', '', 'CSV Files (*.csv)')
        if fname:
            try:
                df_comp = pd.read_csv(fname)
                # Check for 'timestamp' column and convert to datetime
                if 'timestamp' in df_comp.columns:
                    try:
                        df_comp['timestamp'] = pd.to_datetime(df_comp['timestamp'], unit='s')
                    except Exception:
                        pass
                # Clean to keep only numeric and datetime columns
                df_comp = clean_numeric_df(df_comp)
                if df_comp.shape[0] == 0 or df_comp.shape[1] == 0:
                    QMessageBox.warning(self, 'Error', 'Comparison dataset has no valid numeric/datetime data.')
                    return
                self.df_comparison = df_comp
                self.comparison_filename = fname  # Store filename for display
                # Update the label to show the loaded filename
                import os
                self.comparison_file_label.setText(f'Loaded: {os.path.basename(fname)}')
                QMessageBox.information(self, 'Success', f'Comparison dataset loaded: {df_comp.shape[0]} rows, {df_comp.shape[1]} columns')
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to load comparison dataset: {e}')

    def load_csv(self):
        """
        Loads a CSV file, optionally creates a SUM column, allows filtering by non-numeric features,
    plot_type = 'Scatter' if self.radio_scatter.isChecked() else 'Line with Markers'
    plot_overlay(self.df, x_col, y_cols, plot_type)
        """
        fname, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '', 'CSV Files (*.csv)')
        if fname:
            self.loaded_filename = fname  # Store filename for export
            try:
                from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox, QHBoxLayout, QLineEdit
                df = pd.read_csv(fname)
                # Prompt for missing value imputation first
                reply_impute = QMessageBox.question(self, 'Impute Missing Values?', 
                    'If there are missing values, that index will be purged from the dataframe for processing. Before this process occurs, would you like to impute latest value over missing values?', 
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                self.imputation_applied = (reply_impute == QMessageBox.Yes)
                if reply_impute == QMessageBox.Yes:
                    # Forward fill: replace NaN with the value from the index above
                    df = df.fillna(method='ffill')
                # Check for 'timestamp' column and convert to datetime
                if 'timestamp' in df.columns:
                    try:
                        # Convert Unix timestamp (seconds since 1970-01-01 00:00:00) to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    except Exception as e:
                        QMessageBox.warning(self, 'Warning', f'Could not convert timestamp column: {e}')
                # Binary value detector - check for columns with exactly 2 distinct values
                for col in df.columns:
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) == 2:
                        reply_binary = QMessageBox.question(self, 'Binary Value Detected', 
                            f'A binary value set has been detected in the "{col}" column. Would you like to assign integer values to it?', 
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                        if reply_binary == QMessageBox.Yes:
                            class BinaryValueDialog(QDialog):
                                def __init__(self, column_name, values, parent=None):
                                    super().__init__(parent)
                                    self.setWindowTitle('Select Value for 0')
                                    self.layout = QVBoxLayout(self)
                                    self.layout.addWidget(QLabel(f'Select which value will be represented by the number 0,\nbearing in mind that the other value will be represented by the number 1.'))
                                    self.radio1 = QRadioButton(str(values[0]))
                                    self.radio2 = QRadioButton(str(values[1]))
                                    self.radio1.setChecked(True)
                                    self.layout.addWidget(self.radio1)
                                    self.layout.addWidget(self.radio2)
                                    self.btn_ok = QPushButton('OK')
                                    self.btn_ok.clicked.connect(self.accept)
                                    self.layout.addWidget(self.btn_ok)
                                    self.values = values
                                def get_zero_value(self):
                                    return self.values[0] if self.radio1.isChecked() else self.values[1]
                            
                            binary_dialog = BinaryValueDialog(col, unique_values, self)
                            if binary_dialog.exec_() == QDialog.Accepted:
                                zero_value = binary_dialog.get_zero_value()
                                # Map the values: selected value -> 0, other value -> 1
                                df[col] = df[col].map({zero_value: 0, unique_values[0] if unique_values[0] != zero_value else unique_values[1]: 1})
                                # Track binary column detection
                                self.binary_columns_detected[col] = {'zero_value': zero_value, 'one_value': unique_values[0] if unique_values[0] != zero_value else unique_values[1]}
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
                            self.filters_applied = filters
                            self.filter_logic = logic_ops
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
                # Prompt user to create a SUM column from numeric features (moved here so binary columns can be summed)
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                reply = QMessageBox.question(self, 'Create SUM Column?', 'Would you like to create a SUM column from numeric features?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                self.sum_column_created = (reply == QMessageBox.Yes)
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
                            self.sum_features = sum_features
                            self.sum_column_name = sum_name
                            df[sum_name] = df[sum_features].sum(axis=1)
                # Prompt for custom column (moved here so custom columns can use previously created columns)
                reply_custom = QMessageBox.question(self, 'Create Custom Column?', 'Would you like to create a column with custom values?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                self.custom_column_created = (reply_custom == QMessageBox.Yes)
                if reply_custom == QMessageBox.Yes:
                    custom_dialog = CustomColumnDialog(df, self)
                    if custom_dialog.exec_() == QDialog.Accepted:
                        col_name, col_vals = custom_dialog.get_custom_column()
                        if col_name and col_vals is not None:
                            self.custom_column_name = col_name
                            self.custom_column_formula = custom_dialog.formula_edit.text().strip()
                            self.custom_column_var_map = {var: custom_dialog.var_combos[var].currentText() 
                                                          for var in custom_dialog.var_combos 
                                                          if custom_dialog.var_combos[var].currentText()}
                            df[col_name] = col_vals
                # Prompt user to delete columns LAST (after all other operations)
                reply_delete = QMessageBox.question(self, 'Delete Columns?', 'Would you like to delete any columns?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply_delete == QMessageBox.Yes:
                    class DeleteColumnsDialog(QDialog):
                        def __init__(self, columns, parent=None):
                            super().__init__(parent)
                            self.setWindowTitle('Select Columns to Delete')
                            self.layout = QVBoxLayout(self)
                            self.checkboxes = []
                            self.layout.addWidget(QLabel('Select columns to delete:'))
                            for col in columns:
                                cb = QCheckBox(col)
                                self.layout.addWidget(cb)
                                self.checkboxes.append(cb)
                            self.btn_delete = QPushButton('Delete These Columns')
                            self.btn_delete.clicked.connect(self.accept)
                            self.layout.addWidget(self.btn_delete)
                        def get_columns_to_delete(self):
                            return [cb.text() for cb in self.checkboxes if cb.isChecked()]
                    
                    delete_dialog = DeleteColumnsDialog(list(df.columns), self)
                    if delete_dialog.exec_() == QDialog.Accepted:
                        cols_to_delete = delete_dialog.get_columns_to_delete()
                        if cols_to_delete:
                            self.columns_deleted = cols_to_delete
                            df = df.drop(columns=cols_to_delete)
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
                    self.thresholds_file = fname_thresh if fname_thresh else None
                    if fname_thresh:
                        try:
                            df_thresh = pd.read_csv(fname_thresh)
                            if not df_thresh.empty:
                                # Create dialog to choose threshold method
                                class ThresholdMethodDialog(QDialog):
                                    def __init__(self, parent=None):
                                        super().__init__(parent)
                                        self.setWindowTitle('Select Threshold Method')
                                        self.layout = QVBoxLayout(self)
                                        self.layout.addWidget(QLabel('Choose which threshold values to use:'))
                                        self.radio_latest = QRadioButton('Latest value (last row)')
                                        self.radio_minmax = QRadioButton('Highest and lowest values')
                                        self.radio_latest.setChecked(True)
                                        self.layout.addWidget(self.radio_latest)
                                        self.layout.addWidget(self.radio_minmax)
                                        self.btn_ok = QPushButton('OK')
                                        self.btn_ok.clicked.connect(self.accept)
                                        self.layout.addWidget(self.btn_ok)
                                    def get_method(self):
                                        return 'latest' if self.radio_latest.isChecked() else 'minmax'
                                
                                threshold_dialog = ThresholdMethodDialog(self)
                                if threshold_dialog.exec_() == QDialog.Accepted:
                                    method = threshold_dialog.get_method()
                                    self.threshold_method = method
                                    if method == 'latest':
                                        last_row = df_thresh.iloc[-1]
                                        thresholds = {col: last_row[col] for col in df_thresh.columns}
                                    else:  # minmax
                                        thresholds = {}
                                        for col in df_thresh.columns:
                                            min_val = df_thresh[col].min()
                                            max_val = df_thresh[col].max()
                                            thresholds[col] = [min_val, max_val]
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
                        self.highlight_feature = highlight_feature
                # Convert datetime columns to numeric (Unix timestamp) for pair plot
                df_for_pairgrid = df.copy()
                for col in df_for_pairgrid.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_for_pairgrid[col]):
                        # Convert to Unix timestamp (seconds since epoch) as float
                        df_for_pairgrid[col] = df_for_pairgrid[col].astype('int64') / 10**9
                plot_pairgrid(df_for_pairgrid, highlight_feature=highlight_feature, thresholds=thresholds)
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
        # Set alpha mode based on checkbox
        plot_overlay.alpha = self.alpha_checkbox.isChecked()
        # Set comparison mode based on checkbox
        plot_overlay.comparison = self.comparison_checkbox.isChecked()
        # Parse x/y intercepts and names
        x_intercepts = [float(x.strip()) for x in self.x_intercepts_edit.text().split(',') if x.strip()] if self.x_intercepts_edit.text().strip() else None
        x_names = [n.strip() for n in self.x_intercepts_names_edit.text().split(',') if n.strip()] if self.x_intercepts_names_edit.text().strip() else None
        y_intercepts = [float(y.strip()) for y in self.y_intercepts_edit.text().split(',') if y.strip()] if self.y_intercepts_edit.text().strip() else None
        y_names = [n.strip() for n in self.y_intercepts_names_edit.text().split(',') if n.strip()] if self.y_intercepts_names_edit.text().strip() else None
        plot_overlay(self.df, x_col, y_cols, x_intercepts=x_intercepts, x_names=x_names, y_intercepts=y_intercepts, y_names=y_names, df_comparison=self.df_comparison)
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

    def export_dooit_inputs(self):
        """
        Exports all DOOIT user inputs to a CSV file so the plot can be recreated.
        """
        import csv
        fname, _ = QFileDialog.getSaveFileName(self, 'Export DOOIT Inputs', '', 'CSV Files (*.csv)')
        if fname:
            # Gather all inputs
            inputs_data = []
            
            # File loading information
            inputs_data.append(['Loaded Dataset File', self.loaded_filename if self.loaded_filename else 'None'])
            inputs_data.append(['Comparison Dataset File', self.comparison_filename if self.comparison_filename else 'None'])
            
            # Data processing steps (in order)
            inputs_data.append(['Missing Value Imputation Applied', str(self.imputation_applied)])
            
            inputs_data.append(['SUM Column Created', str(self.sum_column_created)])
            if self.sum_column_created and self.sum_features:
                inputs_data.append(['SUM Features', ', '.join(self.sum_features)])
                inputs_data.append(['SUM Column Name', self.sum_column_name])
            
            inputs_data.append(['Custom Column Created', str(self.custom_column_created)])
            if self.custom_column_created and self.custom_column_name:
                inputs_data.append(['Custom Column Name', self.custom_column_name])
                inputs_data.append(['Custom Column Formula', self.custom_column_formula])
                for var, col in self.custom_column_var_map.items():
                    inputs_data.append([f'Custom Column Variable {var}', col])
            
            if self.binary_columns_detected:
                for col, mapping in self.binary_columns_detected.items():
                    inputs_data.append([f'Binary Column Detected: {col}', f"0={mapping['zero_value']}, 1={mapping['one_value']}"])
            
            if self.filters_applied:
                for idx, (col, val) in enumerate(self.filters_applied):
                    logic = self.filter_logic[idx-1] if idx > 0 and idx-1 < len(self.filter_logic) else 'N/A'
                    inputs_data.append([f'Filter {idx+1}', f'{col} == {val} (Logic: {logic})'])
            
            if self.columns_deleted:
                inputs_data.append(['Columns Deleted', ', '.join(self.columns_deleted)])
            
            inputs_data.append(['Thresholds File', self.thresholds_file if self.thresholds_file else 'None'])
            if self.thresholds_file:
                inputs_data.append(['Threshold Method', self.threshold_method if self.threshold_method else 'N/A'])
            inputs_data.append(['Highlight Feature', self.highlight_feature if self.highlight_feature else 'None'])
            
            # X feature selection
            x_items = self.list_x.selectedItems()
            if x_items:
                inputs_data.append(['X Feature', x_items[0].text()])
            
            # Y feature selections
            y_items = self.list_y.selectedItems()
            for idx, item in enumerate(y_items[:5]):
                inputs_data.append([f'Y Feature {idx+1}', item.text()])
            
            # Plot type
            if self.radio_scatter.isChecked():
                inputs_data.append(['Plot Type', 'Scatter'])
            else:
                inputs_data.append(['Plot Type', 'Line with Markers'])
            
            # Data toggles
            inputs_data.append(['Stack Y Features', str(self.stack_checkbox.isChecked())])
            inputs_data.append(['Show Trendline', str(self.trendline_checkbox.isChecked())])
            inputs_data.append(['Plot Residuals', str(self.residuals_checkbox.isChecked())])
            inputs_data.append(['Bootstrap Data', str(self.bootstrap_checkbox.isChecked())])
            inputs_data.append(['Vary Transparency by Density', str(self.alpha_checkbox.isChecked())])
            inputs_data.append(['Annotate with Stats', str(self.annotation_checkbox.isChecked())])
            inputs_data.append(['Plot Comparison', str(self.comparison_checkbox.isChecked())])
            
            # Intercepts and names
            inputs_data.append(['X-intercepts', self.x_intercepts_edit.text()])
            inputs_data.append(['X-intercept Names', self.x_intercepts_names_edit.text()])
            inputs_data.append(['Y-intercepts', self.y_intercepts_edit.text()])
            inputs_data.append(['Y-intercept Names', self.y_intercepts_names_edit.text()])
            
            # Write to CSV
            with open(fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Input Field', 'Value'])
                writer.writerows(inputs_data)
            
            QMessageBox.information(self, 'Exported', f'DOOIT inputs exported to {fname}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PairPlotApp()
    window.show()
    sys.exit(app.exec_())