

import streamlit as st
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import pearsonr
from numpy.polynomial.polynomial import Polynomial
from sklearn.preprocessing import MinMaxScaler
from numpy import abs, sqrt, log, exp, sin, cos, tan, arcsin, arccos, arctan, round, floor, ceil, clip
from math import atan2, pow, degrees, radians, pi
import io

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
plot_overlay.annotation = False  # default
plot_overlay.alpha = False  # default

# Streamlit Web Interface
def main():
    st.set_page_config(page_title="DOOIT Web", layout="wide", page_icon="📊")
    
    st.title("📊 DOOIT - Data Observation & Overlay Insights Tool")
    st.markdown("**Interactive Data Analysis and Visualization**")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_comparison' not in st.session_state:
        st.session_state.df_comparison = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 Data Loading")
        
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
        
        if uploaded_file is not None and not st.session_state.processed:
            df = pd.read_csv(uploaded_file)
            
            # Data processing options
            st.subheader("Data Processing")
            
            impute = st.checkbox("Impute missing values (forward fill)")
            if impute:
                df = df.fillna(method='ffill')
            
            # Convert timestamp if exists
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                except:
                    pass
            
            # Binary column handling
            binary_cols = {}
            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2:
                    convert_binary = st.checkbox(f"Convert binary column '{col}' to 0/1?")
                    if convert_binary:
                        val_0 = st.selectbox(f"Select value for 0 in {col}", unique_vals, key=f"bin_{col}")
                        val_1 = unique_vals[0] if unique_vals[0] != val_0 else unique_vals[1]
                        df[col] = df[col].map({val_0: 0, val_1: 1})
            
            # Filter by non-numeric columns
            non_numeric_cols = [col for col in df.columns if col not in df.select_dtypes(include=[np.number]).columns]
            if non_numeric_cols:
                filter_data = st.checkbox("Filter by non-numeric columns?")
                if filter_data:
                    st.write("**Filter Configuration** (combine up to 5 filters)")
                    filter_queries = []
                    for i in range(5):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            filter_col = st.selectbox(f"Filter {i+1} Column", [""] + non_numeric_cols, key=f"fcol_{i}")
                        with col2:
                            if filter_col:
                                filter_val = st.selectbox(f"Value", df[filter_col].unique(), key=f"fval_{i}")
                            else:
                                filter_val = None
                        with col3:
                            if i > 0:
                                filter_logic = st.selectbox("Logic", ["", "AND", "OR"], key=f"flogic_{i}")
                            else:
                                filter_logic = None
                        
                        if filter_col and filter_val is not None:
                            filter_queries.append((filter_col, filter_val, filter_logic))
                    
                    if filter_queries and st.button("Apply Filters"):
                        query = ''
                        for idx, (col, val, logic) in enumerate(filter_queries):
                            cond = f"`{col}` == @val"
                            if idx == 0:
                                query += f"`{col}` == '{val}'"
                            else:
                                op = logic.lower() if logic else 'and'
                                if op == 'and':
                                    query += f" & `{col}` == '{val}'"
                                else:
                                    query += f" | `{col}` == '{val}'"
                        try:
                            df = df.query(query)
                            st.success(f"Filters applied! {len(df)} rows remaining")
                        except Exception as e:
                            st.error(f"Filter error: {e}")
            
            # SUM column creation
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            create_sum = st.checkbox("Create SUM column?")
            if create_sum and len(numeric_cols) > 1:
                sum_cols = st.multiselect("Select columns to sum", numeric_cols)
                sum_name = st.text_input("Name for SUM column", "SUM")
                if sum_cols and sum_name and st.button("Create SUM Column"):
                    df[sum_name] = df[sum_cols].sum(axis=1)
                    st.success(f"Created column '{sum_name}'")
            
            # Custom column creation
            create_custom = st.checkbox("Create custom column with formula?")
            if create_custom:
                st.write("**Custom Column Formula** (use variables a-j for columns)")
                custom_name = st.text_input("Custom column name", "custom")
                
                # Variable mapping
                st.write("Assign columns to variables:")
                var_map = {}
                numeric_cols_for_custom = list(df.select_dtypes(include=[np.number]).columns)
                cols_per_row = 2
                for i in range(0, 10, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < 10:
                            var = chr(ord('a') + idx)
                            with cols[j]:
                                selected = st.selectbox(f"Variable {var}", [""] + numeric_cols_for_custom, key=f"var_{var}")
                                if selected:
                                    var_map[var] = selected
                
                formula = st.text_input("Formula (e.g., a + b, np.sqrt(c), a * b / c)", "")
                
                if custom_name and formula and st.button("Create Custom Column"):
                    try:
                        # Build local dict for eval
                        local_vars = {'np': np, 'math': math}
                        for var, col in var_map.items():
                            if col:
                                local_vars[var] = df[col]
                        
                        result = eval(formula, {"__builtins__": None}, local_vars)
                        df[custom_name] = result
                        st.success(f"Created custom column '{custom_name}'")
                    except Exception as e:
                        st.error(f"Formula error: {e}")
            
            # Column deletion
            delete_cols_check = st.checkbox("Delete columns?")
            if delete_cols_check:
                cols_to_delete = st.multiselect("Select columns to delete", list(df.columns))
                if cols_to_delete and st.button("Delete Selected Columns"):
                    df = df.drop(columns=cols_to_delete)
                    st.success(f"Deleted {len(cols_to_delete)} columns")
            
            if st.button("Process Data"):
                df_clean = clean_numeric_df(df)
                if df_clean.shape[1] < 2:
                    st.error("Not enough numeric columns for plotting")
                else:
                    st.session_state.df = df_clean
                    st.session_state.processed = True
                    st.success(f"Data loaded: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
                    st.rerun()
        
        # Comparison dataset
        st.header("📊 Comparison Dataset")
        comp_file = st.file_uploader("Upload Comparison CSV", type=['csv'], key='comparison')
        if comp_file:
            df_comp = pd.read_csv(comp_file)
            if 'timestamp' in df_comp.columns:
                try:
                    df_comp['timestamp'] = pd.to_datetime(df_comp['timestamp'], unit='s')
                except:
                    pass
            df_comp = clean_numeric_df(df_comp)
            st.session_state.df_comparison = df_comp
            st.success(f"Comparison loaded: {df_comp.shape[0]} rows, {df_comp.shape[1]} columns")
    
    # Main content area
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Display dataframe
        with st.expander("📋 View DataFrame", expanded=False):
            st.dataframe(df, width='stretch')
            
            # Export CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "dooit_export.csv", "text/csv")
        
        # Pair plot section
        st.header("🔀 Pair Plot")
        col1, col2, col3 = st.columns(3)
        with col1:
            highlight_feature = st.selectbox("Highlight feature (optional)", ["None"] + list(df.select_dtypes(include=[np.number]).columns), key='highlight_pairplot_1')
        with col2:
            # Load thresholds
            load_thresholds = st.checkbox("Load thresholds from CSV?")
            thresholds = None
            if load_thresholds:
                threshold_file = st.file_uploader("Upload threshold CSV", type=['csv'], key='thresholds')
                if threshold_file:
                    df_thresh = pd.read_csv(threshold_file)
                    threshold_method = st.radio("Threshold method", ["Latest value (last row)", "Min/Max values"])
                    
                    if threshold_method == "Latest value (last row)":
                        last_row = df_thresh.iloc[-1]
                        thresholds = {col: last_row[col] for col in df_thresh.columns}
                    else:
                        thresholds = {}
                        for col in df_thresh.columns:
                            min_val = df_thresh[col].min()
                            max_val = df_thresh[col].max()
                            thresholds[col] = [min_val, max_val]
                    st.success("Thresholds loaded!")
        with col3:
            show_pairplot = st.button("Generate Pair Plot", key='gen_pairplot_1')
        
        if show_pairplot:
            df_for_pairgrid = df.copy()
            for col in df_for_pairgrid.columns:
                if pd.api.types.is_datetime64_any_dtype(df_for_pairgrid[col]):
                    df_for_pairgrid[col] = df_for_pairgrid[col].astype('int64') / 10**9
            
            highlight = None if highlight_feature == "None" else highlight_feature
            fig = plt.figure(figsize=(12, 10))
            plot_pairgrid(df_for_pairgrid, highlight_feature=highlight, thresholds=thresholds)
            st.pyplot(plt.gcf())
            plt.close()
        
        # Pair plot section
        st.header("🔀 Pair Plot")
        col1, col2 = st.columns(2)
        with col1:
            highlight_feature = st.selectbox("Highlight feature (optional)", ["None"] + list(df.select_dtypes(include=[np.number]).columns), key='highlight_pairplot_2')
        with col2:
            show_pairplot = st.button("Generate Pair Plot", key='gen_pairplot_2')
        
        if show_pairplot:
            df_for_pairgrid = df.copy()
            for col in df_for_pairgrid.columns:
                if pd.api.types.is_datetime64_any_dtype(df_for_pairgrid[col]):
                    df_for_pairgrid[col] = df_for_pairgrid[col].astype('int64') / 10**9
            
            highlight = None if highlight_feature == "None" else highlight_feature
            fig = plt.figure(figsize=(12, 10))
            plot_pairgrid(df_for_pairgrid, highlight_feature=highlight)
            st.pyplot(plt.gcf())
            plt.close()
        
        # Overlay plot section
        st.header("📈 Overlay Plot")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("X-axis feature", df.columns)
        
        with col2:
            y_cols = st.multiselect("Y-axis features (up to 5)", df.columns, max_selections=5)
        
        with col3:
            plot_type = st.radio("Plot type", ["scatter", "line"])
        
        # Plot options
        st.subheader("Plot Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stack = st.checkbox("Stack Y features")
            trendline = st.checkbox("Show trendline")
        
        with col2:
            residuals = st.checkbox("Plot residuals")
            bootstrap = st.checkbox("Bootstrap (10k samples)")
        
        with col3:
            alpha_mode = st.checkbox("Vary transparency")
            annotation = st.checkbox("Show statistics")
        
        # Comparison and intercepts
        col1, col2 = st.columns(2)
        
        with col1:
            plot_comparison = st.checkbox("Plot comparison dataset") if st.session_state.df_comparison is not None else False
        
        with col2:
            st.text_input("X-intercepts (comma-separated)", key="x_intercepts")
            st.text_input("Y-intercepts (comma-separated)", key="y_intercepts")
        
        # Generate overlay plot
        if st.button("Generate Overlay Plot"):
            if x_col and y_cols:
                # Remove x from y if present
                y_cols_filtered = [y for y in y_cols if y != x_col]
                
                if not y_cols_filtered:
                    st.error("Please select at least one Y feature different from X")
                else:
                    # Set plot parameters
                    plot_overlay.plot_type = plot_type
                    plot_overlay.stack = stack
                    plot_overlay.trendline = trendline
                    plot_overlay.residuals = residuals
                    plot_overlay.bootstrap = bootstrap
                    plot_overlay.alpha = alpha_mode
                    plot_overlay.annotation = annotation
                    plot_overlay.comparison = plot_comparison
                    
                    # Parse intercepts
                    x_intercepts = None
                    y_intercepts = None
                    if st.session_state.get("x_intercepts"):
                        try:
                            x_intercepts = [float(x.strip()) for x in st.session_state.x_intercepts.split(',') if x.strip()]
                        except:
                            pass
                    if st.session_state.get("y_intercepts"):
                        try:
                            y_intercepts = [float(y.strip()) for y in st.session_state.y_intercepts.split(',') if y.strip()]
                        except:
                            pass
                    
                    # Generate plot
                    fig = plt.figure(figsize=(12, 8))
                    plot_overlay(
                        df, x_col, y_cols_filtered,
                        x_intercepts=x_intercepts,
                        y_intercepts=y_intercepts,
                        df_comparison=st.session_state.df_comparison if plot_comparison else None
                    )
                    st.pyplot(plt.gcf())
                    plt.close()
            else:
                st.error("Please select X and Y features")
    
    else:
        st.info("👆 Upload a CSV file in the sidebar to begin")
        
        # Instructions
        st.markdown("""
        ### How to Use DOOIT Web
        
        1. **Upload Data**: Use the sidebar to upload your CSV dataset
        2. **Process Data**: Configure data processing options (imputation, binary conversion, etc.)
        3. **Generate Plots**: 
           - **Pair Plot**: Visualize all feature relationships
           - **Overlay Plot**: Compare specific features with advanced options
        4. **Export**: Download processed data or save plots
        
        ### Features
        - 📊 Interactive pair plots with correlation statistics
        - 📈 Multiple overlay plot modes (stack, residuals, trendlines)
        - 🔄 Dataset comparison capabilities
        - 📥 CSV export functionality
        - 🎨 Statistical annotations and customizations
        """)

if __name__ == '__main__':
    main()

