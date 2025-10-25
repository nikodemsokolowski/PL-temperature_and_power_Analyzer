import customtkinter as ctk
import logging
import matplotlib
matplotlib.use('TkAgg') # Necessary backend for tkinter embedding
from typing import Optional, Tuple
from matplotlib.colors import LogNorm, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

logger = logging.getLogger(__name__)

class PlotCanvas(ctk.CTkFrame):
    """
    A CustomTkinter frame containing an embedded Matplotlib plot and toolbar.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create Matplotlib Figure and Axes
        # Adjust dpi for clarity if needed
        self.figure = Figure(figsize=(7.5, 5.5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel("Energy (eV)")
        self.axes.set_ylabel("Intensity (arb. units)")
        self.figure.tight_layout() # Adjust layout to prevent labels overlapping
        self.colorbar = None
        self._extra_axes = []
        self._info_artist = None

        # Create Canvas to display the figure
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Create Matplotlib Navigation Toolbar
        # Frame to hold the toolbar (optional, for better layout control)
        self.toolbar_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.toolbar_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Initial plot setup (e.g., empty plot)
        self.axes.plot([], []) # Plot empty data initially
        self.canvas.draw()
        logger.debug("PlotCanvas initialized.")

    def _remove_colorbars(self):
        """Remove any existing colorbars or auxiliary artists."""
        if getattr(self, 'colorbar', None):
            try:
                self.colorbar.remove()
            except Exception as exc:
                logger.debug(f"Failed to remove colorbar: {exc}")
            self.colorbar = None
        if getattr(self, '_extra_axes', None):
            for ax in self._extra_axes:
                try:
                    ax.remove()
                except Exception as exc:
                    logger.debug(f"Failed to remove auxiliary axis: {exc}")
        self._extra_axes = []
        if getattr(self, '_info_artist', None):
            try:
                self._info_artist.remove()
            except Exception as exc:
                logger.debug(f"Failed to remove info annotation: {exc}")
        self._info_artist = None

    def apply_style(self, style_options=None):
        """Applies global style options via rcParams."""
        try:
            import matplotlib as mpl
            if style_options is None:
                style_options = {}
            preset = style_options.get('preset')
            font_size = style_options.get('font_size')
            ticks_inside = style_options.get('ticks_inside', True)
            minor_ticks = style_options.get('minor_ticks', False)
            maj_len = float(style_options.get('maj_len', 4.0))
            min_len = float(style_options.get('min_len', 2.0))
            axes_lw = float(style_options.get('axes_linewidth', 1.0))

            # Base preset tweaks
            if preset == 'Compact':
                mpl.rcParams['figure.autolayout'] = False
                mpl.rcParams['axes.labelsize'] = font_size or 9
                mpl.rcParams['axes.titlesize'] = (font_size or 9) + 1
                mpl.rcParams['xtick.labelsize'] = font_size or 8
                mpl.rcParams['ytick.labelsize'] = font_size or 8
            elif preset == 'Nature':
                mpl.rcParams['axes.linewidth'] = axes_lw
                mpl.rcParams['axes.labelsize'] = font_size or 10
                mpl.rcParams['axes.titlesize'] = (font_size or 10) + 1
                mpl.rcParams['xtick.labelsize'] = font_size or 9
                mpl.rcParams['ytick.labelsize'] = font_size or 9
            elif preset == 'APS':
                mpl.rcParams['axes.linewidth'] = axes_lw
                mpl.rcParams['axes.labelsize'] = font_size or 9
                mpl.rcParams['axes.titlesize'] = (font_size or 9)
                mpl.rcParams['xtick.labelsize'] = font_size or 8
                mpl.rcParams['ytick.labelsize'] = font_size or 8
            elif preset == 'Science':
                mpl.rcParams['axes.linewidth'] = axes_lw
                mpl.rcParams['axes.labelsize'] = font_size or 11
                mpl.rcParams['axes.titlesize'] = (font_size or 11) + 1
                mpl.rcParams['xtick.labelsize'] = font_size or 10
                mpl.rcParams['ytick.labelsize'] = font_size or 10

            if font_size is not None:
                mpl.rcParams['font.size'] = font_size
            mpl.rcParams['xtick.direction'] = 'in' if ticks_inside else 'out'
            mpl.rcParams['ytick.direction'] = 'in' if ticks_inside else 'out'
            mpl.rcParams['xtick.major.size'] = maj_len
            mpl.rcParams['ytick.major.size'] = maj_len
            mpl.rcParams['xtick.minor.size'] = min_len
            mpl.rcParams['ytick.minor.size'] = min_len
            mpl.rcParams['axes.linewidth'] = axes_lw
            # Save minor_ticks preference for use in axes creation
            self._minor_ticks_pref = bool(minor_ticks)
        except Exception as e:
            logger.warning(f"Failed to apply style: {e}")

    def set_yscale(self, scale='linear'):
        """Sets the Y-axis scale ('linear' or 'log')."""
        try:
            self.axes.set_yscale(scale)
            self.canvas.draw()
            logger.debug(f"Plot Y-axis scale set to {scale}")
        except ValueError as e:
            logger.error(f"Invalid scale value '{scale}': {e}")
        except Exception as e:
            logger.error(f"Error setting Y-axis scale: {e}", exc_info=True)


    def plot_data(self, energy_ev_list, counts_list, labels_list=None, title=None, y_scale='linear', x_label="Energy (eV)", y_label="Intensity (arb. units)", color_cycle=None, line_width=1.2, xlim=None, style_options=None, colors=None):
        """
        Clears the current plot and plots new data.

        Args:
            energy_ev_list: A list of numpy arrays or lists containing energy values (eV).
            counts_list: A list of numpy arrays or lists containing intensity counts.
                        Must have the same length as energy_ev_list.
            labels_list: An optional list of labels for each dataset.
            title: An optional title for the plot.
            colors: Optional explicit color list aligned with the spectra.
        """
        # Reset to a single-axes figure to avoid leftover grids
        # Preserve figure size before clearing
        current_size = self.figure.get_size_inches()
        self.figure.clear()
        self.figure.set_size_inches(current_size)
        self.axes = self.figure.add_subplot(111)
        self._remove_colorbars()

        # Apply style before plotting
        self.apply_style(style_options)

        if not energy_ev_list or not counts_list or len(energy_ev_list) != len(counts_list):
            logger.warning("Invalid data provided for plotting. Plotting empty.")
            self.axes.plot([], []) # Plot empty if data is invalid
        else:
            num_plots = len(energy_ev_list)
            use_labels = labels_list is not None and len(labels_list) == num_plots

            # Optional color cycle
            if not colors and color_cycle:
                try:
                    cmap = cm.get_cmap(color_cycle)
                    colors = getattr(cmap, 'colors', None)
                    if colors is None:
                        colors = [cmap(i/10.0) for i in range(10)]
                    self.axes.set_prop_cycle(color=colors)
                except Exception:
                    pass

            # Remember data for spike interaction
            try:
                self.set_last_lines_data(energy_ev_list, counts_list)
            except Exception:
                pass

            for i in range(num_plots):
                label = labels_list[i] if use_labels else None
                try:
                    color = None
                    if colors and i < len(colors):
                        color = colors[i]
                    plot_kwargs = {'linewidth': line_width}
                    if label:
                        plot_kwargs['label'] = label
                    if color:
                        plot_kwargs['color'] = color
                    self.axes.plot(energy_ev_list[i], counts_list[i], **plot_kwargs)
                except Exception as e:
                    logger.error(f"Error plotting dataset {i}: {e}", exc_info=True)

            if use_labels and any(labels_list): # Add legend only if labels were provided
                # Optimize legend for many traces
                num_traces = len(labels_list)
                if num_traces > 10:
                    # Many traces: use multiple columns and smaller font
                    self.axes.legend(ncol=3, fontsize=8, loc='best')
                elif num_traces > 6:
                    # Medium number: use 2 columns
                    self.axes.legend(ncol=2, fontsize=9, loc='best')
                else:
                    # Few traces: single column, normal font
                    self.axes.legend()

        # Set labels and title
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        if title:
            self.axes.set_title(title)
        else:
            self.axes.set_title("PL Spectra") # Default title

        self.axes.set_yscale(y_scale) # Apply the specified scale
        if getattr(self, '_minor_ticks_pref', False):
            try:
                self.axes.minorticks_on()
            except Exception:
                pass
        if xlim:
            try:
                self.axes.set_xlim(xlim)
            except Exception:
                pass
        # Grid visibility
        show_grid = True
        if style_options and 'show_grid' in style_options:
            show_grid = bool(style_options['show_grid'])
        self.axes.grid(show_grid, linestyle='--', alpha=0.6) # Add a grid
        self.figure.tight_layout()
        self.canvas.draw() # Redraw the canvas
        logger.info(f"Plotted {len(energy_ev_list)} dataset(s) with y_scale='{y_scale}'.")

    def plot_stacked_data(self, energy_ev_list, counts_list, labels_list=None, title=None, y_scale='linear',
                          offset=1.0, y_label=None, color_cycle=None, line_width=1.2, xlim=None,
                          style_options=None, colors=None, stack_positions=None, alphas=None):
        """
        Clears the current plot and plots new data with a vertical offset.
        """
        # Reset to a single-axes figure to avoid leftover grids
        # Preserve figure size before clearing
        current_size = self.figure.get_size_inches()
        self.figure.clear()
        self.figure.set_size_inches(current_size)
        self.axes = self.figure.add_subplot(111)
        self._remove_colorbars()

        # Apply style before plotting
        self.apply_style(style_options)

        if not energy_ev_list or not counts_list or len(energy_ev_list) != len(counts_list):
            logger.warning("Invalid data provided for plotting. Plotting empty.")
            self.axes.plot([], [])
        else:
            num_plots = len(energy_ev_list)
            use_labels = labels_list is not None and len(labels_list) == num_plots

            # Optional color cycle
            if not colors and color_cycle:
                try:
                    cmap = cm.get_cmap(color_cycle)
                    colors = getattr(cmap, 'colors', None)
                    if colors is None:
                        colors = [cmap(i/10.0) for i in range(10)]
                    self.axes.set_prop_cycle(color=colors)
                except Exception:
                    pass

            try:
                self.set_last_lines_data(energy_ev_list, counts_list)
            except Exception:
                pass

            for i in range(num_plots):
                label = labels_list[i] if use_labels else None
                try:
                    # Add offset to the counts
                    stack_idx = stack_positions[i] if stack_positions and i < len(stack_positions) else i
                    offset_counts = counts_list[i] + (stack_idx * offset)
                    color = None
                    if colors and i < len(colors):
                        color = colors[i]
                    plot_kwargs = {'linewidth': line_width}
                    if label:
                        plot_kwargs['label'] = label
                    if color:
                        plot_kwargs['color'] = color
                    if alphas and i < len(alphas):
                        plot_kwargs['alpha'] = alphas[i]
                    self.axes.plot(energy_ev_list[i], offset_counts, **plot_kwargs)
                except Exception as e:
                    logger.error(f"Error plotting dataset {i}: {e}", exc_info=True)

            if use_labels and any(labels_list):
                # Optimize legend for many traces
                num_traces = len(labels_list)
                if num_traces > 10:
                    # Many traces: use multiple columns and smaller font
                    self.axes.legend(ncol=3, fontsize=8, loc='best')
                elif num_traces > 6:
                    # Medium number: use 2 columns
                    self.axes.legend(ncol=2, fontsize=9, loc='best')
                else:
                    # Few traces: single column, normal font
                    self.axes.legend()

        self.axes.set_xlabel("Energy (eV)")
        if y_label is None:
            y_label = "Normalized Intensity (arb. units)"
        self.axes.set_ylabel(y_label)
        if title:
            self.axes.set_title(title)
        else:
            self.axes.set_title("Stacked PL Spectra")

        self.axes.set_yscale(y_scale)
        if getattr(self, '_minor_ticks_pref', False):
            try:
                self.axes.minorticks_on()
            except Exception:
                pass
        if xlim:
            try:
                self.axes.set_xlim(xlim)
            except Exception:
                pass
        show_grid = True
        if style_options and 'show_grid' in style_options:
            show_grid = bool(style_options['show_grid'])
        self.axes.grid(show_grid, linestyle='--', alpha=0.6)
        self.figure.tight_layout()
        self.canvas.draw()
        logger.info(f"Plotted {len(energy_ev_list)} stacked dataset(s) with offset={offset} and y_scale='{y_scale}'.")

    def save_current_figure(self, filepath, dpi=None, transparent=False):
        """Saves the current figure to the given filepath (png, pdf, svg)."""
        try:
            self.figure.savefig(filepath, dpi=dpi or self.figure.dpi, bbox_inches='tight', transparent=transparent)
            logger.info(f"Figure saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save figure: {e}", exc_info=True)
            return False

    def overlay_lines(self, energy_ev_list, counts_list, labels_list=None, color='gray', alpha=0.7, linestyle='--'):
        """Overlay additional line sets on current axes (for comparison)."""
        if not energy_ev_list or not counts_list:
            return
        use_labels = labels_list is not None and len(labels_list) == len(energy_ev_list)
        for i in range(len(energy_ev_list)):
            label = labels_list[i] if use_labels else None
            try:
                self.axes.plot(energy_ev_list[i], counts_list[i], linestyle=linestyle, color=color, alpha=alpha, label=label)
            except Exception as e:
                logger.warning(f"Overlay error: {e}")
        self.figure.subplots_adjust(left=0.12, right=0.94, bottom=0.12, top=0.92)
        self.canvas.draw_idle()

    # ---- Spike overlay and interaction ----
    def overlay_spikes(self, energy_ev_list, counts_list, spikes_indices_list, selected_list=None):
        """Overlay spike markers on the current axes for each curve.

        spikes_indices_list: list of index arrays per curve
        selected_list: list of boolean arrays per curve (same length as indices) marking active selection
        """
        if not hasattr(self, '_spike_overlays'):
            self._spike_overlays = []
        # Clear previous overlays if any
        self.clear_spike_overlay()
        self._spike_indices = []
        self._spike_selected = []

        for i, idxs in enumerate(spikes_indices_list):
            idxs = np.asarray(idxs, dtype=int)
            if idxs.size == 0:
                self._spike_overlays.append(None)
                self._spike_indices.append(np.array([], dtype=int))
                self._spike_selected.append(np.array([], dtype=bool))
                continue
            x = np.asarray(energy_ev_list[i])[idxs]
            y = np.asarray(counts_list[i])[idxs]
            sel = np.ones_like(idxs, dtype=bool) if selected_list is None else np.asarray(selected_list[i], dtype=bool)
            colors = np.where(sel, 'red', 'gray')
            sc = self.axes.scatter(x, y, s=60, marker='x', c=colors, picker=8)
            self._spike_overlays.append(sc)
            self._spike_indices.append(idxs)
            self._spike_selected.append(sel)

        # Connect pick event
        self._spike_pick_cid = self.canvas.mpl_connect('pick_event', self._on_spike_pick)
        self.canvas.draw_idle()

    def _on_spike_pick(self, event):
        art = event.artist
        if not hasattr(self, '_spike_overlays') or art not in self._spike_overlays:
            return
        i_curve = self._spike_overlays.index(art)
        inds = event.ind
        if inds is None:
            return
        sel = self._spike_selected[i_curve]
        # Toggle selected state for picked points
        for j in inds:
            if 0 <= j < sel.size:
                sel[j] = not sel[j]
        # Refresh colors
        colors = np.where(sel, 'red', 'gray')
        art.set_color(colors)
        self.canvas.draw_idle()

    def enable_manual_spike_add(self, enable=True):
        """Enable left-click to add a spike marker at the nearest point on the nearest curve."""
        if enable:
            self._click_add_cid = self.canvas.mpl_connect('button_press_event', self._on_click_add_spike)
        else:
            if hasattr(self, '_click_add_cid'):
                self.canvas.mpl_disconnect(self._click_add_cid)
                self._click_add_cid = None

    def _on_click_add_spike(self, event):
        if event.inaxes != self.axes:
            return
        # Find nearest curve by x distance
        if not hasattr(self, '_last_lines_data'):
            return
        x0 = event.xdata; y0 = event.ydata
        best = None
        for i, (xs, ys) in enumerate(self._last_lines_data):
            xs = np.asarray(xs); ys = np.asarray(ys)
            j = int(np.argmin(np.abs(xs - x0)))
            dist = (xs[j] - x0)**2 + (ys[j] - y0)**2
            if (best is None) or (dist < best[0]):
                best = (dist, i, j)
        if best is None:
            return
        _, i_curve, j_idx = best
        # Append spike index if not present
        if i_curve >= len(self._spike_indices):
            return
        idxs = self._spike_indices[i_curve]
        if j_idx in idxs:
            # if already present, toggle selection to True
            k = int(np.where(idxs == j_idx)[0][0])
            self._spike_selected[i_curve][k] = True
        else:
            # extend arrays
            self._spike_indices[i_curve] = np.append(idxs, j_idx)
            self._spike_selected[i_curve] = np.append(self._spike_selected[i_curve], True)
        # Repaint overlay for this curve
        xs = np.asarray(self._last_lines_data[i_curve][0])
        ys = np.asarray(self._last_lines_data[i_curve][1])
        sel = self._spike_selected[i_curve]
        idxs = self._spike_indices[i_curve]
        if self._spike_overlays[i_curve] is not None:
            self._spike_overlays[i_curve].remove()
        colors = np.where(sel, 'red', 'gray')
        sc = self.axes.scatter(xs[idxs], ys[idxs], s=30, marker='x', c=colors, picker=True)
        self._spike_overlays[i_curve] = sc
        self.canvas.draw_idle()

    def set_last_lines_data(self, energy_ev_list, counts_list):
        """Store last plotted lines' raw data to support spike interactions."""
        self._last_lines_data = list(zip(energy_ev_list, counts_list))

    def get_selected_spikes(self):
        if not hasattr(self, '_spike_indices'):
            return []
        res = []
        for idxs, sel in zip(self._spike_indices, self._spike_selected):
            if len(idxs) == 0:
                res.append([])
            else:
                res.append(list(np.asarray(idxs)[np.asarray(sel)]))
        return res

    def clear_spike_overlay(self):
        if hasattr(self, '_spike_overlays') and self._spike_overlays:
            for sc in self._spike_overlays:
                if sc is not None:
                    try:
                        sc.remove()
                    except Exception:
                        pass
        self._spike_overlays = []
        if hasattr(self, '_spike_pick_cid') and self._spike_pick_cid:
            self.canvas.mpl_disconnect(self._spike_pick_cid)
            self._spike_pick_cid = None
        # reset indices/selection
        self._spike_indices = []
        self._spike_selected = []
        self.canvas.draw_idle()

    def plot_line_grid(self, subplots_data, titles, cols=4, y_scale='linear', figsize=(12, 8), dpi=None,
                       font_size=None, line_width=1.0, color_cycle='tab10', x_label="Energy (eV)",
                       y_label="Intensity (arb. units)", suptitle=None, xlim=None,
                       legend_mode='per-axes', legend_font=None, legend_ncol=1,
                       wspace=0.3, hspace=0.3, hide_inner_labels=True, tick_font_size=None,
                       style_options=None, title_mode='axes', in_axes_pos=(0.02, 0.95)):
        """
        Plots multiple line subplots in a grid.

        subplots_data: list of dicts with keys:
            - energies: list[np.ndarray]
            - counts: list[np.ndarray]
            - labels: list[str] (optional)
        titles: list[str] per subplot
        """
        try:
            import math
            import matplotlib
            from matplotlib import cm
            import numpy as np

            self.figure.clear()
            if dpi:
                self.figure.set_dpi(dpi)
            if figsize:
                self.figure.set_size_inches(figsize[0], figsize[1])

            # Apply style
            if style_options is None:
                style_options = {}
            if font_size is not None:
                style_options = {**style_options, 'font_size': font_size}
            self.apply_style(style_options)

            n = len(subplots_data)
            rows = math.ceil(n / cols) if cols > 0 else 1
            axes = self.figure.subplots(rows, cols, squeeze=False)
            try:
                self.figure.subplots_adjust(wspace=wspace, hspace=hspace)
            except Exception:
                pass

            # Build color cycle
            try:
                cmap = cm.get_cmap(color_cycle)
                colors = getattr(cmap, 'colors', None)
                if colors is None:
                    # sample 10 colors from the colormap
                    colors = [cmap(i/10.0) for i in range(10)]
            except Exception:
                colors = None

            for idx in range(rows * cols):
                r = idx // cols
                c = idx % cols
                ax = axes[r][c]
                if idx >= n:
                    ax.axis('off')
                    continue
                data = subplots_data[idx]
                energies_list = data.get('energies', [])
                counts_list = data.get('counts', [])
                labels_list = data.get('labels', [])

                if colors is not None:
                    ax.set_prop_cycle(color=colors)

                for i in range(len(energies_list)):
                    lbl = labels_list[i] if labels_list and i < len(labels_list) else None
                    try:
                        ax.plot(energies_list[i], counts_list[i], linewidth=line_width, label=lbl)
                    except Exception as e:
                        logger.error(f"Error plotting subplot {idx} dataset {i}: {e}")

                # Title render mode
                if title_mode == 'axes':
                    ax.set_title(titles[idx])
                elif title_mode == 'in-axes':
                    try:
                        ax.text(in_axes_pos[0], in_axes_pos[1], titles[idx], transform=ax.transAxes, ha='left', va='top')
                    except Exception:
                        ax.set_title(titles[idx])
                # Fallback: none -> no title
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                if tick_font_size is not None:
                    try:
                        ax.tick_params(axis='both', labelsize=tick_font_size)
                    except Exception:
                        pass
                try:
                    ax.set_yscale(y_scale)
                except Exception:
                    pass
                show_grid = True
                if style_options and 'show_grid' in style_options:
                    show_grid = bool(style_options['show_grid'])
                ax.grid(show_grid, linestyle='--', alpha=0.4)
                if labels_list and any(labels_list):
                    if legend_mode == 'per-axes' or (legend_mode == 'last-only' and idx == n-1):
                        ax.legend(fontsize=(legend_font or max(6, (font_size or 10) * 0.8)), ncol=legend_ncol)
                if getattr(self, '_minor_ticks_pref', False):
                    try:
                        ax.minorticks_on()
                    except Exception:
                        pass
                if xlim:
                    try:
                        ax.set_xlim(xlim)
                    except Exception:
                        pass

            # Optionally hide inner axis labels to reduce clutter
            if hide_inner_labels:
                for r in range(rows):
                    for c in range(cols):
                        idx = r*cols + c
                        if idx >= n:
                            continue
                        ax = axes[r][c]
                        if r < rows-1:
                            ax.set_xlabel("")
                        if c > 0:
                            ax.set_ylabel("")

            # Shared legend outside the grid
            if legend_mode == 'outside-right':
                # collect handles/labels from first non-empty axes
                handles, labels = [], []
                for idx in range(n):
                    r = idx // cols; c = idx % cols
                    ax = axes[r][c]
                    h, l = ax.get_legend_handles_labels()
                    if h:
                        handles, labels = h, l
                        break
                if handles:
                    self.figure.legend(handles, labels, loc='center left', bbox_to_anchor=(0.99, 0.5), fontsize=(legend_font or max(6, (font_size or 10) * 0.8)), ncol=legend_ncol)
                    # make room on the right
                    self.figure.tight_layout(rect=[0, 0, 0.85, 1])

            if suptitle:
                self.figure.suptitle(suptitle)
            self.figure.tight_layout()
            self.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Error plotting line grid: {e}", exc_info=True)

    def plot_intensity_map_grid(self, maps_data, titles, cols=4, figsize=(12, 8), dpi=None, font_size=None,
                                cmap='viridis', log_c=False, log_y=False, vmin=None, vmax=None, x_label="Energy (eV)",
                                y_label="Power (uW)", suptitle=None, xlim=None):
        """
        Plots multiple intensity maps in a grid with a shared colorbar.

        maps_data: list of dicts with keys:
            - x: 1D array of energies (common axis for each subplot)
            - y: 1D array (powers)
            - z: 2D array (n_series x n_energy)
        """
        try:
            import math
            import matplotlib
            from matplotlib.colors import LogNorm
            import numpy as np

            self.figure.clear()
            if dpi:
                self.figure.set_dpi(dpi)
            if figsize:
                self.figure.set_size_inches(figsize[0], figsize[1])
            if font_size is not None:
                matplotlib.rcParams['font.size'] = font_size

            # Determine global vmin/vmax if not provided
            if vmin is None or vmax is None:
                z_all = np.concatenate([m['z'].ravel() for m in maps_data])
                z_all = z_all[np.isfinite(z_all)]
                if vmin is None:
                    vmin = float(np.nanmin(z_all)) if z_all.size else 0.0
                if vmax is None:
                    vmax = float(np.nanmax(z_all)) if z_all.size else 1.0

            n = len(maps_data)
            cols = max(1, cols)
            rows = math.ceil(n / cols)
            axes = self.figure.subplots(rows, cols, squeeze=False)
            try:
                self.figure.subplots_adjust(wspace=wspace, hspace=hspace)
            except Exception:
                pass
            mappable = None

            for idx in range(rows * cols):
                r = idx // cols
                c = idx % cols
                ax = axes[r][c]
                if idx >= n:
                    ax.axis('off')
                    continue
                m = maps_data[idx]
                x = m['x']
                y = m['y']
                z = m['z']

                if log_y:
                    try:
                        ax.set_yscale('log')
                    except Exception:
                        pass

                if log_c:
                    z_plot = np.array(z, dtype=float)
                    z_plot[z_plot <= 0] = np.nan
                    norm = LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
                    quad = ax.pcolormesh(x, y, z_plot, shading='auto', cmap=cmap, norm=norm)
                else:
                    quad = ax.pcolormesh(x, y, z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                mappable = quad
                ax.set_title(titles[idx])
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                if tick_font_size is not None:
                    try:
                        ax.tick_params(axis='both', labelsize=tick_font_size)
                    except Exception:
                        pass
                if xlim:
                    ax.set_xlim(xlim)

            if hide_inner_labels:
                for r in range(rows):
                    for c in range(cols):
                        idx = r*cols + c
                        if idx >= n:
                            continue
                        ax = axes[r][c]
                        if r < rows-1:
                            ax.set_xlabel("")
                        if c > 0:
                            ax.set_ylabel("")

            if mappable is not None:
                cbar = self.figure.colorbar(mappable, ax=axes.ravel().tolist(), label='Intensity (arb. units)')
            if suptitle:
                self.figure.suptitle(suptitle)
            self.figure.tight_layout()
            self.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Error plotting intensity map grid: {e}", exc_info=True)

    def clear_plot(self, y_scale='linear'):
        """Clears the plot area."""
        self.axes.clear()
        self._remove_colorbars()
        self.axes.plot([], []) # Plot empty data
        self.axes.set_xlabel("Energy (eV)")
        self.axes.set_ylabel("Intensity (arb. units)")
        self.axes.set_title("PL Spectra")
        self.axes.set_yscale(y_scale) # Apply the specified scale on clear
        self.axes.grid(True, linestyle='--', alpha=0.6)
        self.figure.subplots_adjust(left=0.12, right=0.94, bottom=0.12, top=0.92)
        self.canvas.draw()
        logger.info(f"Plot cleared with y_scale='{y_scale}'.")

    def plot_intensity_map(
        self,
        x,
        y,
        z,
        xlabel,
        ylabel,
        title,
        log_c=False,
        log_y=False,
        cmap='viridis',
        vmin=None,
        vmax=None,
        xlim=None,
        colorbar_label="Intensity (arb. units)",
        info_text: Optional[str] = None
    ):
        """
        Plots a 2D intensity map.
        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self._remove_colorbars()

        if log_y:
            self.axes.set_yscale('log')
        else:
            self.axes.set_yscale('linear')

        z_plot = z.copy()
        if log_c:
            z_plot[z_plot <= 0] = np.nan
        
        if vmin is None:
            vmin = np.nanmin(z_plot)
        if vmax is None:
            vmax = np.nanmax(z_plot)

        if log_c:
            if vmin <= 0:
                vmin = 1e-6
            norm = LogNorm(vmin=vmin, vmax=vmax)
            c = self.axes.pcolormesh(x, y, z_plot, shading='auto', cmap=cmap, norm=norm)
        else:
            norm = None
            c = self.axes.pcolormesh(x, y, z_plot, shading='auto', cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        self.colorbar = self.figure.colorbar(c, ax=self.axes)
        if colorbar_label:
            self.colorbar.set_label(colorbar_label)
        
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_title(title)

        if xlim:
            self.axes.set_xlim(xlim)
        
        if info_text:
            self._info_artist = self.axes.text(
                0.02,
                0.98,
                info_text,
                transform=self.axes.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def plot_rgb_intensity_map(
        self,
        energy_axis,
        y_axis,
        rgb_data,
        xlabel,
        ylabel,
        title,
        xlim=None,
        log_y=False,
        style_options=None,
        show_sigma_plus_bar=False,
        show_sigma_minus_bar=False,
        sigma_plus_norm: Optional[Normalize] = None,
        sigma_minus_norm: Optional[Normalize] = None,
        sigma_plus_cmap: str = "Reds",
        sigma_minus_cmap: str = "Blues",
        show_info_box: bool = False,
        info_text: Optional[str] = None,
        info_position: str = "Top Left (inside)",
        sigma_plus_label: str = "σ+ Intensity",
        sigma_minus_label: str = "σ- Intensity"
    ):
        """
        Plot a dual-colour intensity map using an RGB image.
        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self._remove_colorbars()

        self.apply_style(style_options)

        energy_axis = np.asarray(energy_axis, dtype=float)
        y_axis = np.asarray(y_axis, dtype=float)
        rgb = np.asarray(rgb_data, dtype=float)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB intensity map requires an array of shape (rows, cols, 3).")

        sort_idx = np.argsort(y_axis)
        y_axis_sorted = y_axis[sort_idx]
        rgb_sorted = rgb[sort_idx]

        if log_y:
            try:
                self.axes.set_yscale('log')
            except Exception:
                self.axes.set_yscale('linear')
        else:
            self.axes.set_yscale('linear')

        xmin, xmax = self._calculate_extent_edges(energy_axis)
        ymin, ymax = self._calculate_extent_edges(y_axis_sorted)
        extent = [xmin, xmax, ymin, ymax]
        self.axes.imshow(rgb_sorted, aspect='auto', origin='lower', extent=extent)

        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if title:
            self.axes.set_title(title)

        if xlim:
            try:
                self.axes.set_xlim(xlim)
            except Exception:
                pass

        show_grid = True
        if style_options and 'show_grid' in style_options:
            show_grid = bool(style_options['show_grid'])
        if show_grid:
            self.axes.grid(True, linestyle='--', alpha=0.3)
        else:
            self.axes.grid(False)

        self._extra_axes = []
        bar_width = 0.022
        pad = 0.04
        needs_colorbars = show_sigma_plus_bar or show_sigma_minus_bar
        if needs_colorbars:
            main_rect = [0.12, 0.12, 0.70, 0.78]
        else:
            main_rect = [0.12, 0.12, 0.82, 0.78]
        self.axes.set_position(main_rect)

        if needs_colorbars:
            base_x = main_rect[0] + main_rect[2] + pad
            plus_bottom = main_rect[1] + main_rect[3] * 0.47
            minus_bottom = main_rect[1]
            if show_sigma_plus_bar:
                if sigma_plus_norm is None:
                    sigma_plus_norm = Normalize(vmin=0.0, vmax=1.0)
                if hasattr(sigma_plus_norm, 'vmin') and hasattr(sigma_plus_norm, 'vmax') and sigma_plus_norm.vmax <= sigma_plus_norm.vmin:
                    sigma_plus_norm.vmax = sigma_plus_norm.vmin + max(abs(sigma_plus_norm.vmin), 1.0) * 1e-6
                cax_plus = self.figure.add_axes([base_x, plus_bottom, bar_width, main_rect[3] * 0.45])
                try:
                    cmap_plus_obj = cm.get_cmap(sigma_plus_cmap) if isinstance(sigma_plus_cmap, str) else sigma_plus_cmap
                except Exception:
                    cmap_plus_obj = cm.get_cmap('Reds')
                ColorbarBase(cax_plus, cmap=cmap_plus_obj, norm=sigma_plus_norm)
                cax_plus.set_ylabel(sigma_plus_label, rotation=270, labelpad=12)
                self._extra_axes.append(cax_plus)
            if show_sigma_minus_bar:
                if sigma_minus_norm is None:
                    sigma_minus_norm = Normalize(vmin=0.0, vmax=1.0)
                if hasattr(sigma_minus_norm, 'vmin') and hasattr(sigma_minus_norm, 'vmax') and sigma_minus_norm.vmax <= sigma_minus_norm.vmin:
                    sigma_minus_norm.vmax = sigma_minus_norm.vmin + max(abs(sigma_minus_norm.vmin), 1.0) * 1e-6
                cax_minus = self.figure.add_axes([base_x, minus_bottom, bar_width, main_rect[3] * 0.45])
                try:
                    cmap_minus_obj = cm.get_cmap(sigma_minus_cmap) if isinstance(sigma_minus_cmap, str) else sigma_minus_cmap
                except Exception:
                    cmap_minus_obj = cm.get_cmap('Blues')
                ColorbarBase(cax_minus, cmap=cmap_minus_obj, norm=sigma_minus_norm)
                cax_minus.set_ylabel(sigma_minus_label, rotation=270, labelpad=12)
                self._extra_axes.append(cax_minus)

        if show_info_box and info_text:
            pos = (info_position or "").lower()
            if pos.startswith("outside"):
                self._info_artist = self.figure.text(
                    0.98,
                    0.95,
                    info_text,
                    transform=self.figure.transFigure,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
                )
            else:
                x = 0.02
                ha = 'left'
                if 'right' in pos:
                    x = 0.98
                    ha = 'right'
                self._info_artist = self.axes.text(
                    x,
                    0.98,
                    info_text,
                    transform=self.axes.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment=ha,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

        self.canvas.draw_idle()

    @staticmethod
    def _calculate_extent_edges(axis_values):
        axis = np.asarray(axis_values, dtype=float)
        if axis.size == 0:
            return (-0.5, 0.5)
        if axis.size == 1:
            step = abs(axis[0]) * 0.1 if axis[0] != 0 else 1.0
            return (axis[0] - step, axis[0] + step)
        diffs = np.diff(axis)
        start = axis[0] - diffs[0] / 2.0
        end = axis[-1] + diffs[-1] / 2.0
        return (start, end)

# Example usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running plot_canvas.py directly for testing...")

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("PlotCanvas Test")
    root.geometry("700x600")

    plot_frame = PlotCanvas(master=root)
    plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Example plotting after a delay
    def test_plot():
        # Style
        self.apply_style(style_options)
        import numpy as np
        print("Testing plot_data...")
        energies1 = np.linspace(1.5, 2.5, 100)
        counts1 = np.exp(-(energies1 - 1.8)**2 / 0.1) * 1000 + np.random.rand(100) * 50
        energies2 = np.linspace(1.5, 2.5, 100)
        counts2 = np.exp(-(energies2 - 2.1)**2 / 0.05) * 800 + np.random.rand(100) * 40

        plot_frame.plot_data(
            [energies1, energies2],
            [counts1, counts2],
            labels_list=["Sample A", "Sample B"],
            title="Test Spectra"
        )

    # Example clearing after another delay
    def test_clear():
        print("Testing clear_plot...")
        plot_frame.clear_plot()

    root.after(2000, test_plot) # Plot after 2 seconds
    root.after(5000, test_clear) # Clear after 5 seconds

    root.mainloop()
