#PL-temperature_and_power_Analyzer
Adesktopapplicationforanalyzingphotoluminescence(PL)spectroscopydata,featuringpower-andtemperature-dependentseriesanalysis.

PLAnalyzerisadesktopapplicationdesignedfortheanalysisofphotoluminescence(PL)spectroscopydata.Itprovidesagraphicaluserinterfacetostreamlinetheprocessofloading,processing,visualizing,andanalyzingtemperature-andpower-dependentPLspectra,whichiscommoninphysicsandmaterialsscienceresearch.

![PLAnalyzerScreenshot](https://github.com/nikodemsokolowski/PL-temperature_and_power_Analyzer/blob/main/fig1.png?raw=true)

Thecoreanalysisfeatureallowsforrobustpower-lawfittingonalog-logscaletodetermineexcitonrecombinationmechanisms.

![PowerLawFittingScreenshot](https://github.com/nikodemsokolowski/PL-temperature_and_power_Analyzer/blob/main/fig2.png?raw=true)

##🔬CoreFeatures

***AutomaticMetadataParsing**:Extractsexperimentalparameters(temperature,laserpower,acquisitiontime)directlyfromfilenames,eliminatingmanualdataentry.
***BatchDataProcessing**:Applyprocessingstepstoallloadedfilessimultaneously.
***TimeNormalization**:Normalizesignalcountsbyacquisitiontime(counts/s).
***GreyFilterCorrection**:Rescaledatafromfilestakenwithaneutraldensity(grey)filter.
***SpectrometerResponseCorrection**:Applypredefinedcorrectionfactorsbasedonacquisitiontime.
***InteractiveVisualization**:
*Plotindividualspectraorcomparemultipledatasets.
*Automaticallyplotentire**power-dependent**or**temperature-dependent**serieswithasingleclick.
*Generatea**2Dintensitymap**oftemperature-dependentdata.
*Togglebetweenlinearand**logarithmicscales**fortheY-axistoeasilyviewfeaturesatdifferentintensitylevels.
*Standardplotcontrols(zoom,pan,save)viatheMatplotlibtoolbar.
***QuantitativeAnalysis**:
***SpectralIntegration**:Calculatetheintegratedintensityofselectedspectraoverauser-definedenergyrange.
***PowerDependenceAnalysis**:Theprimaryanalysisworkflow.Automaticallyintegrateafullpowerseriesoveraspecifiedrange,plottheintegratedintensityvs.laserpowerona**log-logplot**,andfitthedatatoapowerlawmodel($I=a\cdotP^k$)todeterminetheexponent`k`.
***TemperatureDependenceAnalysis**:Fittemperature-dependentdatatotheArrheniusmodeltoextractactivationenergies.
***DataExport**:
*Exportprocessedspectra(Energyvs.Counts)toatab-separatedfile.
*Exportintegratedintensityvs.powerdataforaspecificseries,readyforexternalplottingoranalysis.
***SessionPersistence**:
*Theapplicationautomaticallysavesthelistofloadedfilesandreloadsthemonthenextstartup.



### Magnetic Field Enhancements (Phase 3)

- Magnetic sweep settings dialog supports multiple acquisition time ranges per B-field interval and a sweep-direction selector; configuration persists to `config.json` and is applied during filename parsing.
- Parser honours both low->high and high->low sweeps while assigning acquisition times from the configured ranges.
- Temperature Dependence Analysis window now provides dual plots (Arrhenius + intensity vs. T), show/hide toggles, styling controls, in-window integration range editing, and 600 DPI export (PNG/PDF/SVG).
- Polarization dropdown automatically resets to “All Data” when magnetic data is disabled, keeping the UI in sync with available features.

### Magnetic Field Enhancements (Phase 2 Module 2)

- Enhanced integrate-vs-B-field window with simultaneous sigma+, sigma-, and sum traces, optional DCP subplot, styling controls, and CSV export.
- Advanced g-factor analysis window featuring energy-vs-field plots, Zeeman splitting fit with diagnostics, peak detection modes, smoothing, manual overlays, residuals, and export options.
- Virtual sigma+ + sigma- sum datasets that behave like normal datasets while tracking their source spectra and supporting refresh.

##🚀GettingStarted&Installation

Therearetwowaystousethissoftware.Thedirectdownloadisrecommendedformostusers.

###Option1:DirectDownload(Recommended)
ThisistheeasiestwaytoruntheapplicationwithoutinstallingPythonoranydependencies.

1.Gotothe`dist/run_app`folderinthisrepository.
2.Downloadthe`run_app.exe`file.
3.Double-clickthedownloadedfiletorunthePLAnalyzer.

###Option2:RunfromSourceCode
Thisoptionisfordevelopersoruserswhowanttomodifythecode.

**Prerequisites:**
*Python3.8ornewer
*Git

**InstallationSteps:**
1.**Clonetherepository:**
```bash
gitclone[https://github.com/nikodemsokolowski/PL-temperature_and_power_Analyzer.git](https://github.com/nikodemsokolowski/PL-temperature_and_power_Analyzer.git)
cdPL-temperature_and_power_Analyzer
```

2.**Createandactivateavirtualenvironment:**
```bash
#OnWindows
python-mvenvvenv
.\venv\Scripts\activate

#OnmacOS/Linux
python3-mvenvvenv
sourcevenv/bin/activate
```

3.**Installtherequiredpackages:**
```bash
pipinstall-rrequirements.txt
```

4.**Runtheapplication:**
```bash
pythonrun_app.py
```

---

##📖HowtoUse

###1.FileNamingConvention

Fortheautomaticmetadataparsingtowork,yourdatafiles**must**followaspecificnamingconvention.Theparserisflexiblebutexpectsparameterstobedelimitedbyunderscores(`_`).**Decimalsmustbeindicatedwitha'p'**.

**Template:**`Prefix_<Temperature>K_<Power>uW_<Time>s_GF<optional>.csv`

**ComponentBreakdown:**
***Prefix**:Anytextthatdoesnotcontaintheparameterkeys(e.g.,`WSe2_SampleA`,`Data`).
*`<Temperature>K`:Thetemperaturefollowedby`K`.Fordecimals,use'p'(e.g.,`5p5K`for5.5K).
*`<Power>uW`:Thelaserpowerfollowedby`uW`.Fordecimals,use'p'(e.g.,`0p1uW`for0.1uW).
*`<Time>s`:Theacquisitiontimefollowedby`s`.Fordecimals,use'p'(e.g.,`0p5s`for0.5s).
*`GF`:Anoptionaltagindicatingagreyfilterwasused.Thenumberfollowing`GF`isignored;itspresenceiswhatmatters.

**Examplesofvalidfilenames:**
*`WSe2_H5_K6_run1_5K_10uW_0p1s.csv`
*`Data_100p5K_50uW_1s_GF1.csv`
*`measurement_20K_0p2uW_2p5s.dat`

###2.Workflow

1.**LoadData**:Clickthe**"LoadFiles"**buttonandselectallthedatafilesforyourexperiment.Theywillappearinthetableontheleft.
2.**ProcessData**:Inthe"ProcessingSteps"panel,applycorrectionsasneeded.It'srecommendedtoapplytheminorder:
1.**NormalizebyTime**:Togetcountspersecond.
2.**RescalebyGFFactor**:Ifyouusedagreyfilter.Adialogwillaskforthetransmissionfactor(e.g.,0.1fora10%filter).
3.**CorrectSpectrometerResponse**:Ifyouhaveknowncorrectionfactorsforyoursetup.
4.**PlotSpectra**:
*Selectoneormorefilesinthetableandclick**"PlotSelected"**.
*Toseeafullseries,selectanyfilefromthatseriesandclick**"PlotPowerSeries"**or**"PlotTempSeries"**.
*Usethe**"LogY-Axis"**checkboxtotogglethescale.
5.**AnalyzePowerDependence**:
1.Enterthe**MinE(eV)**and**MaxE(eV)**foryourpeakofinterest.
2.Select**one**filefromthetemperatureseriesyouwanttoanalyze.
3.Click**"PowerDependenceAnalysis"**.
4.Anewwindowwillopenshowingthe**integratedintensityvs.power**onalog-logplot.Usethe"FitPowerLaw"buttoninthisnewwindowtoperformthefitandextracttheexponent`k`.

# PL-temperature_and_power_Analyzer
## What's New (Plotting & Publication Tools)

- Equalized temperature series with stacked plotting and annotated scale factors (x N).
- Ultimate Plot Grid: power series at all temperatures as a grid of line plots or intensity maps.
- Plot Range (eV) for x-axis; applies to single, stacked/equalized and grid plots.
- Expanded color palettes: viridis, magma, inferno, cividis, twilight, Spectral, coolwarm, RdYlBu, Blues, Reds, etc.
- Legend controls: per-axes, outside-right, last-only, none; font size and columns.
- Figure Options: columns, size, DPI, line width, font size, color cycle.
- Style presets and themes: ticks inside, minor ticks, major/minor tick lengths, axes linewidth; presets (Compact/Nature/APS/Science) and themes (Nature dense, APS two-col, ACS single-col).
- Subplot title template (e.g., `{T:.0f} K`) with mode (axes/in-axes) and position.
- Show Grid toggle to hide dashed grid lines.
- Export current plot to PNG/PDF/SVG with DPI and optional transparent background.
- Save/Load figure settings (JSON).

### Spike Removal (Advanced)

- Adaptive detector (rolling median + rolling MAD), default ON; robust to NaNs.
- Detect range (eV) and Skip removal range (eV) to target/avoid regions.
- Multi-pixel spikes captured via contiguous-segment detection (≤ Max Width).
- Manual selection expansion via Manual Radius (±N points around clicks).
- Removal methods: interpolation, local median, NaN, neighbor mean (±N).
- One-click Auto Remove (with current parameters), multi-level Undo.
- Review navigation (Prev/Next/Exit) to inspect one curve at a time; “Clean This Curve” + per-curve revert to original.
- “Show Original” overlays originals (dashed orange) on top of cleaned plots for quick comparison.
- Sensitivity Sweep (σ) and Prominence threshold removal for batch cleanups.
- Save/Load spike settings (JSON).
