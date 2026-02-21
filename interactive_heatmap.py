import json
import webbrowser
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.colormap import linear
from shapely import wkt
from shapely.geometry import mapping

PROJECT_ROOT = Path(__file__).resolve().parent

# Load per-year CSVs
# -----------------------------------------------------------------------------
YEARS = list(range(2015, 2023))
paths = {y: (PROJECT_ROOT / "heatmaps" / f"heat_{y}.csv").as_posix() for y in YEARS}

gdfs = {}
for y in YEARS:
    df = pd.read_csv(paths[y])
    gdf = gpd.GeoDataFrame(df, geometry=df["wkt"].apply(wkt.loads), crs="EPSG:4326") # type: ignore
    count_col = "num_trips" if "num_trips" in gdf.columns else "num_tips"

    gdf["year"] = y

    # Preformat strings for tooltips
    gdf["avg_tip_s"] = gdf["avg_tip"].map(
        lambda v: f"${v:,.2f}" if pd.notnull(v) else "—"
    )
    gdf["med_tip_s"] = gdf["med_tip"].map(
        lambda v: f"${v:,.2f}" if pd.notnull(v) else "—"
    )
    gdf["avg_rate_s"] = gdf["avg_rate"].map(
        lambda v: f"{v:.1f}%" if pd.notnull(v) else "—"
    )
    gdf["num_trips_s"] = gdf[count_col].map(
        lambda v: f"{int(v):,}" if pd.notnull(v) else "—"
    )

    # Build HTML for the info card tooltip
    def build_tooltip(row):
        return f"""
        <div class="tipcard">
          <div class="tipcard-header">
            <div class="tipcard-title">{row['zone_name']}</div>
            <div class="tipcard-tag">{row['borough']}</div>
          </div>
          <div class="tipcard-subtitle">Year: {row['year']}</div>
          <div class="tipcard-metrics">
            <div><strong>Avg tip:</strong> {row['avg_tip_s']}</div>
            <div><strong>Median tip:</strong> {row['med_tip_s']}</div>
            <div><strong>Tip rate:</strong> {row['avg_rate_s']}</div>
            <div><strong>Trips:</strong> {row['num_trips_s']}</div>
          </div>
        </div>
        """

    gdf["tooltip_html"] = gdf.apply(build_tooltip, axis=1)
    gdfs[y] = gdf


# Global ranges per metric, each metric sits across all years
# -----------------------------------------------------------------------------
def robust_range(series, is_rate=False):
    arr = pd.Series(series).dropna().to_numpy()
    if arr.size == 0:
        return (0.0, 1.0)
    vmin, vmax = np.nanquantile(arr, [0.003, 0.997])
    if is_rate:
        vmin = max(0.0, vmin)
        vmax = max(vmin + 1.0, vmax)
    return (float(vmin), float(vmax))


metrics = ["avg_tip", "med_tip", "avg_rate"]
ranges = {}
for m in metrics:
    all_vals = pd.concat([gdfs[y][m] for y in YEARS], ignore_index=True)
    ranges[m] = robust_range(all_vals, is_rate=(m == "avg_rate")) # type: ignore


# Palettes and legends
# -----------------------------------------------------------------------------
PALETTES = {m: linear.RdYlGn_11.scale(*ranges[m]) for m in metrics} # type: ignore


def make_continuous_legend_html(caption, vmin, vmax, *, n_ticks=7, money_mode=True, palette=None):
    if palette is None:
        palette = linear.RdYlGn_11.scale(vmin, vmax) # type: ignore

    grad_vals = np.linspace(vmin, vmax, 64)
    grad_css = ", ".join(palette(v) for v in grad_vals)

    raw_ticks = np.linspace(vmin, vmax, n_ticks)
    if money_mode:
        ticks = np.round(raw_ticks * 2) / 2.0
        labels = [f"{t:.2f}" for t in ticks]
    else:
        ticks = np.round(raw_ticks).astype(int)
        labels = [f"{t:d}" for t in ticks]

    return f"""
    <div id="tip-legend" style="
        position: fixed;
        z-index: 9999;
        bottom: 20px; left: 20px;
        background: rgba(255,255,255,0.92);
        padding: 10px 12px;
        border: 1px solid #bbb;
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
      ">
      <div style="font-size: 18px; font-weight: 700; margin-bottom: 8px;">{caption}</div>
      <div style="
          width: 340px; height: 16px;
          background: linear-gradient(to right, {grad_css});
          border: 1px solid #999; border-radius: 2px;">
      </div>
      <div style="width: 340px; display: flex; justify-content: space-between; margin-top: 6px;">
        {''.join(f'<span style="font-size:16px; color:#111;">{lbl}</span>' for lbl in labels)}
      </div>
    </div>
    """


legend_for_metric = {
    "avg_tip": make_continuous_legend_html(
        "Average Tip ($)", *ranges["avg_tip"], n_ticks=7, money_mode=True, palette=PALETTES["avg_tip"]
    ),
    "med_tip": make_continuous_legend_html(
        "Median Tip ($)", *ranges["med_tip"], n_ticks=7, money_mode=True, palette=PALETTES["med_tip"]
    ),
    "avg_rate": make_continuous_legend_html(
        "Average Tip Rate (% of Fare)", *ranges["avg_rate"], n_ticks=7, money_mode=False, palette=PALETTES["avg_rate"]
    ),
}


# Color function (sync with legend)
def color_for(metric, val):
    if val is None or pd.isna(val):
        return "#cccccc"
    return PALETTES[metric](val)


# Default metric/year and precomputed colors
# -----------------------------------------------------------------------------
default_metric = "avg_tip"
default_year = YEARS[0]

for y in YEARS:
    gdf = gdfs[y]
    for mtr in metrics:
        gdf[f"color_{mtr}"] = gdf[mtr].apply(lambda v: color_for(mtr, v))
    gdf["cur_color"] = gdf[f"color_{default_metric}"]


# Build the map
# -----------------------------------------------------------------------------
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles=None)
folium.TileLayer("cartodbpositron", control=False).add_to(m)

YEAR_TO_VARNAME = {}

# Create one feature group per year to not overload with polygons
for y in YEARS:
    gdf = gdfs[y]
    fg = folium.FeatureGroup(name=f"year_{y}", show=(y == default_year))

    def style_fn(feature):
        props = feature["properties"]
        fill = props.get("cur_color", "#cccccc")
        if fill is None:
            fill = "#cccccc"
        return {
            "fillOpacity": 0.7,
            "weight": 0.2,
            "color": "#666",
            "fillColor": fill,
        }

    for _, row in gdf.iterrows():
        feature = {
            "type": "Feature",
            "geometry": mapping(row["geometry"]),
            "properties": {
                "color_avg_tip": row["color_avg_tip"],
                "color_med_tip": row["color_med_tip"],
                "color_avg_rate": row["color_avg_rate"],
                "cur_color": row["cur_color"],
                "tooltip_html": row["tooltip_html"],
            },
        }

        gj = folium.GeoJson(
            data=feature,
            style_function=style_fn,
            show=True,
        )
        folium.Tooltip(row["tooltip_html"], sticky=False).add_to(gj)
        gj.add_to(fg)

    fg.add_to(m)
    YEAR_TO_VARNAME[str(y)] = fg.get_name()

# Initial legend for default metric
m.get_root().html.add_child(folium.Element(legend_for_metric[default_metric])) # type: ignore

# Radio panels (Metric and Years)
# -----------------------------------------------------------------------------
controls_html = """
<div id="control-stack" style="
  position: fixed;
  top: 58px;
  right: 12px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 10px;
">
  <!-- Metric panel -->
  <div class="control-box">
    <div class="control-title">Metric</div>
    <label class="control-item"><input type="radio" name="metric_sel" value="avg_tip" checked> Average Tip ($)</label>
    <label class="control-item"><input type="radio" name="metric_sel" value="med_tip"> Median Tip ($)</label>
    <label class="control-item"><input type="radio" name="metric_sel" value="avg_rate"> Average Tip Rate (%)</label>
  </div>

  <!-- Years panel -->
  <div class="control-box">
    <div class="control-title">Years</div>
    {year_radios}
  </div>
</div>

<style>
  .control-box{{
    background: rgba(255,255,255,0.95);
    padding: 10px 12px;
    border: 1px solid #bbb;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    font-size: 14px;
    min-width: 160px;
  }}
  .control-title{{
    font-weight: 700;
    font-size: 16px;
    text-align: center;
    margin-bottom: 6px;
  }}
  .control-item{{ display:block; margin: 2px 0; white-space: nowrap; }}

  /* Tooltip info card styling */
  .tipcard {{
    background: rgba(255, 255, 255, 0.97);
    border-radius: 6px;
    padding: 8px 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    min-width: 220px;
    max-width: 280px;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    color: #222;
  }}
  .tipcard-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 4px;
  }}
  .tipcard-title {{
    font-weight: 700;
    font-size: 14px;
    flex: 1;
  }}
  .tipcard-tag {{
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 10px;
    background: #f0f0f0;
    border: 1px solid #ddd;
    white-space: nowrap;
  }}
  .tipcard-subtitle {{
    font-size: 11px;
    color: #555;
    margin-bottom: 4px;
  }}
  .tipcard-metrics div {{
    margin: 1px 0;
  }}
</style>
"""

year_radios = "".join(
    f'<label class="control-item"><input type="radio" name="year_sel" value="{y}" {"checked" if y == default_year else ""}> {y}</label>'
    for y in YEARS
)

m.get_root().html.add_child(folium.Element(controls_html.format(year_radios=year_radios))) # type: ignore


# Help button and overlay
# -----------------------------------------------------------------------------
help_html = """
<div id="help-button" title="How to use this map">?</div>

<div id="help-overlay" style="display:none;">
  <div id="help-backdrop"></div>
  <div id="help-box">
    <div id="help-header">
      <span id="help-title">How to read this map</span>
      <button id="help-close" aria-label="Close help">&times;</button>
    </div>
    <div id="help-body">
      <p><strong>What this map shows</strong><br>
      This map displays how taxi tipping behavior varies across New York City taxi zones from 2015–2022.
      You can explore three different tipping metrics and switch between years to see how patterns change over time.</p>

      <p><strong>Colors</strong><br>
      Green zones indicate relatively higher values for the selected metric.<br>
      Yellow, orange, and red zones indicate relatively lower values.<br>
      Gray zones represent areas with limited or missing data.</p>

      <p><strong>Controls</strong><br>
      <em>Metric:</em> choose whether you want to view average tip amount, median tip amount, or tip rate (tip as a percent of the fare).<br>
      <em>Years:</em> select which calendar year’s data you want to display on the map.</p>

      <p><strong>How to interact</strong><br>
      • Hover over any zone to see its detailed statistics in the info card.<br>
      • Use the metric selector to compare the same zone across different tipping measures.<br>
      • Use the year selector to see how patterns shift across 2015–2022.<br>
      • Pan and zoom like any other online map to focus on specific neighborhoods or boroughs.</p>
    </div>
    <div id="help-footer">
      <button id="help-ok">Got it</button>
    </div>
  </div>
</div>

<style>
  /* Help button in bottom-right corner */
  #help-button {
    position: fixed;
    right: 18px;
    bottom: 18px;
    z-index: 9999;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #ffffff;
    border: 1px solid #bbb;
    box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #333;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }
  #help-button:hover {
    background: #f5f5f5;
  }

  /* Overlay + box */
  #help-overlay {
    position: fixed;
    inset: 0;
    z-index: 9998;
  }
  #help-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(0,0,0,0.35);
  }
  #help-box {
    position: absolute;
    max-width: 520px;
    width: 90%;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #222;
    padding: 12px 16px 10px 16px;
  }
  #help-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
  }
  #help-title {
    font-size: 16px;
    font-weight: 700;
  }
  #help-close {
    border: none;
    background: transparent;
    font-size: 20px;
    line-height: 1;
    cursor: pointer;
    padding: 0 4px;
  }
  #help-body {
    font-size: 13px;
    line-height: 1.4;
    max-height: 260px;
    overflow-y: auto;
  }
  #help-body p {
    margin: 6px 0;
  }
  #help-footer {
    display: flex;
    justify-content: flex-end;
    margin-top: 8px;
  }
  #help-ok {
    border-radius: 4px;
    border: 1px solid #0078d4;
    background: #0078d4;
    color: #ffffff;
    font-size: 13px;
    padding: 4px 10px;
    cursor: pointer;
  }
  #help-ok:hover {
    background: #0063b5;
  }
</style>
"""

m.get_root().html.add_child(folium.Element(help_html)) # type: ignore


# Main JS toggler; recolor by metrics, highlighting, etc
# -----------------------------------------------------------------------------
legend_js_map = json.dumps(legend_for_metric)
vars_js_map = json.dumps(YEAR_TO_VARNAME)
map_var_name = m.get_name()
init_metric = default_metric
init_year = str(default_year)

js = """
<script>
const LEGEND_BY_METRIC = __LEGEND__;
const YEAR_TO_VARNAME  = __VARS__;
const MAP_VAR_NAME     = "__MAPVAR__";

function removeLegend() {
  const el = document.getElementById('tip-legend');
  if (el) el.remove();
}
function addLegend(metric) {
  removeLegend();
  const wrap = document.createElement('div');
  wrap.innerHTML = LEGEND_BY_METRIC[metric];
  document.body.appendChild(wrap.firstElementChild);
}

function layerFromVar(varName) {
  return window[varName] || null;
}

function hideAllYears(mapObj) {
  for (const year in YEAR_TO_VARNAME) {
    const vname = YEAR_TO_VARNAME[year];
    const lyr   = layerFromVar(vname);
    if (lyr && mapObj.hasLayer(lyr)) {
      mapObj.removeLayer(lyr);
    }
  }
}

// Recolor a given year layer according to the chosen metric
function recolorLayerByMetric(lyr, metric) {
  const colorProp = "color_" + metric;

  function applyToLayer(l) {
    // If this layer has feature properties, recolor it
    if (l.feature && l.feature.properties) {
      const props = l.feature.properties;
      const fill  = props[colorProp] || "#cccccc";

      if (l.setStyle) {
        l.setStyle({ fillColor: fill });
      }
      props.cur_color = fill;
    }

    // If this layer has children, recurse into them
    if (l.eachLayer) {
      l.eachLayer(applyToLayer);
    }
  }

  if (!lyr) return;
  applyToLayer(lyr);
}

// Attach custom hover handlers
function attachHighlightHandlers(lyr) {
  if (!lyr || !lyr.eachLayer) return;

  lyr.eachLayer(function (featLayer) {
    if (featLayer._highlightBound) return;

    featLayer.on('mouseover', function(e) {
      this.setStyle({
        weight: 2,
        color: "#000",
        fillOpacity: 0.85
      });
    });

    featLayer.on('mouseout', function(e) {
      this.setStyle({
        weight: 0.2,
        color: "#666",
        fillOpacity: 0.7
      });
    });

    featLayer._highlightBound = true;
  });
}

function showYearAndMetric(mapObj, metric, year) {
  hideAllYears(mapObj);
  const vname = YEAR_TO_VARNAME[year];
  const lyr   = layerFromVar(vname);

  if (lyr) {
    mapObj.addLayer(lyr);
    recolorLayerByMetric(lyr, metric);
    attachHighlightHandlers(lyr);
    addLegend(metric);
  } else {
    console.error("Layer not found for year", year, "->", vname);
  }
}

function curMetric() {
  const radios = document.querySelectorAll("input[name='metric_sel']");
  for (const r of radios) {
    if (r.checked) return r.value;
  }
  return "__INIT_METRIC__";
}
function curYear() {
  const radios = document.querySelectorAll("input[name='year_sel']");
  for (const r of radios) {
    if (r.checked) return r.value;
  }
  return "__INIT_YEAR__";
}

function wireControls(mapObj) {
  const metricRadios = document.querySelectorAll("input[name='metric_sel']");
  const yearRadios   = document.querySelectorAll("input[name='year_sel']");

  const handler = () => showYearAndMetric(mapObj, curMetric(), curYear());

  metricRadios.forEach(r => r.addEventListener('change', handler));
  yearRadios.forEach(r   => r.addEventListener('change', handler));

  showYearAndMetric(mapObj, curMetric(), curYear());
}

window.addEventListener('load', () => {
  const mapObj = window[MAP_VAR_NAME];
  setTimeout(() => wireControls(mapObj), 0);
});
</script>
"""

js = (
    js.replace("__LEGEND__", legend_js_map)
      .replace("__VARS__", vars_js_map)
      .replace("__MAPVAR__", map_var_name)
      .replace("__INIT_METRIC__", init_metric)
      .replace("__INIT_YEAR__", init_year)
)

m.get_root().html.add_child(folium.Element(js)) # type: ignore


# Help overlay wiring
# -----------------------------------------------------------------------------
help_js = """
<script>
window.addEventListener('load', function() {
  const helpBtn    = document.getElementById('help-button');
  const helpOverlay = document.getElementById('help-overlay');
  const helpClose   = document.getElementById('help-close');
  const helpOk      = document.getElementById('help-ok');
  const helpBackdrop = document.getElementById('help-backdrop');

  if (!helpBtn || !helpOverlay) return;

  function openHelp() {
    helpOverlay.style.display = 'block';
  }
  function closeHelp() {
    helpOverlay.style.display = 'none';
  }

  helpBtn.addEventListener('click', openHelp);
  helpClose.addEventListener('click', closeHelp);
  helpOk.addEventListener('click', closeHelp);
  helpBackdrop.addEventListener('click', closeHelp);

  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      closeHelp();
    }
  });
});
</script>
"""

m.get_root().html.add_child(folium.Element(help_js)) # type: ignore


# Title
# -----------------------------------------------------------------------------
map_title = "New York City Yellow Taxi Tipping Trends (2015–2022)"
title_html = f"""
<div style="
    position: fixed;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    background: rgba(255, 255, 255, 0.9);
    padding: 8px 18px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    font-size: 22px;
    font-weight: 700;
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #222;
    letter-spacing: 0.5px;
">
    {map_title}
</div>
"""
m.get_root().html.add_child(folium.Element(title_html)) # type: ignore


# Save and open
# -----------------------------------------------------------------------------
out = str((PROJECT_ROOT / "tips_map_slim.html").resolve())
m.save(out)
webbrowser.open(f"file://{out}")
