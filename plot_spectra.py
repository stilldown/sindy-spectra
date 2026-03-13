import pandas as pd
import matplotlib.pyplot as plt

# read data
fn = r"d:\OneDrive\桌面\OPERA\2_corrected_final.csv"
df = pd.read_csv(fn)

# wavelengths columns start with numeric after c2
wl_cols = [c for c in df.columns if c not in ['Sample','c1','c2']]
wavelengths = [float(x) for x in wl_cols]

# create a scatter 3d or color-coded lines
fig, ax = plt.subplots(figsize=(8,6))
for idx,row in df.iterrows():
    intens = row[wl_cols].values.astype(float)
    ax.plot(wavelengths, intens, alpha=0.5)

ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('intensity')
ax.set_title('spectra')
fig.savefig('spectra_lines.png')
print('saved spectra_lines.png')

# color by c1
fig2,ax2=plt.subplots(figsize=(8,6))
norm = plt.Normalize(df['c1'].min(),df['c1'].max())
for idx,row in df.iterrows():
    intens = row[wl_cols].values.astype(float)
    ax2.plot(wavelengths,intens,color=plt.cm.viridis(norm(row['c1'])),alpha=0.6)
ax2.set_xlabel('wavelength')
ax2.set_ylabel('intensity')
ax2.set_title('colored by c1')
fig2.savefig('spectra_c1.png')
print('saved spectra_c1.png')

# scatter of intens at single wavelength vs c1,c2
w0 = wavelengths[0]
int0 = df[wl_cols].iloc[:,0].astype(float)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(df['c1'], df['c2'], int0)
ax3.set_xlabel('c1')
ax3.set_ylabel('c2')
ax3.set_zlabel(f'intensity at {w0}nm')
fig3.savefig('scatter3d.png')
print('saved scatter3d.png')

# ------------------------------------------------
# heatmap of all spectra (samples × wavelength)
heat = df[wl_cols].astype(float).values
fig4,ax4 = plt.subplots(figsize=(10,6))
im = ax4.imshow(heat, aspect='auto', cmap='viridis',
                extent=(wavelengths[0], wavelengths[-1], 0, heat.shape[0]))
ax4.set_xlabel('wavelength (nm)')
ax4.set_ylabel('sample index')
fig4.colorbar(im, ax=ax4, label='intensity')
ax4.set_title('heatmap of spectra (samples vs wavelength)')
fig4.savefig('spectra_heatmap.png')
print('saved spectra_heatmap.png')

# ------------------------------------------------
# optional interactive scatter with plotly if available
try:
    import plotly.express as px
    long = df.melt(id_vars=['Sample','c1','c2'], value_vars=wl_cols,
                   var_name='wavelength', value_name='intensity')
    long['wavelength'] = long['wavelength'].astype(float)
    fig5 = px.scatter(long, x='wavelength', y='intensity',
                      color='c1', size='c2', hover_data=['Sample'],
                      title='interactive spectrum points (size=c2)')
    fig5.write_html('interactive_spectra.html')
    print('saved interactive_spectra.html')

    # 3D scatter: two concentrations, wavelength, color by intensity
    fig6 = px.scatter_3d(long,  # all points
                         x='c1', y='c2', z='wavelength',
                         color='intensity',
                         color_continuous_scale='Viridis',
                         title='3D scatter (c1,c2,wavelength) colored by intensity')
    fig6.write_html('3d_scatter_color.html')
    print('saved 3d_scatter_color.html')

    # line-version: draw a separate line per sample, color by intensity
    import plotly.graph_objs as go
    fig7 = go.Figure()
    for samp,grp in long.groupby('Sample'):
        fig7.add_trace(go.Scatter3d(
            x=grp['c1'], y=grp['c2'], z=grp['wavelength'],
            mode='lines',
            line=dict(color=grp['intensity'], colorscale='Viridis', width=4),
            name=str(samp)
        ))
    fig7.update_layout(title='3D lines (per sample) colored by intensity',
                       scene=dict(xaxis_title='c1', yaxis_title='c2', zaxis_title='wavelength'))
    fig7.write_html('3d_lines_color.html')
    print('saved 3d_lines_color.html')
except ImportError:
    print('plotly not installed; skipping interactive plot')
