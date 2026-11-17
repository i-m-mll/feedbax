# Dataset rendering

`render` accepts a validated Figure document and a Dataset for each named role. The renderer resolves the authored channel bindings and selectors, groups samples in Dataset coordinate order, and assembles the Plotly figure. The authored theme applies to that figure without modifying the process default.

::: feedbax.plot.dataset_figure.render
