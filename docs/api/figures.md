# Figure documents

A Figure document binds prepared Dataset variables to line and band layers. It declares named Dataset roles, panel positions, layer encodings, and a Plotly theme. In a saved Analyze view, bind each role to the producing element and product. The view can contain a Figure document inline or reference a named Figure document in the project.

An installed package can declare a template that takes typed settings and returns a Figure document. Select a template with, for example, `{"template":"feedbax.VelocitySummary","settings":{"checkpoint_index":0}}` in the view's `figure` field or in the referenced Figure document's payload. Feedbax validates the settings and lowers the template to the same Render request as an equivalent inline figure. The Studio authoring catalog exposes each template's settings schema and provider.

Render reads the bound Dataset products and stores a single Plotly JSON artifact. A Report document can reference the saved figure product. The managed project Plan exposes artifact URLs for both outputs.

::: feedbax.contracts.figure_document.FigureDocument
