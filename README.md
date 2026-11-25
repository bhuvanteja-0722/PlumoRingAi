# PlumoRing AI

An interactive product landing page and ring-architecture concept, with a model workspace based on the telemetry layout your teammate is developing.

Open `index.html` in a modern browser or serve the `outputs` folder locally. The ring viewer uses Three.js and the page uses Anime.js from public CDNs, so the first load needs an internet connection.

## What is on the page

- A full-size 3D assembly in three finishes: Gold / Aurelia, Onyx / Nocturne, and Doomsday / Latverian Armor. The selected finish changes the page palette, studio halo, ring finish, and Three.js lighting together.
- Ring interaction: drag to orbit, use the wheel while over the ring to open or close the assembly, Shift + wheel or the zoom controls to zoom, and select any part for its role. The explode slider, play button, and E shortcut also control the assembly.
- A twelve-part representative architecture, including red/infrared optical PPG, a 3-axis accelerometer, NTC temperature, EDA electrodes, flex circuit, optical seal, liner, battery, charging contacts, and optional charging coil.
- An input-stream dashboard with eight adjustable demo channels that match the companion-system reference: heart rate, oxygen saturation, respiration rate, an acoustic feature, skin-temperature delta, IMU motion, skin conductance, and ambient particles. Presets and reset controls only change illustrative values.
- A model-output workspace that stays in a waiting state. It does not produce disease predictions from demo data. It is ready to be connected when the trained model, input schema, and validation materials are available.

The four ring sensor systems are PPG, motion, skin temperature, and EDA. Acoustic and ambient-particle data are shown as optional companion inputs, not ring hardware. Model channels are representative UI fields; the final schema must come from the model owner.

## Supporting pages and device layouts

- `settings.html` lets visitors choose the Gold, Onyx, or Doomsday theme and set motion to follow their device, stay fully animated, or use reduced motion. Both preferences are stored only in browser local storage and carry into the ring viewer.
- `privacy.html` explains the current prototype data behavior, external asset requests, and the fact that the dashboard is not connected to a real model or health-data service.
- `404.html` is the branded not-found page for static hosting such as GitHub Pages.
- `responsive.css` adapts the ring viewer, assembly controls, part list, and companion dashboard for tablets and phones, including narrow 320 px screens. Touch input remains available for rotating and exploring the model.
- `pages.css` styles the settings, privacy, and not-found pages using the same Gold, Onyx, and Doomsday palettes.

The frontend does not request or submit personal or health data through its current interactions. Theme and motion choices are stored in local browser storage; dashboard sliders and presets only change sample values in the open page. See [Privacy](privacy.html) for the current frontend behavior.

## Accuracy boundary

This is a concept viewer, not production CAD. Ring dimensions, materials, battery, charging method, sensor selection, and sensor placement are not confirmed. The optical, thermistor, accelerometer, and EDA geometry use common public examples as proportion references only. The pulse trace and telemetry values are illustrative and do not represent a person or a clinical result.

## Design references

The page borrows interaction patterns and layout ideas from the supplied 3D guide, supplied dashboard reel, and the following public component libraries. It does not copy template artwork or use screenshots as product imagery.

- [GetLayers templates / Artefakt](https://www.getlayers.ai/templates?layer=artefakt) — visual direction for immersive presentation and editorial pacing.
- [Motion Primitives](https://motion-primitives.com/docs/in-view) — in-view reveals and restrained staggered entrances, implemented with browser-native observation for this static page.
- [Watermelon UI Landing Page 01](https://ui.watermelon.sh/template/landing-01) — its dark product-page rhythm, animated entrances, dot-grid atmosphere, and bento feature layout were adapted around the ring and telemetry instead of adding an unrelated generic template demo.
- [Kokonut UI](https://kokonutui.com/) — compact status, metric, and AI-workspace patterns.
- [Bklit UI](https://ui.bklit.com/) — telemetry chart and data-visualization references, drawn here as custom SVG.
- [Anime.js](https://animejs.com/documentation/getting-started/installation) — progressive hero entrance motion via its documented CDN ESM import.

### Ring sensor proportion references

- [Analog Devices MAX86141 optical AFE](https://www.analog.com/en/products/MAX86141.html) — representative optical front-end package reference.
- [ST LIS2DW12 3-axis accelerometer](https://www.st.com/content/st_com/en/products/mems-and-sensors/accelerometers/lis2dw12.html) — representative 2 × 2 × 0.7 mm LGA package reference.
- [TDK 0402 NTC thermistor data sheet](https://www.tdk-electronics.tdk.com/inf/50/db/ntc/NTC_SMD_Standard_series_0402-0803.pdf) — common small thermistor body dimensions.
- [Multi-sensor wearable EDA study](https://www.mdpi.com/1424-8220/21/10/3499) — reference for finger-contact EDA electrode geometry.
