# Design QA — Ala de Papel

## Evidencia de comparación

- Fuente visual: `docs/images/pegasus-rag-redesign-target.png`.
- Implementación: `docs/images/pegasus-rag-redesign-final.png`.
- Comparación conjunta: `docs/images/pegasus-rag-design-comparison.png`.
- Vista móvil: `docs/images/pegasus-rag-redesign-mobile.png`.
- Estado comparado: pantalla inicial vacía, base documental lista y sin conversación.
- Viewport de escritorio: 1440 × 1024 CSS px, `deviceScaleFactor` 1.
- Fuente original: 1487 × 1058 px, normalizada a 1440 × 1024 px.
- Captura de implementación: 1440 × 1024 px.
- Viewport móvil: 390 × 844 CSS px, sin desbordamiento horizontal del contenido visible.

## Historial de comparación

### Iteración 1

- [P1] El título completo heredaba el azul de acento en vez de aplicarlo únicamente a
  “evidencia”. Se restringió el token de acento a `.accent`.
- [P1] El botón principal conservaba el estilo secundario de Streamlit. Se añadió un selector
  específico para `stFormSubmitButton` y se recuperó el peso visual del diseño.
- [P2] La marca lateral se veía demasiado pequeña por el margen interno de la imagen. El recurso
  se recortó y optimizó para conservar legibilidad a 38 px.
- [P2] Las preguntas sugeridas no mostraban la jerarquía editorial numerada. Se separaron número
  y acción, aumentando tamaño, ritmo y área interactiva.
- [P2] La primera ilustración tenía demasiadas plumas y se alejaba del ala plegada seleccionada.
  Se generó un recurso más compacto, de tres grandes facetas, y se uniformó el fondo para evitar
  un rectángulo visible.
- [P2] La barra lateral iniciaba abierta en móvil. Se cambió el estado inicial a adaptativo y se
  añadió espacio superior para no solapar el control de apertura.

### Iteración 2

- Evidencia posterior: `docs/images/pegasus-rag-redesign-final.png` y
  `docs/images/pegasus-rag-redesign-mobile.png`.
- No quedan hallazgos P0, P1 o P2.
- La fina repetición lateral observada al extremo derecho de la captura completa corresponde al
  compositor de capturas del navegador. El DOM reportó `scrollWidth === innerWidth` tanto en
  escritorio como en móvil, por lo que no existe desbordamiento horizontal de la aplicación.

## Superficies de fidelidad

- **Tipografía:** Manrope mantiene el carácter geométrico del diseño. El cuerpo se conserva entre
  14 y 17 px, con mayor contraste y altura de línea para textos auxiliares y de privacidad.
- **Espaciado y ritmo:** se preservan la barra lateral estrecha, el hero asimétrico, el compositor
  dominante y la relación en dos columnas entre sugerencias y evidencia.
- **Colores:** fondo tinta oscuro integral, texto marfil, azul ultramar como acción y coral reservado
  para citas. No se introdujeron gradientes de interfaz ni efectos de vidrio.
- **Imágenes:** la marca y el ala son recursos raster propios, nítidos y alineados con el concepto
  de papel plegado. No se sustituyeron con SVG, emoji ni arte construido con CSS.
- **Contenido:** se conserva el texto real del producto y los 317 fragmentos indexados. La muestra
  de cita inicial es deliberadamente genérica para no inventar documentos; las respuestas reales
  muestran archivo, ubicación, similitud y extracto.
- **Iconos y controles:** los iconos interactivos visibles proceden de los controles nativos de
  Streamlit/Material y mantienen áreas de interacción adecuadas.
- **Responsividad:** en 390 px el hero elimina la ilustración, la entrada y el botón se apilan, la
  barra lateral inicia cerrada y no hay desbordamiento horizontal del contenido principal.
- **Accesibilidad:** contraste alto, etiquetas semánticas, texto alternativo en la marca, controles
  de teclado nativos y foco visible de Streamlit.

## Interacciones verificadas

- Carga inicial con 317 fragmentos.
- Envío de “¿Cuántas aprobaciones necesita un Pull Request?”.
- Respuesta correcta con al menos dos aprobaciones y referencia a Fuente 2.
- Apertura del panel de cinco fuentes con documento, página, similitud y extracto.
- Estado vacío de escritorio y adaptación móvil.
- Consola del navegador sin errores ni advertencias en las capturas finales.

## Follow-up polish

- [P3] El mock muestra una cita documental concreta en el estado vacío; la implementación usa una
  explicación genérica para evitar presentar evidencia ficticia. Es una desviación intencional.
- [P3] Algunos textos internos del componente de carga permanecen en inglés porque pertenecen al
  widget nativo de Streamlit; no afectan el flujo ni la comprensión del límite y los formatos.

final result: passed
