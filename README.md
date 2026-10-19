# HarnessPro project page

Live site: https://pzs19.github.io/HarnessPro/

This branch contains the static website, figures, fonts, and manuscript PDF.
The public site is deployed from the `HarnessPro/` directory on the `main`
branch of [`pzs19/pzs19.github.io`](https://github.com/pzs19/pzs19.github.io/tree/main/HarnessPro).
This branch keeps a copy of the website in the research repository. No frontend
build step is required.

The research code is on the [`main`](https://github.com/pzs19/HarnessPro/tree/main) branch.

To preview locally:

```sh
python3 -m http.server 8000
```

When updating the page, keep `index.html`, `styles.css`, `app.js`, and `assets/`
together at the branch root, then copy those files into the public repository's
`HarnessPro/` directory and push its `main` branch. Keep asset links relative so
the website works under `/HarnessPro/`. Font license notices are included in
`assets/fonts/`.
