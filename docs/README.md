# Writing documentation

The documentation source is organized by locale under `docs/source/<locale>/`.
English is the default source language, and each translated page keeps the same
relative path as its English counterpart.

```text
docs/
├── README.md
├── openapi.json
├── site.yml
└── source/
    ├── en/
    │   ├── _toctree.yml
    │   └── *.md
    └── zh_cn/
        ├── _toctree.yml
        └── *.md
```

## Add or update a page

1. Update the English page in `docs/source/en/`.
2. Add the page to `docs/source/en/_toctree.yml` if it is new.
3. Update the matching page and navigation entry under `docs/source/zh_cn/`.
4. Keep filenames and relative paths aligned across locales so links and website
   routes remain stable.

Use GitHub-style alerts such as `> [!NOTE]`, `> [!TIP]`, and `> [!WARNING]`.
Keep deployment commands reproducible and avoid committing credentials, private
endpoints, or machine-specific absolute paths.

## API reference

`docs/openapi.json` is generated from the FastAPI application. Do not edit it by
hand. After changing routes or schemas, run:

```bash
python scripts/export_openapi.py
python scripts/export_openapi.py --check
```

The website reads `docs/site.yml`, publishes the Markdown sources for each
locale, and generates the API reference from this schema.
