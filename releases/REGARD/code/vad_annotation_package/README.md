# Human annotation interface

The Streamlit interface collects VAD ratings for pre-assigned response items.
Its research inputs and anonymized outputs are distributed in the OSF archive.

To deploy it, place `annotation_items.jsonl` and `assignments.csv` in `data/`,
copy `data/users.example.csv` to the ignored file `data/users.csv`, and replace
every placeholder password with a unique credential. Then run:

```bash
poetry install --with human
poetry run streamlit run vad_annotation_package/app.py
```

Never commit `users.csv`. The released annotation identifiers (`ann01`, etc.)
are pseudonyms and are not a mapping to annotator identities.
