# DepositProof AI

A local Gradio application for organizing rental-deposit evidence. It keeps cases,
uploaded files, photo pairs, deduction links, and audit findings in SQLite and
generates a downloadable PDF report.

The UI opens on a dedicated welcome screen and then guides the user through
four stages: audit case, evidence, review findings, and final report. The main
navigation stays at the top so earlier information can be revisited without
discarding entered values.

This is an evidence-review aid, not legal advice. A visible photo difference does
not prove damage, responsibility, or repair cost. Missing evidence is not proof
that a claim is false.

## Current features and limits

- Create cases and enter each landlord deduction separately.
- Extract text from text-based PDFs with PyMuPDF. Scanned PDFs are reported as
  unreadable; OCR is not installed or promised.
- Extract invoice totals, dates, and descriptions with transparent Python
  heuristics. Check them against the original document.
- Retrieve agreement passages using the optional
  `sentence-transformers/all-MiniLM-L6-v2` embedding model. Returned quotations
  are copied from extracted text and verified; page references are best-effort.
  Without the optional package/model, a labelled keyword search is used.
- Upload a move-in / move-out image pair, validate and resize it locally, and
  optionally send both images to a hosted vision-language model.
- Link that photo pair and an invoice to a specific deduction. Audits use the
  linked evidence, inspection-report text, and agreement passages.
- Include stored photo-comparison conclusions in the final audit report. Linked
  comparisons appear with their deduction; unlinked comparisons appear in a
  separate section and are not attributed to any claim.
- Run a conservative rules-based assessment even when hosted AI is unavailable.
  Hosted language-model responses can explain or downgrade a rules result, but
  cannot override a confirmed contradiction or promote an unsupported claim.
- Store findings in SQLite and generate a PDF containing classifications,
  per-deduction assessment modes, verified agreement passages, invoice details,
  and separate move-in / move-out photo-comparison observations where available.
- Compare a deduction against the root-level
  `DepositProof_10Metros_50Localities_Repair_Rates-v3 (1).csv` in the
  **Repair Price Intelligence** dashboard. Matching and range calculations use
  city, locality, category, and the listed minimum/maximum. Leaving locality
  blank reports a city-wide range aggregated across its listed localities.
  These records are sample data, not verified local quotations. A saved
  comparison is added to the next audit report as a separate context-only
  section and never changes the audit classification.
- Optionally ask the configured hosted language model for one short explanation
  of an already-calculated benchmark. This is off by default, may use API quota,
  cannot change the values, and is omitted when the request fails. The
  deterministic comparison remains available without hosted AI.

By default, hosted inference uses one [`gemini-3.1-flash-lite`](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
model for both image comparison and text assessment. Google documents that model
for text and image inputs, and its OpenAI-compatible endpoint supports the
chat-completion request format used by this app. The app keeps the same image
limits and uses the same model ID for both tasks; these operations still make
separate requests.

Google lists free input and output for eligible models, but free access is
limited to certain models and has project-specific rate limits. Check your
project's active [AI Studio rate limits](https://aistudio.google.com/rate-limit)
before the demo; availability and quota are not guaranteed. Google says content
submitted through the free tier may be used to improve its products, so use
synthetic or redacted demo documents and photos unless that is acceptable.
The mocked test suite does not make a live API call and cannot verify your key,
quota, or current model access.

Hugging Face remains an optional provider. Set `LLM_PROVIDER=huggingface`,
configure `HF_TOKEN`, and choose its models with `HF_VLM_MODEL` and
`HF_LLM_MODEL` to use the existing Hugging Face router integration instead.

## Windows PowerShell setup

Run these commands from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` in VS Code and put your Google AI Studio API key after
`GEMINI_API_KEY=`. Keep `LLM_PROVIDER=google` and the model setting
`GEMINI_MODEL=gemini-3.1-flash-lite`. Create a key in
[Google AI Studio](https://aistudio.google.com/apikey). Never paste the key
into chat, screenshots, source files, or `README.md`; `.env` is excluded by
`.gitignore`. The current Hugging Face token, if present, is not used when
`LLM_PROVIDER=google` and is not replaced by a Google key.

Start the application:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open the local Gradio URL printed in the terminal. If you do not configure a
provider key, the application still runs with the rules-based fallback; photo
files are saved but not described by an AI model.

### Optional semantic retrieval

The application does not download an embedding model as part of its basic
installation. To enable local semantic search, install the optional package:

```powershell
.\.venv\Scripts\python.exe -m pip install sentence-transformers
```

On the first audit that searches agreement clauses, the package downloads
`all-MiniLM-L6-v2` model files (roughly 90 MB, plus package dependencies). No
GPU is required. If that package or model cannot load, the application labels
and uses its keyword fallback instead. This affects local agreement-clause
retrieval only; it does not prevent Gemini inference or the rules-based audit.

## Try an audit

1. Create a case and enter the deposit and the landlord-stated total.
2. Continue to **Evidence**. Upload a text-based rental agreement PDF, relevant invoice PDFs, and any
   move-in / move-out inspection PDFs.
3. Add each claimed deduction. Choose its invoice if available.
4. Add a matched move-in / move-out pair. Link it to the right claim under
   **Itemised deductions and evidence links**. For a deduction already entered,
   use **Attach evidence to an existing deduction**.
5. Optionally compare a repair claim against sample rates.
6. Continue to **Review findings**, run the audit, and inspect its explanations,
   missing/conflicting evidence, photo limitations, and generated report.
   **Final report** displays the report and its PDF download.

To use **Repair Price Intelligence**, select a category, enter the city and
(optionally) a locality from the 10-metro dataset, and enter the claimed amount.
An exact locality comparison uses that locality's category row; leaving
locality blank aggregates the category ranges across the city's listed
localities and labels the broader comparison. If no matching sample records
exist, the dashboard reports that no estimate was created. To place a
comparison in an audit report, refresh and select the saved deduction; the
comparison amount must match the stored claim and the case currency must be
INR. Run the audit again to generate the report with the separate benchmark
section. The dashboard does not verify prices or establish whether a deduction
is legally recoverable.

Use images and documents that you own or have permission to use. Synthetic or
redacted files are recommended for a public hackathon demo. A scanned PDF needs
OCR before its text can contribute evidence.

## Tests

The focused tests use local fixtures and mocked model requests; they do not make
paid or external inference calls:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover rules-based classification, exact-quote verification, malformed
image-model output, hosted request construction, photo-to-deduction linking,
audit reruns, deterministic sample-rate matching, benchmark report separation,
and mocked hosted explanation failure. They do not prove that your token,
account credits, or a live provider request will work.

If a PDF upload reports `pymupdf_unavailable`, the PDF library could not load its
native backend. This can happen if Windows security policy blocks the installed
binary; use an approved Python environment or ask your administrator rather than
trying to bypass that policy.

## Local storage and privacy

SQLite data is stored at `data\depositproof.sqlite3`. Uploaded files are stored
under `uploads\`, and generated reports under `reports\`. Do not deploy this
demo publicly with real rental documents without adding authentication,
authorization, retention/deletion controls, and secure storage.
