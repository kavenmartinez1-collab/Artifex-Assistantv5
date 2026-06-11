# Local models

Drop GGUF files or model folders here and they appear in the in-app model
browser as `local/<name>`:

- **Loose GGUF**: `models/MyModel-Q4_K_M.gguf` → `local/MyModel-Q4_K_M`
- **Model folder**: `models/my-model/` containing a `.gguf` or a
  `config.json` + `*.safetensors`

To point at models that live elsewhere on disk, create
`model-dirs.local.json` next to `package.json` (gitignored) — see
`model-dirs.local.example.json`. The dev server also auto-discovers your
Ollama store and HuggingFace cache; nothing here is required.
