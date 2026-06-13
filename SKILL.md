---
name: update-priority-authors-secret
description: Update the arXiv newsletter priority author list in local .env and mirror it to the GitHub Actions PRIORITY_AUTHORS repository secret.
---

# Update Priority Authors

1. Inspect `.env` for the `PRIORITY_AUTHORS` variable without printing unrelated secrets.
2. Add the requested author to the semicolon-delimited `PRIORITY_AUTHORS` value, preserving existing names.
3. Set the GitHub Actions repository secret from the updated local value:

```bash
value=$(awk 'BEGIN{FS="="} $1=="PRIORITY_AUTHORS" {sub(/^[^=]*=/, ""); print; exit}' .env)
printf '%s' "$value" | gh secret set PRIORITY_AUTHORS --repo hideaki-j/arxiv-newsletter
```

4. Verify only the secret metadata, not the secret value:

```bash
gh secret list --repo hideaki-j/arxiv-newsletter | rg '^PRIORITY_AUTHORS\b'
```

Never paste API keys, app passwords, or full `.env` contents into the response.
