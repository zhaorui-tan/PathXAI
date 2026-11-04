# Publish the committed example artifacts with GitHub Pages

`PathXAI_tutorial/` is the standalone repository root. Its manual-only Pages workflow publishes the static bundle in `examples/`: the landing page, SO1 and SO4 reports, shared assets and completed notebooks. It does not run inference, build a notebook, or publish the rest of the repository.

**Deployment status: not deployed.** The GitHub repository is [`zhaorui-tan/PathXAI`](https://github.com/zhaorui-tan/PathXAI). Its known Pages URL is [https://zhaorui-tan.github.io/PathXAI/](https://zhaorui-tan.github.io/PathXAI/), but that URL must not be treated as live until a Pages deployment has completed successfully.

The notebook workflow remains local Jupyter. GitHub Pages only serves the already-created static examples; it is not a Colab, Docker or hosted-compute path.

## One-time repository setup

Prepare the reviewed source release using `python3 scripts/build_release.py --output-dir dist/release_for_upload` from the tutorial root. Extract `PathXAI_tutorial/` from the generated `PathXAI_tutorial_source.zip` and use its contents as the GitHub repository root, with `.github/`, `examples/` and `README.md` at the top level. The source archive uses an explicit allowlist and is authoritative for release contents. Do not upload the raw working folder: it can contain local workspace data and the separately fetched, ignored MedSAM3 runtime source.

After the first successful deployment, the expected default Pages URL is **https://zhaorui-tan.github.io/PathXAI/**, with reports at `SO1/report.html` and `SO4/report.html`. These are configured destinations, not evidence that a deployment has already happened.

1. Commit `.github/workflows/report-pages.yml` and the complete `examples/` bundle to the repository's default branch. The bundle must include `examples/index.html`, `examples/SO1/report.html`, `examples/SO4/report.html`, the shared `examples/assets/` directory, `examples/notebooks/SO1_completed.ipynb`, and `examples/notebooks/SO4_completed.ipynb`. Both report pages reference the shared assets with `../assets` paths.
2. On GitHub, open **Settings → Pages**. Under **Build and deployment**, set **Source** to **GitHub Actions** and save. This is the publishing source required by the [GitHub Pages custom-workflow instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Publish manually

1. Open the repository's **Actions** tab.
2. Select **Publish tutorial report examples**.
3. Select **Run workflow** on the default branch and confirm.
4. When the job succeeds, open its `github-pages` environment link or the `page_url` deployment output. Use that generated URL; the repository owner and name determine it, so this documentation does not guess a URL.
5. From the published landing page, open the SO1 or SO4 example. Both report pages reference the same `../assets/` files; the deploy artifact includes the complete `examples/` tree.

The workflow runs only on `workflow_dispatch`; a push to the repository does not publish or update the site. It checks the landing page, both report files, shared asset directory and the two named completed notebooks. It uploads only `examples/` and deploys that Pages artifact. The job uses `contents: read`, `pages: write`, and `id-token: write`, following GitHub's [deployment requirements](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages#deploying-github-pages-artifacts). It uses the official [`configure-pages`](https://github.com/actions/configure-pages), [`upload-pages-artifact`](https://github.com/actions/upload-pages-artifact), and [`deploy-pages`](https://github.com/actions/deploy-pages) actions.

## Viewing the examples

Opening an `.html` file from the GitHub repository's file browser or raw-file view does not run it as a web page. Use the deployed Pages URL to view each report in a browser. GitHub can preview committed `.ipynb` files in the repository; the Pages workflow also serves `SO1_completed.ipynb` and `SO4_completed.ipynb` from `examples/notebooks/`.

Before publishing, review the exact prepared source release and `examples/` contents for credentials, private files, or material you do not have permission to make public. The Pages workflow uploads `examples/` as-is and does not redact or transform files. Model weights, the TCGA source WSI, local workspace data and the external MedSAM3 runtime tree are outside the prepared source upload path.
