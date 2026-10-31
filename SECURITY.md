# Security policy

## Supported versions

CueMem is pre-1.0. Security fixes go to the latest `main` and to the 0.1.x line; older commits are
not patched.

| Version | Supported |
| --- | --- |
| latest `main` | yes |
| 0.1.x | yes |
| older | no |

## Reporting a vulnerability

Report vulnerabilities privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability**, or go straight to
<https://github.com/FlickYan/cuemem/security/advisories/new>. This is the only reporting channel;
there is no security e-mail address.

- Never open a public issue, pull request or discussion for a suspected vulnerability.
- Never attach secrets or personal data: no `.env` file, no API tokens or provider keys
  (`CUEMEM_API_TOKEN`, scoped keys, `*_API_KEY`), no keys file (`CUEMEM_API_KEYS_FILE`), and nothing
  from `data/` or from `runs/` (traces contain conversation text). Use placeholder values and
  redact anything copied from logs.
- Helpful details: the commit or version, how CueMem runs (`cuemem serve`, Docker,
  `docker compose`, `cuemem proxy`, `cuemem mcp`), the bind address and authentication mode (admin
  token, scoped keys or `--insecure-no-auth`), and the steps to reproduce.

You will get an acknowledgement within 7 days. The fix and the disclosure date are then agreed in
the private advisory, and the published advisory credits you unless you ask otherwise.

## Scope

In scope:

- `cuemem serve`: the HTTP API (`/api`, `/api/v1`), its authentication (the admin token and scoped
  keys) and the memory browser UI;
- MCP over HTTP (`/mcp` in `cuemem serve`) and `cuemem mcp` (stdio);
- the OpenAI-compatible memory proxy (`cuemem proxy`, and `/v1` in `cuemem serve`);
- import and export (`cuemem import` / `export`, `/api/spaces/import`,
  `/api/spaces/{space}/export`, `/api/v1/import`, `/api/v1/export`);
- forgetting: data that can still be read after `forget`, `delete`, `delete_all` or `reset`
  (residue in the database, vector store, caches, traces or stored images);
- the Docker image and `docker-compose.yml` as shipped.

Known and accepted limits are listed in the decision records
([D-070](docs/en/decisions.md#d-070-the-third-audit-round-a-fresh-seven-lens-audit-22-verified-findings-fixed-evaluation-requests-and-published-numbers-unchanged),
"Left for later or partial", item 4, and "Remaining limits", plus later rounds). A report showing
that one of them is worse than described, for example that it crosses tenants, is still welcome.

Out of scope: deployments that switch authentication off (`--insecure-no-auth`) outside a trusted
network, and advisories in third-party packages that CueMem does not make exploitable (report those
upstream; Dependabot and the dependency audit below track them).

## For maintainers

The repository files do part of the work: `.github/dependabot.yml` (weekly version updates for the
GitHub Actions, `uv.lock` and the Dockerfile's base image) and the `依赖审计` workflow
(`.github/workflows/audit.yml`: `pip-audit` on the locked dependencies on every push and pull
request and weekly; deliberately not a required check, since new advisories appear without code
changes). The rest are repository settings that only an admin can change:

1. **Settings → Advanced Security** (all free for public repositories): enable **Private
   vulnerability reporting**, **Dependabot alerts** and **Dependabot security updates**, and confirm
   that **Dependency graph**, **Secret scanning** and **Push protection** are on. From a shell:

   ```sh
   gh api -X PUT repos/FlickYan/cuemem/private-vulnerability-reporting
   gh api -X PUT repos/FlickYan/cuemem/vulnerability-alerts
   gh api -X PUT repos/FlickYan/cuemem/automated-security-fixes
   gh api -X PATCH repos/FlickYan/cuemem \
     -f 'security_and_analysis[secret_scanning][status]=enabled' \
     -f 'security_and_analysis[secret_scanning_push_protection][status]=enabled'
   gh api repos/FlickYan/cuemem/private-vulnerability-reporting   # {"enabled":true}
   ```

2. **Protect `main`** by importing `.github/rulesets/protect-main.json`: **Settings → Rules →
   Rulesets → New ruleset → Import a ruleset**, pick the file and save; or

   ```sh
   gh api -X POST repos/FlickYan/cuemem/rulesets --input .github/rulesets/protect-main.json
   gh api repos/FlickYan/cuemem/rules/branches/main
   ```

   The second command must list `deletion`, `non_fast_forward`, `pull_request` and
   `required_status_checks`. The ruleset forbids deleting and force-pushing the default branch,
   requires a pull request (no approval needed, review threads resolved) and requires the `test`,
   `package`, `docker` and `qdrant-server` jobs of `.github/workflows/ci.yml`, as reported by GitHub
   Actions (integration 15368). The external-baseline and dependency-audit workflows are not
   required. It has no bypass actors, so hotfixes also go through a pull request with green checks.
   Renaming one of those four jobs means editing the ruleset too.
