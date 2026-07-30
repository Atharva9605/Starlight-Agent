# Starlight AI-CRM Mailer — VPS deployment
#
# Domains (Caddy edge soft-reload only; never bind 80/443 here):
#   https://mailer.starlightlinearled.com       → starlight-mailer-web:80
#   https://api.mailer.starlightlinearled.com   → starlight-mailer-api:7860
#
# Stack path: `/opt/starlight/mailer`
# Compose project: `starlight-mailer`
# Network: external `deploy_default` (shared with Caddy)
#
# ## Automated deploy (GitHub Actions)
#
# `.github/workflows/deploy-vps.yml` on every push to `main` (also `workflow_dispatch`):
# 1. Builds & pushes API + web images to Docker Hub (`starlight-mailer-api` / `starlight-mailer-web`)
# 2. SSHes to the VPS, migrates from any legacy path, `scp`s `deploy/docker-compose.cicd.yml`
# 3. `docker compose pull && up -d`, waits for healthy api+web, prunes old images
#
# Required GitHub secrets (repo → Settings → Secrets → Actions):
#
# | Secret | Purpose |
# |--------|---------|
# | `DOCKERHUB_USERNAME` | Docker Hub namespace |
# | `DOCKERHUB_TOKEN` | Docker Hub access token |
# | `VPS_SSH_KEY` | Private SSH key for `root@` VPS |
# | `VPS_IP` | VPS host / IP |
# | `MAIL_USERNAME` | Gmail for deploy emails |
# | `MAIL_PASSWORD` | Gmail App Password |
# | `MAIL_TO` | Notification recipient(s) |
#
# App secrets (Azure, GSuite, JWT, DB) live only in the VPS `.env` — not in GitHub.
#
# ## One-time VPS prep
#
# ```bash
# # Caddy routes (soft-reload only). The shared edge sites dir lives on the host;
# # Starlight product files live under /opt/starlight/mailer.
# cp deploy/mailer-web.caddy deploy/mailer-api.caddy /opt/technoventor/edge/sites/
# docker exec mis-vps-caddy-1 caddy reload --config /etc/caddy/Caddyfile
#
# # Ensure secrets file exists (never commit this)
# test -f /opt/starlight/mailer/.env
# ```
#
# ## Manual deploy / rollback
#
# Build on VPS:
# ```bash
# cd /opt/starlight/mailer
# docker compose --env-file .env -f deploy/docker-compose.prod.yml up -d --build
# ```
#
# Pull a specific CI image tag:
# ```bash
# cd /opt/starlight/mailer
# APP_IMAGE_TAG=<git-sha> DOCKERHUB_USERNAME=<user> \
#   docker compose --env-file .env -f deploy/docker-compose.cicd.yml up -d
# ```
#
# ## Verify
#
# ```bash
# docker ps --filter name=starlight-mailer
# curl -fsS https://api.mailer.starlightlinearled.com/
# curl -fsS -o /dev/null -w '%{http_code}\n' https://mailer.starlightlinearled.com/
# ```
