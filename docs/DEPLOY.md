# Deploying VisionInspect AI with Docker Compose

This guide runs the full stack (PostgreSQL, the FastAPI backend and the React frontend behind nginx) on a single
Linux VM from any provider. Commands assume Ubuntu 22.04/24.04 and a user with `sudo`.

```
browser ──► :8080  web (nginx)  ── /api/* ──►  backend (uvicorn :8000)  ──►  db (PostgreSQL 18)
                     static SPA                    │  /app/ai_models  (read-only, host folder)
                                                   │  /dataset        (read-only, host folder)
                                                   └  /app/storage    (uploads + heatmaps, volume)
```

Only the web port is published. The backend and the database are reachable only inside the compose network.

## 1. VM sizing

| | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB |
| vCPU | 2 | 4 |
| Disk | 30 GB | 40 GB or more |

The disk holds the Docker images (about 2 GB for the backend because of PyTorch), the models (about 850 MB), the
MVTec AD dataset (about 5 GB), the database and the uploaded images and heatmaps.

## 2. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker                      # or log out and back in
docker version && docker compose version
```

Open the web port in the provider's firewall or security group: TCP 8080, or 80 and 443 once HTTPS is set up
(section 9).

## 3. Get the code

```bash
git clone <repository-url> ~/VisionInspect-AI
cd ~/VisionInspect-AI
git checkout <branch-or-tag-to-deploy>
```

## 4. Copy the models and the dataset

`backend/ai_models/` and `dataset/` are not in Git. Copy them from the machine that has them into the clone on
the VM. Replace `user@VM_IP` with your login.

With rsync (Linux, macOS or WSL), which resumes and only sends changes:

```bash
# Run from the repository root on the machine that has the files.
rsync -a --info=progress2 backend/ai_models/ user@VM_IP:~/VisionInspect-AI/backend/ai_models/
rsync -a --info=progress2 dataset/          user@VM_IP:~/VisionInspect-AI/dataset/
```

With scp (also available in Windows PowerShell):

```bash
scp -r backend/ai_models user@VM_IP:~/VisionInspect-AI/backend/
scp -r dataset           user@VM_IP:~/VisionInspect-AI/
```

Check on the VM:

```bash
ls ~/VisionInspect-AI/backend/ai_models/_pretrained
# resnet18-f37072fd.pth  wide_resnet50_2-95faca4d.pth
ls ~/VisionInspect-AI/backend/ai_models ~/VisionInspect-AI/dataset
# one folder per MVTec category in each
```

These folders are listed in `.dockerignore`, so they are never copied into an image. They are mounted read-only
when the containers start.

## 5. Configure

```bash
cp .env.docker.example .env.docker
chmod 600 .env.docker
python3 -c "import secrets; print(secrets.token_hex(32))"   # paste as JWT_SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(24))"  # paste as POSTGRES_PASSWORD
nano .env.docker
```

Set at least `POSTGRES_PASSWORD`, `JWT_SECRET_KEY` and `FRONTEND_ORIGINS` (the address users type in the
browser, for example `http://203.0.113.10:8080`). The defaults for `AI_MODELS_DIR=./backend/ai_models` and
`DATASET_DIR=./dataset` match section 4. `.env.docker` is Git-ignored; never commit it.

## 6. Build and start

```bash
docker compose --env-file .env.docker up -d --build
docker compose --env-file .env.docker ps
```

The first build downloads PyTorch (CPU-only) and takes several minutes. The build fails on purpose if a package
version differs from `backend/requirements.txt`, if torch is not the CPU build, or if the Grid study files in
`backend/scripts/experiments/patchcore_wrn50/` do not match their locked SHA-256.

On start the backend waits for PostgreSQL, runs `alembic upgrade head` and starts uvicorn. Wait until all three
services show `healthy`, then open `http://VM_IP:8080`.

```bash
curl -s http://localhost:8080/api/health
curl -s http://localhost:8080/api/health/db
```

## 7. Create the first users (seed script)

The seed script creates a quality engineer and a factory supervisor, 15 demo products and demo inspections. It
runs inside the backend container against the same database. Passwords are never passed on the command line:
either type them when prompted:

```bash
docker compose --env-file .env.docker exec backend python scripts/seed_demo_data.py \
  --qe-email qe@example.com --qe-name "Quality Engineer" \
  --supervisor-email supervisor@example.com --supervisor-name "Factory Supervisor"
```

or pass them through environment variables without echoing them:

```bash
read -rs -p "QE password: " VISIONINSPECT_SEED_QE_PASSWORD; echo
read -rs -p "Supervisor password: " VISIONINSPECT_SEED_SUPERVISOR_PASSWORD; echo
export VISIONINSPECT_SEED_QE_PASSWORD VISIONINSPECT_SEED_SUPERVISOR_PASSWORD
docker compose --env-file .env.docker exec \
  -e VISIONINSPECT_SEED_QE_PASSWORD -e VISIONINSPECT_SEED_SUPERVISOR_PASSWORD \
  backend python scripts/seed_demo_data.py \
  --qe-email qe@example.com --qe-name "Quality Engineer" \
  --supervisor-email supervisor@example.com --supervisor-name "Factory Supervisor"
unset VISIONINSPECT_SEED_QE_PASSWORD VISIONINSPECT_SEED_SUPERVISOR_PASSWORD
```

Add `--dry-run` first to see the plan. The full seed runs real AI inference for every inspection, so it takes a
while; `--per-category 1 --extra-uploads 2` gives a small data set for a quick check. The seed process loads its
own copy of the models, so memory use roughly doubles while it runs.

To create only a quality engineer without demo data, use `scripts/create_quality_engineer.py` the same way
(see its `--help`).

### Demo timestamps

The seeded inspections get demo timestamps spread over the last `--days` days (default 14) counted from the
moment the seed runs. Re-run the seed the day before a demo so the dashboard charts end at "today":

```bash
docker compose --env-file .env.docker exec backend python scripts/seed_demo_data.py \
  --qe-email qe@example.com --qe-name "Quality Engineer" \
  --supervisor-email supervisor@example.com --supervisor-name "Factory Supervisor" \
  --reset --reset-ids --yes
```

`--reset --yes` deletes **every** inspection (and its uploaded image and heatmap); users are kept. Back up first
(section 10) if the database holds anything you want to keep.

## 8. Logs and status

```bash
docker compose --env-file .env.docker ps
docker compose --env-file .env.docker logs -f backend      # Ctrl+C to stop following
docker compose --env-file .env.docker logs --tail 100 web
docker stats                                                # live CPU and memory per container
```

## 9. HTTPS (later)

Put a TLS-terminating reverse proxy in front of the web port. With [Caddy](https://caddyserver.com) on the same
VM and a DNS name pointing at it:

1. In `.env.docker` set `WEB_PORT=127.0.0.1:8080` (no longer reachable from outside) and
   `FRONTEND_ORIGINS=https://inspect.example.com`, then
   `docker compose --env-file .env.docker up -d`.
2. Install Caddy (see its documentation) and use this `/etc/caddy/Caddyfile`:

   ```
   inspect.example.com {
       reverse_proxy 127.0.0.1:8080
   }
   ```

3. `sudo systemctl reload caddy`, open ports 80 and 443, close 8080. Caddy obtains and renews the certificate.

## 10. Update

```bash
cd ~/VisionInspect-AI
git pull
docker compose --env-file .env.docker up -d --build
docker image prune -f
```

Migrations run automatically when the backend starts. If the models changed, copy them again (section 4) and run
`docker compose --env-file .env.docker restart backend`.

## 11. Backup and restore

Database (custom-format dump, run on the VM):

```bash
mkdir -p ~/backups
docker compose --env-file .env.docker exec -T db \
  sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > ~/backups/visioninspect_$(date +%F).dump
```

Uploaded images and heatmaps (the `storage` volume):

```bash
docker run --rm -v visioninspect_storage:/data:ro -v ~/backups:/backup alpine \
  tar czf /backup/storage_$(date +%F).tgz -C /data .
```

Restore (stops the API while restoring):

```bash
docker compose --env-file .env.docker stop backend web
docker compose --env-file .env.docker exec -T db \
  sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists' < ~/backups/visioninspect_YYYY-MM-DD.dump
docker run --rm -v visioninspect_storage:/data -v ~/backups:/backup alpine \
  sh -c 'tar xzf /backup/storage_YYYY-MM-DD.tgz -C /data && chown -R 10001:10001 /data'
docker compose --env-file .env.docker start backend web
```

Copy the backups off the VM (for example with `scp` or `rsync`) as well.

`docker compose down` keeps the volumes. `docker compose down -v` **deletes the database and all uploads**.

## Tested configuration

Built and smoke-tested on 2026-10-07 with Docker Desktop 29.8.2 / Compose v5.5.1 (WSL2, 6 CPUs, 7.8 GB for
Docker) as a separate project (`docker compose -p vi-test`, `WEB_PORT=18080`), all requests through the web
container:

- First build about 4.5 minutes; images: backend 2.24 GB, web 93 MB. The build checks passed (18 pins, torch
  2.14.0+cpu, Grid study file hashes). The `opencv-python-headless` pin needs no extra system libraries.
- Migrations applied to head on first start; no model or dataset warnings.
- Health, SPA deep links, login, `/ai/models` (15), uploads for tile, grid (WRN-50 full320) and wood
  (`MANUAL_REVIEW`), heatmap, analytics, dataset categories (15), a 5 MB upload (accepted) and a 61 MB request
  (413 from nginx) all passed. The seed script ran inside the backend container.
- Inference time per image (CPU): tile 1.1-1.3 s cold, 0.2 s warm; Grid 6.4-8.3 s cold, about 2 s warm.
  Memory: backend about 1 GB with tile and Grid loaded, db about 45 MB, web about 9 MB.
- `restart backend` and a backend-only recreate kept the same container address, so `web` kept proxying
  without a restart; the restart advice below is a precaution for when the address changes.
- On Windows, a repository path with a space works when `AI_MODELS_DIR` and `DATASET_DIR` are absolute paths
  with forward slashes, unquoted (e.g. `C:/Users/Jane Doe/VisionInspect-AI/dataset`).

## Known limits

- Inference is CPU-only and synchronous: each request occupies the worker while the model runs. Expect roughly
  2 inspections per second on 8 cores for the lighter models, and about 2 s per image for Grid (WRN-50 PatchCore).
  Batches and concurrent users queue; nginx allows up to 120 s per API request.
- The first prediction for a category is slower (cold start) because its model and backbone are loaded and
  hash-verified. Only `VISIONINSPECT_AI_MODEL_CACHE_SIZE` category models stay in memory per worker.
- More `UVICORN_WORKERS` means more parallel inference but a full model cache per worker; size RAM accordingly.
- The models and the dataset are mounted from the host, not baked into the image. A new VM needs them copied
  (section 4) before predictions work. The backend still starts without them and logs a warning.
- The interactive API docs (`/api/docs`) are not usable behind the `/api` prefix; use the Postman collection in
  `postman/` against `http://VM_IP:8080/api` instead.
- If the backend container is recreated on its own, restart `web` as well
  (`docker compose --env-file .env.docker restart web`) so nginx resolves the new backend address.
- Single VM, single database, no automatic backups: schedule section 11 with cron if the data matters.
