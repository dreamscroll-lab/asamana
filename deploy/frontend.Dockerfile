# Frontend image — build the React SPA, serve it via nginx. nginx also reverse-
# proxies /api and /ws to the backend, so the app uses same-origin relative URLs
# (no CORS). Build context is the frontend/ directory.

# Stage 1: build the static bundle
FROM node:22-alpine AS build
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY . ./
# Empty base → relative /api and /ws (nginx proxies them). Override to point the
# bundle straight at a remote API (e.g. a CDN-hosted frontend, no proxy).
ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

# Stage 2: serve + reverse-proxy
FROM nginx:1.27-alpine AS runtime
# Backend upstream, substituted into the nginx template at container start by the
# stock nginx entrypoint (envsubst over ${BACKEND_URL}; nginx's own $vars are kept).
ENV BACKEND_URL=http://backend:7860
COPY --from=build /build/dist /usr/share/nginx/html
COPY <<'EOF' /etc/nginx/templates/default.conf.template
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # SPA: unknown paths fall back to index.html (client-side routing).
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API + WebSocket → backend service.
    location /api/ {
        # A map template is ~20 MB of art and is uploaded whole. nginx's default of
        # 1m rejects it here, before the backend ever sees it — and `npm run dev`
        # proxies without any such limit, so the failure appears only in the
        # deployed stack. Matches template_import.MAX_ARCHIVE_BYTES.
        client_max_body_size 64m;
        proxy_pass ${BACKEND_URL};
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    location /ws/ {
        proxy_pass ${BACKEND_URL};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
    }
}
EOF

EXPOSE 80
