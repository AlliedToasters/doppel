# Caddy with the Cloudflare DNS module: certificates via DNS-01, so issuance and renewal work behind the
# Cloudflare proxy with no port-80 challenge.
FROM caddy:2.10-builder AS build
RUN xcaddy build --with github.com/caddy-dns/cloudflare
FROM caddy:2.10
COPY --from=build /usr/bin/caddy /usr/bin/caddy
