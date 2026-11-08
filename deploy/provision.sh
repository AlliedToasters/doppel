#!/usr/bin/env bash
# Create the Doppel VM on Hetzner Cloud and point doppelbot.org at it on Cloudflare. Idempotent: re-running
# finds the existing key, firewall, server and records.
#   Tokens (outside git): ~/.config/doppel/hcloud.token (Read & Write), ~/.config/doppel/cloudflare.token
#   (Zone.DNS edit + Zone.Zone Settings edit on the zone). Writes ~/.config/doppel/deploy.env (DOPPEL_HOST=<ip>).
set -euo pipefail
cd "$(dirname "$0")"
CFG=~/.config/doppel
DOMAIN=${DOMAIN:-doppelbot.org}
NAME=${NAME:-doppel-1} TYPE=${TYPE:-ccx23} LOCATION=${LOCATION:-fsn1} IMAGE=${IMAGE:-ubuntu-24.04}
KEY=${KEY:-$HOME/.ssh/id_rsa.pub}
HC="Authorization: Bearer $(cat $CFG/hcloud.token)"
CF="Authorization: Bearer $(cat $CFG/cloudflare.token)"
hc() { curl -sf -H "$HC" -H 'content-type: application/json' "https://api.hetzner.cloud/v1$1" "${@:2}"; }
cf() { curl -sf -H "$CF" -H 'content-type: application/json' "https://api.cloudflare.com/client/v4$1" "${@:2}"; }

# SSH key
PUB=$(cat "$KEY")
KID=$(hc "/ssh_keys" | jq -r --arg k "$PUB" '.ssh_keys[] | select(.public_key == ($k | split(" ")[0:2] | join(" "))) | .id' | head -1)
[ -n "$KID" ] || KID=$(hc /ssh_keys -X POST -d "$(jq -n --arg k "$PUB" '{name: "doppel-deploy", public_key: $k}')" | jq -r .ssh_key.id)
echo "ssh key $KID"

# Firewall: SSH + web only
FID=$(hc "/firewalls?name=doppel-web" | jq -r '.firewalls[0].id // empty')
if [ -z "$FID" ]; then
  FID=$(hc /firewalls -X POST -d '{"name":"doppel-web","rules":[
    {"direction":"in","protocol":"tcp","port":"22","source_ips":["0.0.0.0/0","::/0"]},
    {"direction":"in","protocol":"tcp","port":"80","source_ips":["0.0.0.0/0","::/0"]},
    {"direction":"in","protocol":"tcp","port":"443","source_ips":["0.0.0.0/0","::/0"]},
    {"direction":"in","protocol":"udp","port":"443","source_ips":["0.0.0.0/0","::/0"]}]}' | jq -r .firewall.id)
fi
echo "firewall $FID"

# Server
S=$(hc "/servers?name=$NAME")
if [ "$(echo "$S" | jq '.servers | length')" = 0 ]; then
  UD=$(sed "s|SSH_KEY|$PUB|" cloud-init.yaml)
  S=$(hc /servers -X POST -d "$(jq -n --arg n "$NAME" --arg t "$TYPE" --arg l "$LOCATION" --arg i "$IMAGE" \
      --arg ud "$UD" --argjson k "$KID" --argjson f "$FID" \
      '{name: $n, server_type: $t, location: $l, image: $i, user_data: $ud, ssh_keys: [$k], firewalls: [{firewall: $f}],
        labels: {app: "doppel"}}')" | jq '{servers: [.server]}')
  echo "created $NAME ($TYPE, $LOCATION)"
fi
IP4=$(echo "$S" | jq -r '.servers[0].public_net.ipv4.ip')
IP6=$(echo "$S" | jq -r '.servers[0].public_net.ipv6.ip' | sed 's|::/64|::1|')
echo "server $IP4 $IP6"
printf 'DOPPEL_HOST=%s\n' "$IP4" > $CFG/deploy.env

# DNS: proxied A + AAAA for @ and www; TLS Full (strict)
ZID=$(cf "/zones?name=$DOMAIN" | jq -r '.result[0].id')
upsert() {   # name type content
  local rid; rid=$(cf "/zones/$ZID/dns_records?type=$2&name=$1" | jq -r '.result[0].id // empty')
  local body; body=$(jq -n --arg n "$1" --arg t "$2" --arg c "$3" '{name: $n, type: $t, content: $c, proxied: true, ttl: 1}')
  if [ -n "$rid" ]; then cf "/zones/$ZID/dns_records/$rid" -X PUT -d "$body" >/dev/null; else cf "/zones/$ZID/dns_records" -X POST -d "$body" >/dev/null; fi
  echo "dns $1 $2 -> $3"
}
clear_other() {   # name: drop A/AAAA/CNAME records on this name that don't point at the VM (e.g. registrar parking)
  cf "/zones/$ZID/dns_records?name=$1&per_page=100" | jq -r --arg a "$IP4" --arg b "$IP6" \
    '.result[] | select((.type == "A" or .type == "AAAA" or .type == "CNAME") and .content != $a and .content != $b) | .id + " " + .type + " " + .content' |
  while read -r rid t c; do cf "/zones/$ZID/dns_records/$rid" -X DELETE >/dev/null; echo "dns removed $1 $t $c"; done
}
for n in "$DOMAIN" "www.$DOMAIN"; do clear_other "$n"; upsert "$n" A "$IP4"; upsert "$n" AAAA "$IP6"; done
cf "/zones/$ZID/settings/ssl" -X PATCH -d '{"value":"strict"}' >/dev/null && echo "tls full (strict)"
cf "/zones/$ZID" | jq -r '"zone status: " + .result.status + " (needs to be active: nameservers switched)"'
