#!/usr/bin/env bash
set -euo pipefail

MITSUBA_CMD=${MITSUBA_CMD:-mitsuba}

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [-integrator NAME] [-minspp N] [-maxspp N]

Options:
  -integrator NAME   Render only scene_quads*_integratorNAME.xml files
  -minspp N           Minimum spp to render (must be one of the allowed set)
  -maxspp N           Maximum spp to render (must be one of the allowed set)
  -alpha N            Roughness value for the BSDF (must be one of the allowed set)
  -h, --help          Show this help

Examples:
  $(basename "$0") -integrator bdpt
  $(basename "$0") -minspp 16 -maxspp 1024
  MITSUBA_CMD=/full/path/to/mitsuba $(basename "$0") -integrator bdpt -minspp 16 -maxspp 1024

Behavior:
  For each matching scene file named like scene_quads{N}_integrator{NAME}_alpha{ALPHA}.xml,
  the script iterates the allowed spp values and runs:
    mitsuba -S {spp} scene_quads{N}_integrator{NAME}_alpha{ALPHA}.xml -o scene_quads{N}_integrator{NAME}_alpha{ALPHA}_spp{SPP}.exr
EOF
}

INTEGRATOR=""
MINSPP=""
MAXSPP=""
ALPHA=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) print_usage; exit 0;;
    -integrator)
      INTEGRATOR="$2"
      shift 2
      ;;
    -minspp)
      MINSPP="$2"
      shift 2
      ;;
    -maxspp)
      MAXSPP="$2"
      shift 2
      ;;
    -alpha)
      ALPHA="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

allowed_spps=(2 4 8 16 32 64 128 256 512 1024 4096 8092 16184)

is_allowed_in_list() {
  local val="$1"
  for a in "${allowed_spps[@]}"; do
    if [[ "$a" -eq "$val" ]]; then
      return 0
    fi
  done
  return 1
}

# determine glob pattern for the new filename format
if [[ -n "$INTEGRATOR" ]] && [[ -n "$ALPHA" ]]; then
  pattern="scene_quads*_integrator${INTEGRATOR}_alpha${ALPHA}.xml"
else
  pattern="scene_quads*_integrator*.xml"
fi

shopt -s nullglob
files=( $pattern )
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No files found for pattern: $pattern"
  exit 0
fi

for f in "${files[@]}"; do
  # extract nquads and integrator name from filename
  if [[ "$f" =~ scene_quads([0-9]+)_integrator([^.]*) ]]; then
    nquads="${BASH_REMATCH[1]}"
    integratorName="${BASH_REMATCH[2]}"
  else
    echo "Skipping $f (filename doesn't match expected pattern)"
    continue
  fi

  for spp in "${allowed_spps[@]}"; do
    # apply minspp/maxspp filters if provided
    if [[ -n "$MINSPP" ]]; then
      if (( spp < MINSPP )); then
        continue
      fi
    fi
    if [[ -n "$MAXSPP" ]]; then
      if (( spp > MAXSPP )); then
        continue
      fi
    fi

    out="scene_quads${nquads}_integrator${integratorName}_alpha${ALPHA}_spp${spp}.exr"
    echo "Running: $MITSUBA_CMD -S $spp $f -o $out"
    "$MITSUBA_CMD" -S "$spp" "$f" -o "$out"
  done
done

shopt -u nullglob
