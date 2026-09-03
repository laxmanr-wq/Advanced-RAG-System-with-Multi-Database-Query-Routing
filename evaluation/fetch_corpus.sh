#!/usr/bin/env bash
# Fetch the three-domain benchmark corpus (financial / products / support).
#
#   bash evaluation/fetch_corpus.sh
#
# ~232 MB across 31 PDFs, written to data/<category>/. Every source is freely
# redistributable: US Federal Reserve reports (US government work, public
# domain), ECB annual report, Apple's public SEC filing, official Raspberry Pi
# datasheets, and Raspberry Pi Press books (CC BY-NC-SA 3.0).
#
# Already-downloaded files are skipped, so re-running is cheap.
set -u

UA="RAG-Agent-Eval/1.0 (benchmark corpus fetch)"
BASE="$(cd "$(dirname "$0")/.." && pwd)/data"

get() {  # get <category> <filename> <url>
  out="$BASE/$1/$2"
  mkdir -p "$BASE/$1"
  if [ -s "$out" ]; then printf "  skip (exists)  %s\n" "$2"; return; fi
  code=$(curl -sL -A "$UA" -m 300 -o "$out" -w "%{http_code}" "$3")
  if [ "$code" != "200" ]; then printf "  FAIL %s      %s\n" "$code" "$2"; rm -f "$out"; return; fi
  if [ "$(head -c 4 "$out")" != "%PDF" ]; then printf "  NOT-PDF        %s\n" "$2"; rm -f "$out"; return; fi
  printf "  ok  %6s KB  %s\n" "$(( $(wc -c < "$out") / 1024 ))" "$2"
}

echo "=== FINANCIAL (annual reports, 10-K, stability reports) ==="
F="https://www.federalreserve.gov/publications/files"
get financial fed-annual-report-2018.pdf       "$F/2018-annual-report.pdf"
get financial fed-annual-report-2019.pdf       "$F/2019-annual-report.pdf"
get financial fed-annual-report-2020.pdf       "$F/2020-annual-report.pdf"
get financial fed-annual-report-2021.pdf       "$F/2021-annual-report.pdf"
get financial fed-annual-report-2022.pdf       "$F/2022-annual-report.pdf"
get financial fed-annual-report-2023.pdf       "$F/2023-annual-report.pdf"
get financial fed-annual-report-2024.pdf       "$F/2024-annual-report.pdf"
get financial fed-financial-stability-2024.pdf "$F/financial-stability-report-20241122.pdf"
get financial apple-10k-fy2024.pdf             "https://s2.q4cdn.com/470004039/files/doc_earnings/2024/q4/filing/10-Q4-2024-As-Filed.pdf"
get financial ecb-annual-report-2023.pdf       "https://www.ecb.europa.eu/pub/pdf/annrep/ecb.ar2023~d033c21ac2.en.pdf"
# NOTE: SEC EDGAR blocks scripted downloads (403) even with a declared User-Agent.

echo "=== PRODUCTS (hardware datasheets & product briefs) ==="
D="https://datasheets.raspberrypi.com"
get products rpi4-datasheet.pdf                "$D/rpi4/raspberry-pi-4-datasheet.pdf"
get products rpi5-product-brief.pdf            "$D/rpi5/raspberry-pi-5-product-brief.pdf"
get products rpi400-product-brief.pdf          "$D/rpi400/raspberry-pi-400-product-brief.pdf"
get products pico-datasheet.pdf                "$D/pico/pico-datasheet.pdf"
get products pico-2-datasheet.pdf              "$D/pico/pico-2-datasheet.pdf"
get products pico-w-datasheet.pdf              "$D/picow/pico-w-datasheet.pdf"
get products rp2040-datasheet.pdf              "$D/rp2040/rp2040-datasheet.pdf"
get products rp2350-datasheet.pdf              "$D/rp2350/rp2350-datasheet.pdf"
get products cm4-datasheet.pdf                 "$D/cm4/cm4-datasheet.pdf"
get products cm5-product-brief.pdf             "$D/cm5/cm5-product-brief.pdf"
get products camera-module-3-brief.pdf         "$D/camera/camera-module-3-product-brief.pdf"

echo "=== SUPPORT (how-to guides, troubleshooting, user manuals) ==="
R="https://raw.githubusercontent.com/raspberrypipress/released-pdfs/main"
get support help-my-computer-is-broken.pdf     "$R/help-my-computer-is-broken.pdf"
get support conquer-the-command-line.pdf       "$R/conquer-the-command-line-v2.pdf"
get support camera-guide.pdf                   "$R/camera-guide.pdf"
get support simple-electronics-gpio-zero.pdf   "$R/simple-electronics-with-gpio-zero.pdf"
get support experiment-sense-hat.pdf           "$R/experiment-with-the-sense-hat.pdf"
get support learn-to-code-with-scratch.pdf     "$R/learn-to-code-with-scratch.pdf"
get support make-games-with-python.pdf         "$R/make-games-with-python.pdf"
get support micropython-pico-guide.pdf         "$R/get-started-with-micropython-raspberry-pi-pico.pdf"
get support raspberry-pi-beginners-book.pdf    "$R/raspberry-pi-beginners-book.pdf"
get support virtualbox-user-manual.pdf         "https://download.virtualbox.org/virtualbox/7.1.4/UserManual.pdf"

echo
echo "corpus:"
for c in financial products support; do
  printf "  %-10s %2s PDFs\n" "$c" "$(ls "$BASE/$c"/*.pdf 2>/dev/null | wc -l | tr -d ' ')"
done
