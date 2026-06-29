## build-hashes.ps1
## Berechnet dHash für jede Riftbound-Karte und schreibt riftbound-hashes.json.
## Muss dHash-IDENTISCH zu computeDHash() in index.html sein:
##   1) Bild auf 9x8 hochwertig skalieren
##   2) Grayscale (0.299R + 0.587G + 0.114B)
##   3) Pro Zeile: jedes Pixel mit rechtem Nachbarn vergleichen → 64 bit
##   4) bit positions 0..63 → lo32 + hi32 (kein Vorzeichen)

Add-Type -AssemblyName System.Drawing

$ApiBase = "https://api.riftcodex.com"
$OutFile = "$PSScriptRoot\riftbound-hashes.json"

Write-Host "[1/3] Karten-Liste laden …"
$allCards = @()
$page = 1
do {
    try {
        $resp = Invoke-RestMethod -Uri "$ApiBase/cards?size=100&page=$page" -ErrorAction Stop
    } catch {
        Write-Host "  Seite $page fehlgeschlagen: $_"
        break
    }
    if (-not $resp.items) { break }
    $allCards += $resp.items
    Write-Host "  Seite $page → $($resp.items.Count) Karten (gesamt $($allCards.Count))"
    $pages = if ($resp.pages) { $resp.pages } else { 1 }
    $page++
} while ($page -le $pages)
Write-Host "→ $($allCards.Count) Karten insgesamt"

Write-Host "[2/3] Hashes berechnen …"
$hashes = New-Object System.Collections.ArrayList
$cardIds = New-Object System.Collections.ArrayList
$skipped = 0
$idx = 0

foreach ($card in $allCards) {
    $idx++
    $url = $null
    if ($card.media -and $card.media.image_url) { $url = $card.media.image_url }
    if (-not $url) { $skipped++; continue }

    try {
        $bytes = (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 15).Content
        $stream = New-Object IO.MemoryStream(,$bytes)
        $img = [System.Drawing.Image]::FromStream($stream)
        $small = New-Object System.Drawing.Bitmap(9, 8, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
        $g = [System.Drawing.Graphics]::FromImage($small)
        $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $g.PixelOffsetMode   = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $g.SmoothingMode     = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $g.DrawImage($img, 0, 0, 9, 8)
        $g.Dispose()

        # 1D-Array mit Index = y*9+x — vermeidet PowerShell-Comma-Operator-Fallen
        $gray = New-Object 'int[]' 72  # 8 Zeilen × 9 Spalten
        for ($y = 0; $y -lt 8; $y++) {
            for ($x = 0; $x -lt 9; $x++) {
                $p = $small.GetPixel($x, $y)
                $gray[$y * 9 + $x] = [int](0.299 * $p.R + 0.587 * $p.G + 0.114 * $p.B)
            }
        }

        [uint32]$lo = 0
        [uint32]$hi = 0
        for ($y = 0; $y -lt 8; $y++) {
            for ($x = 0; $x -lt 8; $x++) {
                $a = $gray[$y * 9 + $x]
                $b = $gray[$y * 9 + $x + 1]
                if ($a -lt $b) {
                    $pos = $y * 8 + $x
                    if ($pos -lt 32) { $lo = $lo -bor ([uint32]1 -shl $pos) }
                    else             { $hi = $hi -bor ([uint32]1 -shl ($pos - 32)) }
                }
            }
        }

        [void]$hashes.Add(@([int64]$lo, [int64]$hi))
        [void]$cardIds.Add($card.id)
        $img.Dispose()
        $small.Dispose()

        if ($idx % 50 -eq 0) {
            Write-Host "  $idx / $($allCards.Count) verarbeitet"
        }
    } catch {
        $skipped++
        Write-Host "  Karte $idx ($($card.name)) übersprungen: $($_.Exception.Message)"
    }
}

Write-Host "→ $($hashes.Count) Hashes berechnet, $skipped übersprungen"

Write-Host "[3/3] JSON schreiben …"
$result = @{
    hashes  = $hashes
    cardIds = $cardIds
}
$json = $result | ConvertTo-Json -Compress -Depth 4
[System.IO.File]::WriteAllText($OutFile, $json, [System.Text.UTF8Encoding]::new($false))
$kb = [Math]::Round((Get-Item $OutFile).Length / 1024, 1)
Write-Host "→ $OutFile geschrieben ($kb KB)"
Write-Host ""
Write-Host "Fertig. Datei kann jetzt committet werden."
