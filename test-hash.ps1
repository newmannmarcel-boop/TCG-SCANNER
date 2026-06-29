## Schneller Test des Hash-Algorithmus mit 3 Karten
Add-Type -AssemblyName System.Drawing

$resp = Invoke-RestMethod "https://api.riftcodex.com/cards?size=3&page=1"
$out = @{ hashes = New-Object System.Collections.ArrayList; cardIds = New-Object System.Collections.ArrayList }

foreach ($card in $resp.items) {
    $url = $card.media.image_url
    if (-not $url) { continue }
    try {
        $bytes = (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 15).Content
        $stream = New-Object IO.MemoryStream(,$bytes)
        $img = [System.Drawing.Image]::FromStream($stream)
        $small = New-Object System.Drawing.Bitmap(9, 8, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
        $g = [System.Drawing.Graphics]::FromImage($small)
        $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $g.DrawImage($img, 0, 0, 9, 8)
        $g.Dispose()

        $gray = New-Object 'int[]' 72
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
        [void]$out.hashes.Add(@([int64]$lo, [int64]$hi))
        [void]$out.cardIds.Add($card.id)
        Write-Host "$($card.name) → lo=$lo hi=$hi"
        $img.Dispose(); $small.Dispose()
    } catch {
        Write-Host "FEHLER: $($card.name) → $($_.Exception.Message)"
    }
}

$json = $out | ConvertTo-Json -Compress -Depth 4
Write-Host "JSON: $json"
