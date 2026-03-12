param(
  [Parameter(Mandatory = $true)][string]$InputPng,
  [Parameter(Mandatory = $true)][string]$OutputIco
)

$pngBytes = [System.IO.File]::ReadAllBytes($InputPng)
$dir = [System.IO.Path]::GetDirectoryName($OutputIco)
if ($dir -and -not (Test-Path $dir)) {
  [System.IO.Directory]::CreateDirectory($dir) | Out-Null
}

$fs = [System.IO.File]::Open($OutputIco, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
$bw = New-Object System.IO.BinaryWriter($fs)
try {
  $bw.Write([UInt16]0)
  $bw.Write([UInt16]1)
  $bw.Write([UInt16]1)

  $bw.Write([Byte]0)
  $bw.Write([Byte]0)
  $bw.Write([Byte]0)
  $bw.Write([Byte]0)
  $bw.Write([UInt16]1)
  $bw.Write([UInt16]32)
  $bw.Write([UInt32]$pngBytes.Length)
  $bw.Write([UInt32]22)

  $bw.Write($pngBytes)
}
finally {
  $bw.Dispose()
  $fs.Dispose()
}
