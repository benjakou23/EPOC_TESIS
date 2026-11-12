# Cadena v3 (830 grabaciones): 06 -> 08 -> 14 -> 13 -> 16 -> 17. Se detiene si algÃºn paso falla.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"
$out = "reports\validacion_cruzada"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($p in @(@("06_validacion_cruzada","log.txt"), @("08_modelos_desplegables","log_08.txt"), @("14_cnn_8s","log_14.txt"),
                 @("13_comparaciones_desplegables","log_13.txt"), @("16_piso_ruido_fijo","log_16.txt"), @("17_estratos_duracion","log_17.txt"))) {
    & $py ("scripts\" + $p[0] + ".py") *>> ($out + "\" + $p[1])
    if ($LASTEXITCODE -ne 0) { "FALLO en $($p[0]) (codigo $LASTEXITCODE)" | Out-File "$out\FALLO.txt"; exit 1 }
}
"OK" | Out-File "$out\CADENA_OK.txt"

