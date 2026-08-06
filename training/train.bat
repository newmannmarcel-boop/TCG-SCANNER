@echo off
REM Initial-Training (Stufe 1) fuer Riftbound-Karten-Klassifizierer
REM Trainiert MobileNetV3-Small von ImageNet-Backbone auf ALLEN aktuellen
REM Riftcodex-Karten (inkl. neuer Sets wie Vendetta/Unleashed).
REM Laeuft unabhaengig von Claude-Code-Sessions.
REM
REM Der Trainingsprozess selbst drosselt sich auf max. 320W GPU-Leistung
REM (siehe throttle_gpu_power in train.py) -> die GPU bleibt fuer andere
REM Anwendungen (Spiele etc.) voll nutzbar, keine Admin-Rechte noetig.

setlocal
set PYTHONIOENCODING=utf-8
set PATH=C:\Users\newma\AppData\Local\Programs\Python\Python311;C:\Users\newma\AppData\Local\Programs\Python\Python311\Scripts;%PATH%

cd /d "%~dp0"

echo ============================================================
echo Initial-Training startet (Trainingsprozess auf max. 320W gedrosselt).
echo Fenster minimieren OK, aber NICHT schliessen.
echo Laedt neue Karten-Bilder herunter, dann 40 Epochs Training.
echo Anschliessend train_phone.bat fuer die Handy-Realismus-Stufe starten.
echo ============================================================
echo.

python train.py

echo.
echo ============================================================
echo Training beendet. Beliebige Taste zum Schliessen.
echo ============================================================
pause
