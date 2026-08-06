@echo off
REM Boost-Runde (Stufe 3) fuer Riftbound-Karten-Klassifizierer
REM Frischer LR-Warm-Restart auf dem Phone-Realistic-Modell aus train_phone.bat,
REM um aus einem Plateau rauszukommen. Ueberschreibt best_model.pth NUR bei
REM echter Verbesserung gegenueber dem bisherigen val_phone-Bestwert.
REM Laeuft unabhaengig von Claude-Code-Sessions.
REM
REM Der Trainingsprozess selbst drosselt sich auf max. 320W GPU-Leistung
REM (siehe throttle_gpu_power in train_boost.py) -> die GPU bleibt fuer andere
REM Anwendungen (Spiele etc.) voll nutzbar, keine Admin-Rechte noetig.

setlocal
set PYTHONIOENCODING=utf-8
set PATH=C:\Users\newma\AppData\Local\Programs\Python\Python311;C:\Users\newma\AppData\Local\Programs\Python\Python311\Scripts;%PATH%

cd /d "%~dp0"

echo ============================================================
echo Boost-Training startet (Trainingsprozess auf max. 320W gedrosselt).
echo Fenster minimieren OK, aber NICHT schliessen.
echo Checkpoint nach jeder Epoch. 25 Epochs, LR-Warm-Restart.
echo Achte auf "val_phone" - das ist der Handy-Zielwert.
echo ============================================================
echo.

python train_boost.py

echo.
echo ============================================================
echo Training beendet. Beliebige Taste zum Schliessen.
echo ============================================================
pause
