@echo off
REM Phone-Realistic Fine-Tuning fuer Riftbound-Karten-Klassifizierer
REM Baut auf best_model.pth auf, trainiert mit handy-realistischer Augmentation.
REM Laeuft unabhaengig von Claude-Code-Sessions.
REM
REM Der Trainingsprozess selbst drosselt sich auf max. 320W GPU-Leistung
REM (siehe throttle_gpu_power in train_phone.py) -> die GPU bleibt fuer andere
REM Anwendungen (Spiele etc.) voll nutzbar, keine Admin-Rechte noetig.

setlocal
set PYTHONIOENCODING=utf-8
set PATH=C:\Users\newma\AppData\Local\Programs\Python\Python311;C:\Users\newma\AppData\Local\Programs\Python\Python311\Scripts;%PATH%

cd /d "%~dp0"

echo ============================================================
echo Phone-Realistic Training startet (Trainingsprozess auf max. 320W gedrosselt).
echo Fenster minimieren OK, aber NICHT schliessen.
echo Checkpoint nach jeder Epoch. Dauert wegen Drosselung deutlich laenger.
echo Achte auf "val_phone" - das ist der Handy-Zielwert.
echo ============================================================
echo.

python train_phone.py

echo.
echo ============================================================
echo Training beendet. Beliebige Taste zum Schliessen.
echo ============================================================
pause
