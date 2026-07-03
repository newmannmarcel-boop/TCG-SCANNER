@echo off
REM Phone-Realistic Fine-Tuning fuer Riftbound-Karten-Klassifizierer
REM Baut auf best_model.pth auf, trainiert mit handy-realistischer Augmentation.
REM Laeuft unabhaengig von Claude-Code-Sessions.

setlocal
set PYTHONIOENCODING=utf-8
set PATH=C:\Users\newma\AppData\Local\Programs\Python\Python311;C:\Users\newma\AppData\Local\Programs\Python\Python311\Scripts;%PATH%

cd /d "%~dp0"

echo ============================================================
echo Phone-Realistic Training startet ...
echo Fenster minimieren OK, aber NICHT schliessen.
echo Checkpoint alle 5 Epochs. Dauer: ca. 3-5 Stunden.
echo Achte auf "val_phone" - das ist der Handy-Zielwert.
echo ============================================================
echo.

python train_phone.py

echo.
echo ============================================================
echo Training beendet. Beliebige Taste zum Schliessen.
echo ============================================================
pause
