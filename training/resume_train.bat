@echo off
REM Resume-Training für Riftbound-Karten-Klassifizierer
REM Läuft unabhängig von Claude-Code-Sessions

setlocal
set PYTHONIOENCODING=utf-8
set PATH=C:\Users\newma\AppData\Local\Programs\Python\Python311;C:\Users\newma\AppData\Local\Programs\Python\Python311\Scripts;%PATH%

cd /d "%~dp0"

echo ============================================================
echo Resume-Training startet ...
echo Dieses Fenster kannst du minimieren, aber NICHT schliessen.
echo Alle 5 Epochs wird ein Checkpoint gespeichert.
echo ============================================================
echo.

python resume_train.py

echo.
echo ============================================================
echo Training beendet. Beliebige Taste zum Schliessen.
echo ============================================================
pause
