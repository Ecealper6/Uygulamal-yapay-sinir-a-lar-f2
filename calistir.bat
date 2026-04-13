@echo off
cd /d %~dp0

cd triage_ann_project

echo Gerekli paketler kontrol ediliyor...
pip install -r requirements.txt

echo Uygulama baslatiliyor...
python app.py

pause
