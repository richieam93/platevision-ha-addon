# PlateVision 0.12.0 Release-Checkliste

1. `python3 tools/validate_release.py` ausführen.
2. `python3 tools/test_people_history.py` ausführen.
3. Add-on installieren und `/people/gallery` öffnen.
4. Aufbewahrungsfrist speichern und manuellen Cleanup mit Testbildern prüfen.
5. RTSP- und Upload-Personenbilder tageweise kontrollieren.
6. `git add -A && git commit -m "release: PlateVision 0.12.0 image archive"`.
