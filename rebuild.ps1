py -m pip uninstall -y scalerqec
Remove-Item -Recurse -Force .\build, .\dist -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\scalerqec.egg-info -ErrorAction SilentlyContinue
py -m pip install . --no-build-isolation --no-cache-dir --force-reinstall