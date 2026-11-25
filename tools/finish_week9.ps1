# Run from the repo root: powershell -ExecutionPolicy Bypass -File tools\finish_week9.ps1
Write-Host "1/4 Pinned environment"
pip freeze | Out-File -Encoding utf8 requirements_pinned.txt

Write-Host "2/4 Unit counts (before removal, isolated removed)"
python -c "from src import eda_utils as U; from src.model.graphs import build_units; [print(d, build_units(*U.load(d),5,'union','inhibitory').meta['units_before_removal'], build_units(*U.load(d),5,'union','inhibitory').meta['isolated_removed']) for d in ('fafb','malecns')]" 2>&1 | Tee-Object -FilePath units_check.txt

Write-Host "3/4 Commit and push"
git add README.md requirements_pinned.txt units_check.txt tools/finish_week9.ps1
git commit -m "week9: merge README, pinned environment, unit check"
git push

Write-Host "4/4 Move tag week9"
git tag -f week9
git push origin week9 --force
git log --oneline -1 week9
Write-Host "Done. Send me the two lines from step 2 (fafb ... / malecns ...)."
