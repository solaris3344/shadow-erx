# Development versions

`run_v10.py` and `run_v11.py` are the last two of the eleven exploratory versions that were run on the development scenes Beach and SNP before the first preregistration (`../preregistration/PREREG_TR.md`). They are included for transparency and are not needed to reproduce the paper. `plot_v11.py` draws their figures and `smoke_test.py` tests them.

They import `baseline_v3.py` and `run_shadow.py` from the repository root, so run them with the root on the Python path, for example from the root folder:

```
PYTHONPATH=. python development/smoke_test.py          (Linux, macOS)
$env:PYTHONPATH="."; python development\smoke_test.py   (Windows PowerShell)
```
