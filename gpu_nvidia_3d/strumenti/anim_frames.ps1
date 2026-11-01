$ErrorActionPreference = "Continue"
Set-Location "\\wsl.localhost\Ubuntu\home\zorka\simulazioni\Test 1\gpu_nvidia_3d"
New-Item -ItemType Directory -Force frames | Out-Null
Remove-Item frames\*.png -ErrorAction SilentlyContinue
Remove-Item frames\progress.log -ErrorAction SilentlyContinue
Remove-Item omega3d_volume.mp4 -ErrorAction SilentlyContinue

$files = Get-ChildItem vtk3d\vtk3d_*.vti | Sort-Object Name
$sel = @()
for ($i = 0; $i -lt $files.Count; $i += 5) { $sel += $files[$i] }
"selezionati $($sel.Count) frame" | Add-Content frames\progress.log

$n = 0
foreach ($f in $sel) {
    $out = "frames\f_{0:D4}.png" -f $n
    python -u visualizza3d.py $f.Name --png $out *>> frames\frame.err
    if ($LASTEXITCODE -ne 0) {
        "ERRORE $($f.Name)" | Add-Content frames\progress.log
    } else {
        $n++
        "OK $n/$($sel.Count) $($f.Name)" | Add-Content frames\progress.log
    }
}

python -u -c "import glob, imageio.v2 as iio; fs=sorted(glob.glob('frames/f_*.png')); print(len(fs),'frame'); w=iio.get_writer('omega3d_volume.mp4', fps=2); [w.append_data(iio.imread(f)) for f in fs]; w.close(); print('video ok')" *>> frames\assemble.err
"fatto" | Add-Content frames\progress.log
