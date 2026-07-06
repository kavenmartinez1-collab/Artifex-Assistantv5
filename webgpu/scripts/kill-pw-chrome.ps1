$procs = Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" |
  Where-Object { $_.CommandLine -match 'enable-unsafe-webgpu' }
if (-not $procs) { Write-Output 'no playwright chrome found'; exit 0 }
foreach ($p in $procs) {
  $mb = [math]::Round($p.WorkingSetSize / 1MB)
  Write-Output ("kill PID {0} ({1} MB)" -f $p.ProcessId, $mb)
  Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
