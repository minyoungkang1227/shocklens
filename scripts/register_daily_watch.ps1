# ShockLens 데일리 워치를 Windows 작업 스케줄러에 등록한다.
#   트리거 2개: (1) 로그온할 때(3분 뒤, 네트워크 연결 대기)  (2) 매일 정해진 시각(PC를 켜 둔 채로 날이 바뀔 때)
#   daily_watch.bat이 '오늘 보고서가 이미 있으면 건너뜀'을 하므로 하루 한 번만 실제로 돈다.
#   설정: 놓친 실행은 켜지면 바로(StartWhenAvailable), 절전이면 깨워서(WakeToRun), 배터리 조건으로 막지 않음.
# 사용: PowerShell에서  powershell -ExecutionPolicy Bypass -File scripts\register_daily_watch.ps1  [-Time 07:40]
# 해제: schtasks /Delete /TN ShockLensDailyWatch /F
param([string]$Time = "07:40")
$bat = Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\daily_watch.bat"
if (-not (Test-Path $bat)) { $bat = Join-Path $PSScriptRoot "daily_watch.bat" }
$action = New-ScheduledTaskAction -Execute $bat
$logon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$logon.Delay = "PT3M"
$daily = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "ShockLensDailyWatch" -Action $action -Trigger $logon, $daily -Settings $settings -Force | Out-Null
Write-Host "등록 완료: 로그온 3분 뒤 + 매일 $Time 에 $bat 실행 (하루 한 번만 실제 실행). 결과는 data\daily\ 에 저장됩니다."
