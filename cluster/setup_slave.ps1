# Sets up a Windows+WSL PC as a slave reachable over SSH from the master.
# Run in PowerShell as Administrator on the slave.
#
#   .\setup_slave.ps1                 # mirrored WSL networking (Windows 11 22H2+)
#   .\setup_slave.ps1 -PortProxy 2222 # NAT fallback: Windows:2222 -> WSL:22
param(
    [int]$PortProxy = 0,
    [int]$SshPort = 22,
    [string]$SshUser = $env:USERNAME
)
$ErrorActionPreference = 'Stop'
Write-Host "=== cluster slave setup ==="

# 1) sshd inside WSL
Write-Host "-- installing/starting sshd in WSL"
wsl -e bash -lc "sudo apt-get update -qq && sudo apt-get install -y openssh-server"
wsl -e bash -lc "sudo sed -i 's/^#*Port .*/Port 22/' /etc/ssh/sshd_config; sudo service ssh restart"

# 2) firewall
$ruleName = "cluster-ssh"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP `
        -LocalPort $SshPort -Action Allow | Out-Null
    Write-Host "-- firewall rule '$ruleName' added (port $SshPort)"
}

# 3) networking
if ($PortProxy -gt 0) {
    $wslIp = (wsl -e hostname -I).Trim().Split(' ')[0]
    netsh interface portproxy delete v4tov4 listenport=$PortProxy listenaddress=0.0.0.0 2>$null | Out-Null
    netsh interface portproxy add v4tov4 listenport=$PortProxy listenaddress=0.0.0.0 connectport=22 connectaddress=$wslIp | Out-Null
    $sshPort = $PortProxy
    Write-Host "-- portproxy: Windows port $PortProxy -> WSL $wslIp:22"
    Write-Host "   (WSL address can change after a reboot: re-run this script or add a logon task)"
} else {
    $sshPort = 22
    Write-Host "-- assuming mirrored WSL networking; add to %USERPROFILE%\.wslconfig:"
    Write-Host "[wsl2]"
    Write-Host "networkingMode=mirrored"
    Write-Host "   then run 'wsl --shutdown' and reopen WSL"
}

# 4) addresses for the master
Write-Host ""
Write-Host "=== next steps ==="
Write-Host "IPv4 addresses of this PC:"
Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '172.*' } |
    Select-Object InterfaceAlias, IPAddress | Format-Table -AutoSize
Write-Host "On the master add to ~/.ssh/config:"
Write-Host "Host <slave-name>"
Write-Host "    HostName <ip-above>"
Write-Host "    User $SshUser"
Write-Host "    Port $sshPort"
Write-Host ""
Write-Host "Then: ssh-copy-id <slave-name>"
