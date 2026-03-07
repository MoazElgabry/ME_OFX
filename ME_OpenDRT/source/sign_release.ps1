# sign_release.ps1
$ErrorActionPreference = "Stop"

$signtool = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe"
$plugin   = "C:\Users\mizo_\OneDrive\Documents\GitHub\OpenDRT-OFX\bundle\ME_OpenDRT.ofx.bundle\Contents\Win64\ME_OpenDRT.ofx"
$installer= "C:\Users\mizo_\OneDrive\Documents\GitHub\OpenDRT-OFX\installer\ME_OpenDRT_v1.2.9c_Windows_cuda_opencl_Installer.exe"
$pfx      = "$env:USERPROFILE\Desktop\ME_OpenDRT_SelfSign.pfx"

if (!(Test-Path $signtool)) { throw "signtool not found: $signtool" }
if (!(Test-Path $plugin))   { throw "plugin not found: $plugin" }
if (!(Test-Path $installer)){ throw "installer not found: $installer" }

if (!(Test-Path $pfx)) {
  Write-Host "No PFX found. Creating self-signed code-signing cert..."
  $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject "CN=Moaz Elgabry" -CertStoreLocation "Cert:\CurrentUser\My" -HashAlgorithm SHA256
  $pwSecure = Read-Host "Set password for new PFX" -AsSecureString
  Export-PfxCertificate -Cert $cert -FilePath $pfx -Password $pwSecure | Out-Null
  Write-Host "Created: $pfx"
}

$pwSecure2 = Read-Host "Enter PFX password" -AsSecureString
$pwPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($pwSecure2))

Write-Host "Signing plugin..."
& $signtool sign /fd SHA256 /f $pfx /p $pwPlain /tr http://timestamp.digicert.com /td SHA256 $plugin

Write-Host "Signing installer..."
& $signtool sign /fd SHA256 /f $pfx /p $pwPlain /tr http://timestamp.digicert.com /td SHA256 $installer

Write-Host "Verifying plugin..."
& $signtool verify /pa /v $plugin

Write-Host "Verifying installer..."
& $signtool verify /pa /v $installer

Write-Host "`nDone."
