[Setup]
AppId={{9F3D3F6D-5D8B-4D16-A9A0-6A8D6F7E7A10}
AppName=ME_OpenDRT
AppVersion=1.2.5
AppVerName=ME_OpenDRT OFX v1.2.5 (OpenDRT 1.1.0)
AppPublisher=Moaz ELgabry
AppPublisherURL=https://moazelgabry.com
AppSupportURL=https://github.com/MoazElgabry/ME_OFX/issues
AppUpdatesURL=https://github.com/MoazElgabry/ME_OFX
DefaultDirName={commoncf}\OFX\Plugins
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=ME_OpenDRT_v1.2.5_Windows_cuda_opencl_Installer
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin

[Files]
Source: "..\bundle\ME_OpenDRT.ofx.bundle\*"; DestDir: "{commoncf64}\OFX\Plugins\ME_OpenDRT.ofx.bundle"; Flags: ignoreversion recursesubdirs createallsubdirs

[Code]
function ResolveRunning: Boolean;
var
  ResultCode: Integer;
begin
  Result := False;

  if Exec(
      ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      '-NoProfile -ExecutionPolicy Bypass -Command "if (Get-Process Resolve -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"',
      '',
      SW_HIDE,
      ewWaitUntilTerminated,
      ResultCode) then
  begin
    Result := (ResultCode = 1);
  end;
end;

function InitializeSetup(): Boolean;
var
  Clicked: Integer;
begin
  while ResolveRunning() do
  begin
    Clicked := SuppressibleMsgBox(
      'DaVinci Resolve is currently running.' + #13#10 +
      'Please close Resolve before installing ME_OpenDRT.',
      mbError,
      MB_RETRYCANCEL,
      IDRETRY);

    if Clicked = IDCANCEL then
    begin
      Result := False;
      Exit;
    end;
  end;

  Result := True;
end;
