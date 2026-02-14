' Ollama Memory Proxy - Silent Background Launcher for Windows
' Usage: Double-click to start, or place in Startup folder for auto-start.
'
' To add to Windows startup:
'   copy start_proxy.vbs "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\"

Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Use the directory where this script lives
ProxyDir = FSO.GetParentFolderName(WScript.ScriptFullName)

' If running from Startup folder, look for the proxy in common locations
If InStr(LCase(ProxyDir), "startup") > 0 Then
    Home = WshShell.ExpandEnvironmentStrings("%USERPROFILE%")
    Candidates = Array( _
        Home & "\Desktop\ollama-memory-proxy", _
        Home & "\ollama-memory-proxy", _
        "C:\ollama-memory-proxy" _
    )
    For Each p In Candidates
        If FSO.FolderExists(p) Then
            ProxyDir = p
            Exit For
        End If
    Next
End If

If Not FSO.FileExists(ProxyDir & "\run.py") Then
    MsgBox "Could not find run.py in " & ProxyDir, vbCritical, "Ollama Memory Proxy"
    WScript.Quit 1
End If

WshShell.CurrentDirectory = ProxyDir
WshShell.Run "pythonw run.py", 0, False
