CreateObject("WScript.Shell").Run "taskkill /f /im python.exe", 0, True
CreateObject("WScript.Shell").Run "taskkill /f /im streamlit.exe", 0, True
MsgBox "Processes stopped", 64, "Classifier"