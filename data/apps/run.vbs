CreateObject("WScript.Shell").Run "taskkill /f /im streamlit.exe", 0, True
CreateObject("WScript.Shell").Run "python api.py", 0, False
CreateObject("WScript.Shell").Run "streamlit run app.py", 0, False