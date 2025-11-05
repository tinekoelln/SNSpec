#testing a standalone terminal run
#Idea: check that that python environment works as intended

import sys
import datetime

print("Hello World!")
print(f"Python Executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Today's date is {datetime.date.today()}")
