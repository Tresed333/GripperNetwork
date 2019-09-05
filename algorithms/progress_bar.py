import sys

def startProgress(title):
    global progress_x
    sys.stdout.write(title + ": [" + "-"*20 + "]" + chr(8)*21)
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    x = int(x * 20 // 100)
    sys.stdout.write("▒" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x

def endProgress():
    sys.stdout.write(chr(8)*19 + "completed" + "]\n")
    sys.stdout.flush()