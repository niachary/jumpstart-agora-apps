import subprocess

print("benchmarking started.")
subprocess.run(["/runscript.sh"], shell=True)
print("benchmarking done.")