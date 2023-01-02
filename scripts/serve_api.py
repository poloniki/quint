import sys, os
sys.path.append(os.getcwd())

try:
	import quint
except ImportError:
	print("Could not find quint. Make sure you are in the root directory (are you in the scripts folder?) of quint or have run `python setup.py install` in quint's root directory")
	sys.exit(1)

try:
	import uvicorn
except ImportError:
	print("Could not import uvicorn for FastAPI. Are you in the pipenv shell? Have you run pipenv install?")
	sys.exit(1)

if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-help", "-h", "/?"]:
	print("usage: serve_api.py [OUTPUT_PATH]")
	print("    OUTPUT_PATH: only used for transcription. Outputs transcriptions to this directory")
	print("        default: current directory")
	sys.exit(0)

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
os.system(f"OUTPUT_PATH=${OUTPUT_DIR} uvicorn quint.api.fast:app")
