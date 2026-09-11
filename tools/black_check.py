import subprocess
import sys

def run_black_check():
    try:
        subprocess.check_call([sys.executable, '-m', 'black', '--check', '.'])
        return True
    except subprocess.CalledProcessError:
        return False

if __name__ == '__main__':
    ok = run_black_check()
    if not ok:
        print('Black check failed. Run `python -m black .` to format.'); sys.exit(1)
    else:
        print('Black check passed')
