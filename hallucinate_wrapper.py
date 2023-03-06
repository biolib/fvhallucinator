import subprocess
import sys 

args = sys.argv[1:]

number_of_designs = 1
if '--num' in args:
    number_of_designs = args.pop(args.index('--num') + 1)
    args.remove('--num')

for i in range(int(number_of_designs)):
    subprocess.run([
        '/opt/conda/envs/fvhallucinator/bin/python', 'hallucinate.py',
        '--suffix', str(i),
        '--seed', str(i),
        *args
    ])