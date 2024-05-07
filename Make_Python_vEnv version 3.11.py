import subprocess

def make_env(name):
    try:
        subprocess.run(f'py -3.11 -m venv .\\{name}', shell=True)
        print(f"Successfully created the environment: {name}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating the environment: {e.output}")



def write_bat_file(bat_filename, env_name):
    with open(bat_filename, 'w') as bat_file:
      bat_file.write(r'''@ECHO OFF
''')
      bat_file.write(('cmd /k ".\%s\Scripts\\activate"' % env_name ))
      bat_file.close()
    print(f'Successfully created the .bat file: {bat_filename}')

def main():
    env_name = input("Enter the name of the environment you want to create: ")
    env_name="{}".format(env_name).replace(" ","_")
    env_name="{}".format(env_name).replace("\\","_")
    make_env(env_name)
    bat_filename = ("Activate_enviroment_%s.bat" % env_name)
    write_bat_file(bat_filename,env_name)
    print(f'''
Warning: Name of both Files must not be Changed
Do NOT move files or change their path/position in the folder
As this will cause the Python enviroment to no longer Function properly
''')


main()
input()

