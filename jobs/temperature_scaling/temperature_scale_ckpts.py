import os
import shutil
import os.path as osp
from glob import glob

logdirs = glob('/scratch/acd13649ev/logs/*') # modify to be flexible if have time
print(logdirs)

for dir in logdirs:
    date = dir.split('/')[-1][:19] # 19 is the length of the formatted date
    config_file = osp.join(dir, 'configs', f'{date}-project.yaml')
    command = f'python scripts/temperature_scaling.py --classifier_config {config_file}'
    
    template_script = '/home/acd13649ev/summer2022/latent-diffusion/jobs/temperature_scaling/temperature_scale_ckpts.sh'
    og_config_name = dir.split('/')[-1][20:]
    copy_path = f'/home/acd13649ev/summer2022/latent-diffusion/jobs/temperature_scaling/temperature_scale_ckpts{og_config_name}.sh'
    path = shutil.copy(template_script, copy_path) # copy file with permissions
    print(path)

    f = open(path, 'a')
    f.write(command)
    f.close()

    os.chdir(os.getcwd())
    os.system(f'qsub -g gcc50495 {path}')
    os.system(f'rm {path}')