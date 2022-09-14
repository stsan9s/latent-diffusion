import os
import shutil
import os.path as osp
from glob import glob

logdirs = glob('/scratch/acd13649ev/logs/*')
print(logdirs)

for dir in logdirs:
    date = dir.split('/')[-1][:19] # 19 is the length of the formatted date
    config_file = osp.join(dir, 'configs', f'{date}-project.yaml')
    command = f'python scripts/sample_diffusion.py \
       -r models/ldm/ffhq256/model.ckpt \
       -l {dir} \
       -n 250 \
       --batch-size 128 \
       --classifier_config {config_file} \
       -c 200 \
       -e 0.0 \
       --plms'
    
    template_script = '/home/acd13649ev/summer2022/latent-diffusion/jobs/generate_samples_plms/generate_cg_samples.sh'
    og_config_name = dir.split('/')[-1][20:]
    copy_path = f'/home/acd13649ev/summer2022/latent-diffusion/jobs/generate_samples_plms/sample_{og_config_name}.sh'
    path = shutil.copy(template_script, copy_path) # copy file with permissions
    print(path)

    f = open(path, 'a')
    f.write(command)
    f.close()

    os.chdir(os.getcwd())
    os.system(f'qsub -g gcc50495 {path}')
    os.system(f'rm {path}')