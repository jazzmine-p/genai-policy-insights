import os
from helpers import create_directory

# Directory
data_dir = 'data/'
save_dir = 'results/'
docs_type = 'education' # 'all-docs', 'education, 'others'
model_dir = f'model_num-{docs_type}'
create_directory(save_dir + model_dir)
log_dir = os.path.join(save_dir, model_dir)