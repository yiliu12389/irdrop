import os
import time
import shutil
import tarfile
import glob

t = time.time()
features_path = '../data'
decompress_path = '../IR_drop'

# 创建并清空解压目录（跨平台替代 mkdir -p 和 rm -rf）
print('Create decompress dir.')
if os.path.exists(decompress_path):
    shutil.rmtree(decompress_path)
os.makedirs(decompress_path, exist_ok=True)


# 遍历并解压所有 .tar.gz 文件
for parent, dirnames, filenames in os.walk(features_path):
    for filename in filenames:
        if filename.endswith('.gz'):
            filepath = os.path.join(parent, filename)
            print('Process %s.' % filename)

            # 计算对应的解压目标目录
            dest_path = parent.replace('data', 'IR_drop')
            os.makedirs(dest_path, exist_ok=True)

            # tarfile 直接支持 .tar.gz，无需中间步骤（替代 gzip -dk + tar -xf + rm）
            try:
                with tarfile.open(filepath, 'r:gz') as tar:
                    tar.extractall(path=dest_path)
                print('finished')
            except Exception as e:
                print(f'Error processing {filename}: {e}')

print('Decompress finished in %.2fs' % (time.time() - t))