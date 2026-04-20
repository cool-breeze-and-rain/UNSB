import time
import psutil
import subprocess

# process = psutil.Process(4161)
#
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "Process will start after \033[31m 400*3 second \033[0m:\n")
# time.sleep(400*3)
#
# while process.is_running():
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "train Process is still running...")
#     time.sleep(30)  # 每秒检查一次
#
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "train Process has finished. We will restart \033[31m train.py \033[0m with another dataset:\n")

# subprocess.run([
#     "python", "test.py",
#     "--dataroot", "/home/ubuntu/zhonghaiqin/dataset/MIST/Ki67-003/Ki67/TrainValAB",
#     "--model", "cut",
#     "--name", "SIMGAN_MIST_Ki67_new",
#     "--gpu_ids", "3",
#     "--checkpoints_dir", "./checkpoints/MIST_Ki67",
#     "--results_dir", "./results",
#     "--crop_size", "512",
#     # "--load_size", "256"
#     "--preprocess", "resize_and_crop",
#     "--batch_size", "8",
#     # "--lambda_pathology", "0.01"
# ])

# subprocess.run([
#     "python", "train.py",
#     "--dataroot", "/home/ubuntu/zhonghaiqin/dataset/MIST/HER2-003/HER2/TrainValAB",
#     "--model", "sb",
#     "--name", "UNSB_MIST_HER2_cropsize_256_loadsize_256",
#     "--gpu_ids", "1",
#     "--checkpoints_dir", "./checkpoints/MIST_HER2",
#     "--display_env", "UNSB_MIST_HER2_cropsize_256_loadsize_256",
#     "--crop_size", "256",
#     "--load_size", "256",
#     "--preprocess", "resize_and_crop",
#     "--batch_size", "8",
#     "--lambda_SB", "1.0",
#     "--lambda_NCE", "1.0"
#     # "--use_moe",
#     # "--num_experts", "2",
#     # "--top_k", "1",
#     # "--lambda_orth", "50.0",
#     # "--lambda_load", "50.0",
#     # "--continue_train",
#     # "--epoch_count", "11",
#     # "--epoch", "10",
#     # "--lambda_PSS", "50.0"
#     # "--lambda_pathology", "1.0"
# ])

subprocess.run([
    "python", "test.py",
    "--dataroot", "/home/ubuntu/zhonghaiqin/dataset/MIST/HER2-003/HER2/TrainValAB",
    "--model", "sb",
    "--name", "UNSB_MIST_HER2_cropsize_256_loadsize_256",
    "--gpu_ids", "5",
    "--checkpoints_dir", "./checkpoints/MIST_HER2",
    "--results_dir", "./results_1",
    "--crop_size", "256",
    "--load_size", "256",
    "--preprocess", "resize_and_crop",
    "--batch_size", "1",
    "--phase", "test",
    "--eval",
    "--lambda_SB", "1.0",
    "--lambda_NCE", "1.0"
])

# python train.py --dataroot="/home/ubuntu/zhonghaiqin/dataset/MIST/PR-001/PR/TrainValAB" --model="cut" --name="Ours_MIST_PR" --gpu_ids=4 --checkpoints_dir="./checkpoints/MIST_PR" --display_env="Ours_MIST_PR" --crop_size=512 --load_size=512 --batch_size=2 --preprocess="resize_and_crop"