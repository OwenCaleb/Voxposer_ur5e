import openai
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks

#openai的key和base
openai.api_key = "sk-cgpibTHnxWRPOUxdb05uaf9wPc687e0mc4EJIzpNjT3G2q5F"
openai.api_base= "https://api.chatanywhere.tech/v1"


#日志输出 工具
################################################
import os
from datetime import datetime

# 定义日志保存目录
log_dir = "/home/ur5/voxposer/Voxposer_ur5e/check"

#（确保目录存在）
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 时间戳
def get_timestamp():
    # 返回当前时间的字符串，格式为 YYYYMMDD_HHMMSS
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# 方法 把指定内容放到指定文件名的txt中
def log_to_file(filename_prefix, content):
    filename = f"{filename_prefix}_{get_timestamp()}.txt"
    path = os.path.join(log_dir, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content + "\n")
################################################



#配置文件加载
config = get_config('rlbench')

# 模型类型
# 遍历配置中所有的语言模型（LMP）的设置，将其模型更改
# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
for lmp_name, cfg in config['lmp_config']['lmps'].items():
    cfg['model'] = 'gpt-3.5-turbo'

# 对象初始化
# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config['visualizer']) # 可视化类
env = VoxPoserRLBench(visualizer=visualizer) # 主环境类

# lmps是一个字典，存LMP类对象
lmps, lmp_env = setup_LMP(env, config, debug=False) # 包装对象

# voxposer_ui 是一个LMP类对象
voxposer_ui = lmps['plan_ui'] 

# 加载任务，选择具体任务
# below are the tasks that have object names added to the "task_object_names.json" file
# uncomment one to use
# env.load_task(tasks.PutRubbishInBin)
# env.load_task(tasks.LampOff)
# env.load_task(tasks.OpenWineBottle)
# env.load_task(tasks.PushButton)
# env.load_task(tasks.TakeOffWeighingScales)
# env.load_task(tasks.MeatOffGrill)
# env.load_task(tasks.SlideBlockToTarget)
env.load_task(tasks.TakeLidOffSaucepan)
# env.load_task(tasks.TakeUmbrellaOutOfUmbrellaStand)

# 重置环境 
descriptions, obs = env.reset()

set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer

# ？ 为什么要random
instruction = np.random.choice(descriptions)

# 记录指令
log_to_file("instruction_log", f"Prompt:\n{instruction}")

# 主要执行方法
voxposer_ui(instruction)

