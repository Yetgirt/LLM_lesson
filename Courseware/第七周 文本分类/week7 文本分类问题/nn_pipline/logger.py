# logger.py   ← 文件名必须叫这个！
import logging
import os
from datetime import datetime

# 自动创建 logs 文件夹
os.makedirs("logs", exist_ok=True)

# 日志文件带时间命名
log_file = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 配置日志（这几行背下来就行）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()  # 同时打印到屏幕
    ]
)

logger = logging.getLogger(__name__)

# 测试一下（运行时会看到输出）
if __name__ == "__main__":
    logger.info("logger 初始化成功！")