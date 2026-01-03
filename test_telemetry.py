
from datetime import datetime
import time


start = datetime.now()

time.sleep(0.1)

delta = datetime.now() - start
print(delta.total_seconds())