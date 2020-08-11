import time

class CODES:
    INFO = "[INFO]"
    ERROR = "[ERROR]"

# Return time elapsed
def timer(start_time=None,msg = None, ):
    if start_time is None:
        return time.time()
    if msg is None:
        return time.time() - start_time
    print(CODES.INFO, msg , ": {:.3f} s".format(time.time() - start_time))
    return time.time() - start_time