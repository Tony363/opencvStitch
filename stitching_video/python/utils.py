import time



class CODES:
    INFO = "[INFO]"
    ERROR = "[ERROR]"


# Return time elapsed
def timer(start_time=None,msg = None, display=False):
    # Initialize timer
    if start_time is None:
        return time.time()
    if msg is not None and display:
        print(CODES.INFO, msg , ": {:.3f} s".format(time.time() - start_time))
    return time.time() - start_time