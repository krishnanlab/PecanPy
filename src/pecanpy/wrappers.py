import time

class Timer:
    def __init__(self, name, verbose):
        self.name = name
        self.verbose = verbose
        
    def __call__(self, func):

        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            hrs = duration // 3600
            mins = duration % 3600 // 60
            secs = duration % 60
            print("Took %02d:%02d:%05.2f to %s"%(hrs, mins, secs, self.name))

            return result
            

        return wrapper if self.verbose else func
