import time

def get_time():
    return time.time()

def comp_time(t0,time_fun):
    if time_fun is None:
        time_fun = lambda x: x
    used_time = time_fun(round(time.time() - t0,2))
    measure = "seconds"
    if used_time >= 60.0:
        used_time /= 60.0
        #used_time = round(used_time,2)
        measure = "minutes"
    if used_time >= 60.0:
        used_time /= 60.0
        #used_time = round(used_time,2)
        measure = "hours"
    used_time = round(used_time,2)
    return str(used_time) + " " + measure
