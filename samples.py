import numpy as np

def x2str(x):
    # assume flat
    return ".".join([str(i) for i in x])

def str2x(s):
    x = np.array([int(i) for i in s.split(".")])
    return x

def x2m(x):
    n = round(np.sqrt(x.shape[0]))
    x = x.reshape((n,n))
    return x

def id2m(i,smap):
    x_str = smap["id2x"][i]
    x = str2x(x_str)
    m = x2m(x)
    return m

def x2id(x,smap):
    x_str = x2str(x)
    i = smap["x2id"][x_str]
    return i


def sample2m(x):
    X = x.reshape((4,4))
    return X

def rot_sample(x):
    X = x.reshape((4,4))
    X = np.rot90(X)
    x = X.reshape(16)
    return x

def gen_n(n):
    res = {}
    x = np.array(range(4 ** 2))

    for i in range(n):
        x_str = x2str(x)
        if x_str not in res:
            res[x_str] = x.copy()
        np.random.shuffle(x)

    return res

def save_samples(samps):
    res = []
    n = len(samps)
    for k in samps:
        res.append(samps[k].tolist())
    res = np.array(res,dtype=np.int8)
    np.save("dset_samples",res,allow_pickle=True)
    print("saved " + str(n) + " samples to dset_samples")

def save_ants(samps):
    res = []
    n = samps.shape[0]

    for x in samps:
        xr = rot_sample(x)
        res.append(xr.tolist())

    res = np.array(res,dtype=np.int8)

    np.save("dset_annot",res,allow_pickle=True)
    print("saved " + str(n) + " annotations to dset_annot")

def load_samples():
    res = np.load("dset_samples.npy",allow_pickle=True)
    return res

def load_ants():
    res = np.load("dset_annot.npy",allow_pickle=True)
    return res

def create_annotmaps(ants):
    n = ants.shape[0]
    res = { "x2id":{}
          , "id2x":{}
          }
    i = 0

    for x in ants:
        xstr = x2str(x)
        res["x2id"][xstr] = i
        res["id2x"][i] = xstr
        i += 1

    return res

def load_dset():
    samps,ants = load_samples(),load_ants()
    ants_map = create_annotmaps(ants)

    res = []
    n = samps.shape[0]

    for x,y in zip(samps,ants):
        yi = x2id(y,ants_map)
        res.append(x.tolist() + [yi])

    res = np.array(res,dtype=np.int32)

    return res,n,ants_map





def main():
    do_gensets = False

    if do_gensets:
        n0 = 10 ** 5
        r0 = gen_n(n0)
        save_samples(r0)
        s0 = load_samples()
        save_ants(s0)


    """
    i0 = 52

    x0_str = ants_map["id2x"][i0]
    print(x0_str)
    x0 = str2x(x0_str)
    print(x0)
    m0 = x2m(x0)
    print(m0)

    print(i2m(i0,ants_map))
    """

    load_dset()




if __name__ == "__main__":
    main()


