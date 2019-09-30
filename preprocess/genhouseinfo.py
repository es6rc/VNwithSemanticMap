import pickle

from House3D import objrender, Environment, load_config

HOUSEDIR = '/media/z/Data/Object_Searching/code/Environment/houses'
CONFIGFILEPATH = "/media/z/Data/Object_Searching/code/Environment/House3D/tests/config.json"

house_ids = [
    '04f4590d85e296b4c81c5a62f8a99bce',
    '0880799c157b4dff08f90db221d7f884',
    '0a12acd2a7039cb122c297fa5b145912',
    '0b9285c9f090123caaae85500d48ea8f',
    '0c90efff2ab302c6f31add26cd698bea',
    '1dba3a1039c6ec1a3c141a1cb0ad0757',
    '5cf0e1e9493994e483e985c436b9d3bc',
    '775941abe94306edc1b5820e3a992d75'
    # 'a7e248efcdb6040c92ac0cdc3b2351a6',
    # 'f10ce4008da194626f38f937fb9c1a03'
            ]
if __name__ == '__main__':
    api = objrender.RenderAPI(w=600, h=450, device=0)
    cfg = load_config(CONFIGFILEPATH)
    houseid = '5cf0e1e9493994e483e985c436b9d3bc'
    env = Environment(api, houseid, cfg)
    # '0a0b9b45a1db29832dd84e80c1347854'
    env.reset(yaw=0)  # put the agent into the house

    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # writer = cv2.VideoWriter('out.avi', fourcc, 30, (1200, 900))
    with open('%s/%s/map.txt' % (HOUSEDIR, houseid), 'rb') as mapfile:
        origin_coor = [float(i) for i in mapfile.readline().strip('\n').split(' ')]
    smap = env.house.smap
    fmap = env.gen_2dfmap()
    L_lo = env.house.L_lo
    L_hi = env.house.L_hi
    L_det = env.house.L_det
    n_row = env.house.n_row
    robotRad = env.house.robotRad
    savedata = {'L_lo':L_lo, 'L_hi': L_hi, 'L_det': L_det,
                'robotRad': robotRad, 'n_row': n_row, 'orgn_coor': origin_coor,
                'smap': smap, 'fmap': fmap}

    with open('%s/%s/housemap.pkl' % (HOUSEDIR, houseid), 'wb') as dmp:
        pickle.dump(savedata, dmp)