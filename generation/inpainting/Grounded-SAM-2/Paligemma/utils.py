def raw_bbox_coords(decoded : str):
    #decoded = "<loc0338><loc0055><loc0792><loc0943> a car"
    str1 = decoded.split(' ')[0]
    str2 = str1.split('<loc')[1:]
    normalize_coord = [int(x.split('>')[0])/1024 for x in str2]
    return normalize_coord

def unormalize_bbox_coord(coords : list(int), weight : int, height : int):
    return [[coords[0]*height, coords[1]*width, coords[2]*height, coords[3]*width]]