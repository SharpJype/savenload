import os
import numpy as np
import types
from io import FileIO
from base64 import encodebytes as bs64enc
from base64 import decodebytes as bs64dec

# usage for basic python datatypes and numpy arrays:
#   to/from string:
#       packup(obj) -> string
#       unpack(string) -> obj
#   to/from files:
#       pcksave(path, obj) -> boolean
#       pckload(path) -> obj
#
#   also if a child object is a reference to another object, the reference remade when unpacking/loading
#
#   EXAMPLE:
#       d = {}
#       l = [d]
#       d["a"] = l
#       pcksave(path, d)
#       pckload(path)


# usage of savenload class:
#   EXAMPLE:
#       import savenload
#
#       class savenload(savenload.savenload):
#           load_env = __name__
#           def class_load(self, c): return eval(c)
#
#       class myclass(savenload.savenload):
#           pass
#
#   also if a child object is a reference to another object, the reference remade when unpacking/loading
#   
#   loading breaks when there is a custom class inside basic python objects -> classes stay as dictionaries
#       myclass -> dict/list/tuple/... -> anotherclass
#
#   instead keep class structure continuous:
#       myclass -> anotherclass -> anotheranotherclass -> ...
#   
#   to/from files:
#       x = myclass()
#       x.save(path)
#       x.load(path)



def explode(path): # returns list of dirnames, filename/dirname, extension 
    path = os.path.normpath(path.replace("/", "\\"))
    if ":" in path: location, directories = path.split("\\", 1)
    elif r"\\" in path:
        location, directories = path[2:].split("\\", 1)
        location = "\\"+location
    elif path[0] == ".":
        location, directories = path.split("\\", 1)
    else:
        directories = path
        location = "."
    if "\\" in directories:
        directories = directories.split("\\")
        while "" in directories: directories.remove("")
    else: directories = [directories]
    directories.insert(0, location)
    if "." in directories[-1]:
        filename, extension = directories[-1].rsplit(".", 1)
        return directories[:-1], filename, extension
    elif directories[-1]: return directories[:-1], directories[-1], None
    return directories, None, None
def implode(dirs, name=None, ext=None): return ("\\".join(dirs) if dirs else ".")+str("\\"+name+str("."+ext if ext else "") if name else "")
    
def makedirs(path, **kwargs):
    dirs, name, ext = explode(path)
    os.makedirs(implode(dirs, name if ext==None else None), **kwargs)







def is_array(x): return type(x)==np.ndarray
def is_str(x): return type(x) in [str,np.str_] or (is_array(x) and is_str(getattr(np, str(x.dtype))(1)))
def is_iterable(x): return type(x) in [tuple,set,list,np.ndarray]
def is_mutable(x): return type(x) in [list,np.ndarray,dict]
def is_hashable(x): return not is_mutable(x)

openers = {
    bytes: "b",int: "i",
    float: "f",str: "s",
    tuple: "t",list: "l",
    set: "e", dict: "d",
    bool: "o", None: "n",
    type(None): "n", np.ndarray: "a",
    np.int8: "i", np.int16: "i",
    np.int32: "i", np.int64: "i",
    np.uint8: "i", np.uint16: "i",
    np.uint32: "i", np.uint64: "i",
    np.float16: "f", np.float32: "f",
    np.float64: "f",
    type: "y",
    }

def reverse_dict(d): return {v:k for k,v in d.items()}
reverse_openers = reverse_dict(openers)




def datascrape(obj):
    if type(obj) in openers:
        if is_iterable(obj): return [datascrape(x) for x in obj]
        elif type(obj)==dict: return {k:datascrape(v) for k,v in obj.items()}
        return obj
    try: return {k:datascrape(v) for k,v in obj.__dict__.items()}
    except: return None

def packup(d, depth=0, datascrape=False, id_refs=None, separator=":", depth_ceiling=None, **kwargs):
    if id_refs==None: id_refs = []
    def prefix(depth, t):
        x = openers.get(t, t)
        return x if depth else "0:"+x # str(depth).zfill(dc)+openers.get(t, t)
    def save_ref(x): id_refs.append(id(x)) # save a reference
    def typestring(o): return str(type(o))[8:-2]
    def iterablepackup(i, depth, **kwargs):
        if len(i):
            depth += 1
            sep = "\n"+str(depth)+":"+" "*depth
            return sep+sep.join([packup(u, depth, **kwargs) for u in i])
        return "<empty>"
    def dictpackup(d, depth, **kwargs):
        if len(d):
            depth += 1
            sep = "\n"+str(depth)+":"+" "*depth
            return sep+sep.join([packup(k, depth, **kwargs)+separator+packup(v, depth, **kwargs) for k,v in d.items()])
        return "<empty>"
    dt = type(d)
    if depth==depth_ceiling:
        dt = None
        ds = ""
    elif id(d) in id_refs:
        dt = "r"
        ds = str(id(d))
    else:
        kwargs = {"datascrape":datascrape, "id_refs":id_refs, "separator":separator, "depth_ceiling":depth_ceiling}
        if dt==dict:
            save_ref(d)
            ds = f"{id(d)} "+dictpackup(d, depth, **kwargs)
        elif is_array(d):
            save_ref(d)
            ds = f"{id(d)} "+str(bs64enc(array2bytes(d)).hex())
        elif is_iterable(d):
            save_ref(d)
            ds = f"{id(d)} "+iterablepackup(d, depth, **kwargs)
        elif openers.get(dt, "n") in "fis":
            ds = str(d)
            if separator in ds: # encode and give special type
                dt = "S"
                ds = str(bs64enc(ds.encode("utf8")).hex())
        elif dt==bytes:
            save_ref(d)
            ds = f"{id(d)} "+str(bs64enc(d).hex())
        elif dt==bool: ds = "1" if d else "0"
        elif dt==type: ds = openers[d]
        elif datascrape:
            save_ref(d)
            dt = dict
            ds = f"{id(d)} "+dictpackup(d.__dict__|{0:typestring(d),1:id(d)}, depth, **kwargs)
        else: ds = ""
    return prefix(depth, dt)+ds





def unpack(ds, depth=0, separator=":", do_id_refs=True, id_refs=None, **kwargs):
    kwargs["do_id_refs"] = do_id_refs
    if do_id_refs:
        if id_refs==None: id_refs = {}
        kwargs["id_refs"] = id_refs
    def prefix_split(ds):
        i = ds.find(":")
        depth = int(ds[:i])
        return depth, ds[:i+depth+1], ds[i+depth+1:]
    def iterableunpack(ds, **kwargs):
        i = []
        id_ref, ds = ds.split(" ", 1)
        if kwargs["do_id_refs"]: kwargs["id_refs"][int(id_ref)] = i
        if ds[0]=="\n":
            depth, prefix, ds = prefix_split(ds)
            if ds:
                index = ds.find(prefix)
                while index > 0:
                    i.append(unpack(ds[:index], depth, **kwargs))
                    ds = ds[index+len(prefix):]
                    index = ds.find(prefix)
                i.append(unpack(ds, depth, **kwargs))
        return i
    def dictunpack(ds, **kwargs):
        d = {}
        id_ref, ds = ds.split(" ", 1)
        if kwargs["do_id_refs"]: kwargs["id_refs"][int(id_ref)] = d
        if ds[0]=="\n":
            depth, prefix, ds = prefix_split(ds)
            if ds:
                index = ds.find(prefix)
                while index > 0:
                    try: key, value = ds[:index].split(separator, 1)
                    except: print(ds[:index])
                    ds = ds[index+len(prefix):]
                    d[unpack(key, depth, **kwargs)] = unpack(value, depth, **kwargs)
                    index = ds.find(prefix)
                key, value = ds.split(separator, 1)
                d[unpack(key, depth, **kwargs)] = unpack(value, depth, **kwargs)
        return d
    def arrayunpack(ds, **kwargs):
        id_ref, ds = ds.split(" ", 1)
        array = bytes2array(bs64dec(bytes.fromhex(ds)))
        if kwargs["do_id_refs"]: kwargs["id_refs"][int(id_ref)] = array
        return array
    
    if ds:
        if depth==0: depth, prefix, ds = prefix_split(ds)
        dt = ds[0]
        if dt==openers[type(None)]: return None
        elif dt==openers[dict]: return dictunpack(ds[1:], **kwargs)
        elif dt==openers[list]: return iterableunpack(ds[1:], **kwargs)
        elif dt==openers[tuple]: return tuple(iterableunpack(ds[1:], **kwargs))
        elif dt==openers[set]: return set(iterableunpack(ds[1:], **kwargs))
        elif dt==openers[int]: return int(ds[1:])
        elif dt==openers[float]: return float(ds[1:])
        elif dt==openers[str]: return ds[1:]
        elif dt==openers[bytes]: return bs64dec(bytes.fromhex(ds[1:]))
        elif dt==openers[bool] and ds[1] in "10": return ds[1]=="1"
        elif dt==openers[type]: return reverse_openers[ds[1]]
        elif dt==openers[np.ndarray]: return arrayunpack(ds[1:], **kwargs)
        elif dt=="r":
            id_ref = int(ds[1:])
            if do_id_refs and id_ref in id_refs: return id_refs[id_ref]
            return id_ref
        elif dt=="S": return bs64dec(bytes.fromhex(ds[1:])).decode("utf8")




def array2bytes(a, **kwargs):
    pre = np.array(a.shape).astype(np.uint64).tobytes()+b"0"
    if a.dtype==np.uint8: t = 1
    elif a.dtype==np.uint16: t = 2
    elif a.dtype==np.uint32: t = 3
    elif a.dtype==np.uint64: t = 4
    elif a.dtype==np.int8: t = 5
    elif a.dtype==np.int16: t = 6
    elif a.dtype==np.int32: t = 7
    elif a.dtype==np.int64: t = 8
    elif a.dtype==np.float16: t = 9
    elif a.dtype==np.float32: t = 10
    elif a.dtype==np.float64: t = 11
    elif a.dtype==np.bool_: t = 12
    else: t = 12+int(a.itemsize/4)
    pre = pre+bytes([t])+b"0"
    return pre+a.tobytes()

def bytes2array(b, **kwargs):
    shape, t, b = b.split(b"0", 2)
    shape = np.frombuffer(shape, dtype=np.uint64)
    t = t[0]
    if t==1: t = np.uint8
    elif t==2: t = np.uint16
    elif t==3: t = np.uint32
    elif t==4: t = np.uint64
    elif t==5: t = np.int8
    elif t==6: t = np.int16
    elif t==7: t = np.int32
    elif t==8: t = np.int64
    elif t==9: t = np.float16
    elif t==10: t = np.float32
    elif t==11: t = np.float64
    elif t==12: t = np.bool_
    else: t = f"U{t-12}"
    return np.array(np.frombuffer(b, dtype=t)).reshape(shape)



def pcksave(path, data, ext="pcksave"):
    dirs, name, path_ext = explode(path)
    if not path_ext: path = implode(dirs, name, ext)
    makedirs(path, exist_ok=True)
    f = FileIO(path, "w")
    f.write(bytes(packup(data, datascrape=True), "utf-8"))
    f.close()
def pckload(path, ext="pcksave", **kwargs):
    dirs, name, path_ext = explode(path)
    if not path_ext: path = implode(dirs, name, ext)
    if os.path.isfile(path):
        f = FileIO(path, "r")
        data = unpack(str(f.read(), "utf-8"), **kwargs)
        f.close()
        return data

class savenload():
    load_env = __name__
    def class_load(self, c): return eval(c)
    def load_before(self): pass
    def load_after(self): pass
    
    def save(self, x, **kwargs): pcksave(x, self, **kwargs)
    def load(self, x, object_refs=None, **kwargs):
        if is_str(x): # x is a path
            kwargs["do_id_refs"] = False
            return self.load(pckload(x, **kwargs))
        if type(x)==dict:
            self.load_before()
            if object_refs==None: object_refs = {} # old_id: object it self
            if 0 in x: del x[0]
            if 1 in x:
                object_refs[x[1]] = self
                del x[1]
            for k,v in x.items():
                if type(v)==dict and 0 in v:
                    xx = self.class_load(v[0].replace(self.load_env+".", ""))() # get class and init
                    xx.load(v, object_refs)
                    setattr(self, k, xx)
                elif is_hashable(v) and v in object_refs: setattr(self, k, object_refs[v])
                else: setattr(self, k, v)
            self.load_after()
            return True
        return False
    
class savenload(savenload): # define after importing
    load_env = __name__
    def class_load(self, c): return eval(c)
    def load_before(self): pass
    def load_after(self): pass







if __name__ == "__main__":
##    if 1:
##        print("test1")
##        path = "test1"
##        d = {}
##        l = [d]
##        a = np.random.rand(3)
##        d["a"] = a
##        l.append(a)
##        d["b"] = l
##        pcksave(path, d)
##
##        d = pckload(path)
##        d["a"] += 1
##        print(d)
##    if 1:
##        print("test2")
##        class asd(savenload): # example
##            def __init__(self):
##                pass
##        x = asd()
##        y = asd()
##        x.num = 5
##        x.dict = {}
##        x.list = [] # x,x.dict
##        x.asd = y
##        x.asd.x = x
##        
##        x.save("test2")
##        print(x, x.asd, x.asd.x)
##        
##        x = asd()
##        x.load("test2")
##        print(x, x.asd, x.asd.x)
##    if 1:
##        print("test3")
##        x = asd()
##        d = {
##            "INT":3,
##            "LIST":list(range(5)),
##            "SET":set(range(5)),
##            "emptylist": [x],
##            "a:sd": np.random.rand(5),
##            "x": x,
##            }
##        d["d"] = d
##        d_ = packup(d)
##        print(d_)
##        d = unpack(d_)
##        print(d)
    pass
