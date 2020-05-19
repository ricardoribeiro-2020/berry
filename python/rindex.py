
wfcdirectory = "wfc"

with open(wfcdirectory+"/k0000b0001.wfc",'r') as datfile:
  data = datfile.read().split('\n')
datfile.closed
del data[-1]
nr = len(data)                                       # Number of points in real space
r,x,y,z = {},{},{},{}                                # vector r and coordinates x,y,z
with open(wfcdirectory+"/rindex","w") as rindex:     # builds an index of r-points: not used
  for i in range(nr):
    tmp = data[i].split()
    del tmp[-1]
    del tmp[-1]
    del tmp[-1]
    rindex.write(str(i)+' '+tmp[0]+' '+tmp[1]+' '+tmp[2]+'\n')
    r[i] = tmp
    x[i] = float(tmp[0])
    y[i] = float(tmp[1])
    z[i] = float(tmp[2])
rindex.closed

