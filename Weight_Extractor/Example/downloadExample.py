import gdown
import os.path

if os.path.isfile('./Example/example.pth'):
    print ("Already Downloaded")
else:
    url = 'https://drive.google.com/uc?id=1-0x64Ks7F_f97iDOXI9H0NX1CAjUHF3y'
    output = './Example/example.pth'
    gdown.download(url, output, quiet=False)



