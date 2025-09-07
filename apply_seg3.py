import torch
import nibabel, time, sys
import numpy as np
import os

import torch.nn as nn
import torch.nn.functional as F
import resource

#print("Using 2 threads")
torch.set_num_threads(4)

class CortexModel(nn.Module):
    def __init__(self):
        super(CortexModel, self).__init__()
        self.conv0a = nn.Conv3d(1, 4, 3, padding=1)
        self.conv0b = nn.Conv3d(4, 8, 3, padding=1)
        self.bn0a = nn.InstanceNorm3d(8)

        self.ma1 = nn.MaxPool3d(2)
        self.conv1a = nn.Conv3d(8, 8, 3, padding=1)
        self.conv1b = nn.Conv3d(8, 8, 3, padding=1)
        self.bn1a = nn.InstanceNorm3d(8)

        self.ma2 = nn.MaxPool3d(2)
        self.conv2a = nn.Conv3d(8, 12, 3, padding=1)
        self.conv2b = nn.Conv3d(12, 12, 3, padding=1)
        self.bn2a = nn.InstanceNorm3d(12)

        self.ma3 = nn.MaxPool3d(2)
        self.conv3a = nn.Conv3d(12, 20, 3, padding=1)
        self.conv3b = nn.Conv3d(20, 16, 3, padding=1)
        self.bn3a = nn.InstanceNorm3d(16)

        # up

        self.conv2u = nn.Conv3d(16, 12, 3, padding=1)
        self.bn2u = nn.InstanceNorm3d(12)
        self.conv2v = nn.Conv3d(12+0, 12, 3, padding=1)

        # up

        self.conv1u = nn.Conv3d(12, 8, 3, padding=1)
        self.bn1u = nn.InstanceNorm3d(8)
        self.conv1v = nn.Conv3d(8+0, 8, 3, padding=1)

        # up

        self.conv0u = nn.Conv3d(8, 8, 3, padding=1)
        self.bn0u = nn.InstanceNorm3d(8)
        self.conv0v = nn.Conv3d(8+0, 4, 3, padding=1)

        self.conv1x = nn.Conv3d(4, 1, 1, padding=0)

    def forward(self, x):
        x = F.elu(self.conv0a(x))
        self.li0 = x = F.elu(self.bn0a(self.conv0b(x)))

        x = self.ma1(x)
        x = F.elu(self.conv1a(x))
        self.li1 = x = F.elu(self.bn1a(self.conv1b(x)))

        x = self.ma2(x)
        x = F.elu(self.conv2a(x))
        self.li2 = x = F.elu(self.bn2a(self.conv2b(x)))

        x = self.ma3(x)
        x = F.elu(self.conv3a(x))
        self.li3 = x = F.elu(self.bn3a(self.conv3b(x)))

        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = F.elu(self.bn2u(self.conv2u(x)))
        #x = torch.cat([x, self.li2], 1)
        x = x + self.li2
        x = F.elu(self.conv2v(x))

        self.lo1 = x
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = F.elu(self.bn1u(self.conv1u(x)))
        #x = torch.cat([x, self.li1], 1)
        x = x + self.li1
        x = F.elu(self.conv1v(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        self.la1 = x

        x = F.elu(self.bn0u(self.conv0u(x)))
        #x = torch.cat([x, self.li0], 1)
        x = x + self.li0
        x = F.elu(self.conv0v(x))

        self.out = x = self.conv1x(x)
        x = torch.sigmoid(x)
        return x

scriptpath = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cpu")

net = CortexModel()
net.load_state_dict(torch.load(scriptpath + "/torchparams/params_ribbon_resnet_00089_00000.pt", map_location=device))
net.to(device)


#antsApplyTransforms -i /tmp/anat_sess1_0028637.nii.gz -r igboxL.nii.gz -t /tmp/anat_sess1_0028637_mni0Affine.txt -o /tmp/boxL_anat_sess1_0028637.nii.gz --float
# and resp. R

nibabel.openers.Opener.default_compresslevel = 9

for fnameL in sys.argv[1:]:
    assert("boxL" in fnameL)
    fnameR = fnameL.replace("boxL", "boxR")

    ct = time.time()

    img = nibabel.load(fnameL)
    assert img.shape == (96,208,176)

    binput = torch.from_numpy(np.asarray(img.dataobj).astype("float32", copy = False))
    binput -= binput.mean()
    binput /= binput.std()
    with torch.no_grad():
        out1 = np.asarray(net(binput[None,None].to(device)).to("cpu"))

    # For compatibility, i avoid using scl_slope/inter of nifti images. Instead ribbons are encoded 0-255
    out = np.clip((out1[0,0] * 255), 0, 255).astype("uint8")
    out[out < 13] = 0
    nibabel.Nifti1Image(out, img.affine).to_filename(fnameL.replace("boxL", "boxLribbon"))


    img = nibabel.load(fnameR)
    assert img.shape == (96,208,176)
    binput = torch.from_numpy(np.asarray(img.dataobj)[::-1].astype("float32", copy = True)) # x-flip copy since torch doesn't support it
    binput -= binput.mean()
    binput /= binput.std()
    with torch.no_grad():
        out1 = np.asarray(net(binput[None,None].to(device)).to("cpu"))

    out = np.clip((out1[0,0] * 255), 0, 255)[::-1].astype("uint8")
    out[out < 13] = 0
    nibabel.Nifti1Image(out, img.affine).to_filename(fnameR.replace("boxR", "boxRribbon"))

if 0: #OUTPUT_DEBUG:
    print("Peak memory used (Gb) " + str(resource.getrusage(resource.RUSAGE_SELF)[2] / (1024.*1024)))
