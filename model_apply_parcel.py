import torch
import nibabel
import numpy as np
import os, sys, time
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import inv

device = torch.device("cuda:0")

class ParcelCortexModel(nn.Module):
    def __init__(self):
        super(ParcelCortexModel, self).__init__()
        self.conv0a_0 = l = nn.Conv3d(1, 16, (1,1,3), padding=(0,0,1))
        self.conv0a_1 = l = nn.Conv3d(16, 16, (1,3,1), padding=(0,1,0))
        l = self.conv0a_2 = nn.Conv3d(16, 16, (3,1,1), padding=(1,0,0))
        l = self.bn0 = nn.BatchNorm3d(16, momentum=1, eps=1e-8)
        l.training = False

        self.maxpool1 = nn.MaxPool3d(2)
        l = self.convf1 = nn.Conv3d(16, 32, (3,3,3), padding=1)
        l = self.bn1 = nn.BatchNorm3d(32, momentum=1, eps=1e-8)
        l.training = False

        l = self.convout2r = nn.Conv3d(32, 96, 1, padding=0)
        self.maxpool2 = nn.MaxPool3d(2)
        l = self.convout2 = nn.Conv3d(96, 96, (3,3,3), padding=1)
        l = self.bn2 = nn.BatchNorm3d(96, momentum=1, eps=1e-8)
        l.training = False

        l = self.convout3r = nn.Conv3d(96, 128, 1, padding=0)
        self.maxpool3 = nn.MaxPool3d(2)
        l = self.convout3p = nn.Conv3d(128, 96, (3,3,3), padding=1)
        l = self.convout3 = nn.Conv3d(96, 128, 1, padding=0)
        l = self.bn3 = nn.BatchNorm3d(128, momentum=1, eps=1e-8)
        l.training = False
        self.maxpool3 = nn.MaxPool3d(2)
                
        l = self.convlx4 = nn.Conv3d(133, 128, (3,3,3), padding=1)
        l = self.convout4r = nn.Conv3d(128, 128, 1, padding=0)
        l = self.bn4 = nn.BatchNorm3d(128, momentum=1, eps=1e-8)
        l.training = False

        l = self.convlx5 = nn.Conv3d(128, 128, (3,3,3), padding=1)
        l = self.convout5r = nn.Conv3d(128, 128, 1, padding=0)
        l = self.bn5 = nn.BatchNorm3d(128, momentum=1, eps=1e-8)
        l.training = False

        l = self.convlx6 = nn.Conv3d(128, 96, (3,3,3), padding=1)
        l = self.convout6r = nn.Conv3d(96+96, 96, 1, padding=0)
        l = self.bn6 = nn.BatchNorm3d(96, momentum=1, eps=1e-8)
        l.training = False
        
        l = self.convlx7 = nn.Conv3d(96, 96, (3,3,3), padding=1)
        l = self.convout7r = nn.Conv3d(96+32, 96, 1, padding=0)
        l = self.bn7 = nn.BatchNorm3d(96, momentum=1, eps=1e-8)
        l.training = False

        self.conv8a_0 = l = nn.Conv3d(96, 96, (1,1,3), padding=(0,0,1))
        self.conv8a_1 = l = nn.Conv3d(96, 96, (1,3,1), padding=(0,1,0))
        l = self.conv8a_2 = nn.Conv3d(96, 96, (3,1,1), padding=(1,0,0))
        l = self.convlx8 = nn.Conv3d(96+16, 75, 1, padding=0)

    def forward(self, x, atlas_hint, side_hint):
        x = F.relu(self.conv0a_0(x))
        x = F.relu(self.conv0a_1(x))
        x = self.conv0a_2(x)
        x = F.relu(self.bn0(x))
        li0 = x

        x = self.maxpool1(x)
        x = self.convf1(x)
        x = F.relu(self.bn1(x))
        li1 = x
        
        x = F.relu(self.convout2r(x))
        x = self.maxpool2(x)
        x = self.convout2(x)
        x = F.relu(self.bn2(x))
        li2 = x
        
        x = F.relu(self.convout3r(x))
        x = self.maxpool3(x)
        x = F.relu(self.convout3p(x))
        x = self.convout3(x)
        x = F.relu(self.bn3(x))
        x = self.maxpool3(x)
        
        atlas_hint = F.interpolate(atlas_hint[...,None,None,None], (3,6,6), mode="nearest")
        side_hint = F.interpolate(side_hint[...,None,None,None], (3,6,6), mode="nearest")
        x = torch.cat([x, atlas_hint, side_hint], dim=1)

        x = F.relu(self.convlx4(x))
        x = (self.convout4r(x))
        x = F.relu(self.bn4(x))
    
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.convlx5(x))
        x = (self.convout5r(x))
        x = F.relu(self.bn5(x))
        
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.convlx6(x))
        x = torch.cat([x, li2], dim=1)
        x = (self.convout6r(x))
        x = F.relu(self.bn6(x))

    
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.convlx7(x))
        x = torch.cat([x, li1], dim=1)
        x = (self.convout7r(x))
        x = F.relu(self.bn7(x))

    
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.conv8a_0(x))
        x = F.relu(self.conv8a_1(x))
        x = F.relu(self.conv8a_2(x))

        x = torch.cat([x, li0], dim=1)
        x = self.convlx8(x)
        x = torch.sigmoid(x)
        return x

net = ParcelCortexModel()
net.to(device)
net.eval()
# os.path.dirname(os.path.realpath(__file__)) + "/parcelcortex.pt")

net.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + "/parcelcortex.pt"))



atlas_codes = {"a2009": ([1,0,0], 75), "aseg": ([0, 1, 0], 35), "pals": ([0, 0, 1], 48)}
hemi_template_file = os.path.dirname(os.path.realpath(__file__)) + "/templates/hemi_mask.nii.gz"
roi = nibabel.load(hemi_template_file).get_data() > .5

if len(sys.argv) > 1:
    fnamel = sys.argv[1]
    assert("b96_box128_lout" in fnamel)

    if len(sys.argv) >= 3:
        atlas_list = sys.argv[2:]
        assert all([atlas in ["a2009", "aseg", "pals"] for atlas in atlas_list])
    else:
        atlas_list = ["a2009", "aseg", "pals"]

    T = time.time()
    for atlas in atlas_list:
        print("Applying model for atlas %s" % atlas)
        for fname in [fnamel, fnamel.replace("_lout_", "_rout_")]:
            img = nibabel.load(fname)

            d = img.get_fdata(caching="unchanged", dtype=np.float32)
            if d.max() > 10:
                d /= 255. # d is probably uchar encoded
            d_orr = d
            side_hint = [1, 0]

            if "_rout_" in fname:
                d_orr = d_orr[::-1].copy() # copy because pytorch fail at negative strides
                side_hint = side_hint[::-1]

            #print("Starting inference on %s using atlas %s" % (fname, atlas))
            atlas_code, nb_roi = atlas_codes[atlas]
            d_orr[~roi] = 0
            with torch.no_grad():
                out1 = net( torch.as_tensor(d_orr[None,None], device=device),
                            torch.as_tensor([atlas_code], dtype=torch.float32, device=device),
                            torch.as_tensor([side_hint], dtype=torch.float32, device=device) ).to("cpu")
            #print("Inference " + str(time.time() - T))
            out1 = np.asarray(out1)
            a=np.argmax(out1[:,:nb_roi], axis=1) + 1
            a[out1[:,:nb_roi].max(axis=1) < .001] = 0 # mostly for debug
            outt = a[0].astype(np.uint8)
            outt[~roi] = 0 # no need to fill too far

            if "_rout_" in fname:
                outt = outt[::-1]

            nibabel.Nifti1Image(outt, img.affine).to_filename( fname.replace(".nii.gz", "_outlab_%s_filled.nii.gz" % atlas))
