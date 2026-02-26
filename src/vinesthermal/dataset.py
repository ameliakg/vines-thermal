"""Code for the FlirDataset image loading utilities. This reads in thermal
images that are saved at a specific file path, finds annotated RGB images
for training and validation (if using), extracts the RGB images from all 
thermal images in the dataset and prepares them to be segmented by the U-Net 
"""

# Import Dependencies
import numpy as np
import pandas as pd
import torch
import flyr
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from os import fspath
from PIL import Image

# FlirDataset Class:
class FlirDataset(torch.utils.data.Dataset):
    ""
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, k):
        return self.samples[k]
    def __init__(self, path, datatype='train', image_subsize=64, stride=None):
        """Finds saved images based on a filepath for training, testing,
        validation or non (images for segmentation by the model). Training,
        testing, and validation are FLIR RGB images that have been annotated
        with red to indicate plant material. Annotated images need to have the 
        same file name (minus the file formate suffix) as their corresponding 
        thermal images to be matched with one another.  RGB images for FLIR 
        images that the model is later used on do not need to be saved
        separately since they can be extracted from the thermal image. Divides
        images up into image patches for use by the model and builds up a
        dictionary of image patches whose pixels have been classified by
        the model.
        """ 
        from glob import glob
        from pathlib import Path
        path = Path(path)
        if datatype == "train":
            tt = "train"
        elif datatype == "test":
            tt = "test"
        elif datatype == "validation":
            tt = "validation"
        elif datatype is None:
            tt = "non"
        else:
            raise ValueError("datatype must be 'train', 'test', 'validation', or None")
        annot_pattern = str(path / "training" / "annotated" / tt / "*.png")
        annot_filenames = glob(annot_pattern)
        annot_ims = {
            Path(filename).name[:-4]: plt.imread(filename)
            for filename in annot_filenames}
        flir_pattern = str(path / "*" / "thermal" / "*.jpg")
        flir_filenames = {
            Path(file).name[:-4]: file
            for file in glob(flir_pattern)}
        self.names = list(annot_ims.keys())
        flir_ims = {
            key: self.load_flir(flir_filenames[key])
            for key in self.names}
        self.images = flir_ims
        self.annots = annot_ims
        #for (file, ims, annot) in zip(list(self.names()), list(self.images()), list(self.annots())):
            #if ims[1].shape[:2] != annot.shape[:2]:
               # plt.imshow(annot)
                #print(file, ims[1].shape, annot.shape)
        imsz = image_subsize
        stride = imsz if stride is None else stride
        sdata = {}
        # TODO: This needs to be tested for the rowidx and colidx hangover
        for key in self.names:
            (thr_im, opt_im) = self.images[key]
            ann_im = self.annots[key]
            for (rno, rowidx) in enumerate(range(0, opt_im.shape[0], stride)):
                if rowidx + imsz >= opt_im.shape[0]:
                    rowidx = opt_im.shape[0] - imsz 
                for (cno, colidx) in enumerate(range(0, opt_im.shape[1], stride)):
                    if colidx + imsz >= opt_im.shape[1]:
                        # If we get here, it means that there's a bit of the image
                        # dangling off the end!
                        opt_sub = opt_im[rowidx:rowidx + imsz, -imsz:]
                        thr_sub = thr_im[rowidx:rowidx + imsz, -imsz:]
                        ann_sub = ann_im[rowidx:rowidx + imsz, -imsz:]
                        tup = (rowidx, opt_im.shape[1] - imsz, opt_sub, ann_sub, thr_sub)
                    else:
                        # Get the subimage from the optical and annotation images:
                        opt_sub = opt_im[rowidx:rowidx + imsz, colidx:colidx + imsz]
                        thr_sub = thr_im[rowidx:rowidx + imsz, colidx:colidx + imsz]
                        ann_sub = ann_im[rowidx:rowidx + imsz, colidx:colidx + imsz]
                        tup = (rowidx, colidx, opt_sub, ann_sub, thr_sub)
                    sdata[key, rno, cno] = tup
        self.sample_data = sdata
        self.masks = {}
        self.image_subsize = image_subsize
        self.samples = []
        for ((k,rno,cno), tup) in sdata.items():
            (rowidx, colidx, opt_sub, ann_sub, thr_sub) = tup
            #plant_pixels = np.all(ann_sub[:,:,0:3] == [1, 0, 0], axis=2)
            plant_pixels = (ann_sub[:,:,0] - ann_sub[:,:,1] - ann_sub[:,:,2] > 0.9)
            self.masks[k, rno, cno] = plant_pixels
            opt_for_torch = torch.permute(
                torch.tensor(opt_sub, dtype=torch.float) / 255,
                (2, 0, 1))
            ann_frac = 1 - np.sum(plant_pixels) / plant_pixels.size
            #ann_frac = torch.tensor(
            #    round(ann_frac * 999),
            #    dtype=torch.long)
            ann_frac = torch.tensor(ann_frac, dtype=torch.float)
            #sample = (opt_for_torch, ann_frac)
            mask = torch.tensor(self.masks[k, rno, cno], dtype=torch.float32)
            sample = (opt_for_torch, mask[None,...])
            self.samples.append(sample)
    def load_flir(self, filename, thermal_unit='celsius'):
        """Loads and returns the portion of a FLIR image file that contains both
        optical and thermal data.
        
        Parameters
        ----------
        filename : pathlike
            A ``pathname.Path`` object or a string representing the filename of
            image that is to be loaded.
        thermal_unit : {'celsius' | 'kelvin' | 'fahrenheit'}, optional
            What temperature units to return; the default is ``'celsius'``.
            
        Returns
        -------
        optical_image : numpy.ndarray
            An image-array with shape ``(rows, cols, 3)`` containing the RGB
            optical of the visual FLIR image.
        thermal_image : numpy.ndarray
            An image-array with shape ``(rows, cols)`` containing the thermal
            values in Celsius.
        """
        # Make sure we have a path:
        filename = fspath(filename)
        # Import the raw image data:
        flir_image = flyr.unpack(filename)
        # Extract the optical and thermal data:
        opt = flir_image.optical
        #plt.imshow(opt)
        thr = getattr(flir_image, thermal_unit)
        pip = flir_image.pip_info
        x0 = pip.offset_x
        y0 = pip.offset_y
        ratio = pip.real_to_ir
        ratio = opt.shape[0] / thr.shape[0] / ratio
        # Resize the thermal image to match the optical image in resolution:
        (opt_rs, opt_cs, _) = opt.shape
        (thr_rs, thr_cs) = np.round(np.array(thr.shape) * ratio).astype(int)
        thr = np.array(Image.fromarray(thr).resize([thr_cs, thr_rs]))
        #plt.imshow(thr)
        x0 = round(opt_cs // 2 - thr_cs // 2 + x0)
        y0 = round(opt_rs // 2 - thr_rs // 2 + y0)
        return (thr, opt[y0:y0+thr_rs, x0:x0+thr_cs, :])
    def pred_all(self, model):
        """Returns predicted segmentations for all image patches in
        the dataset."""
        shape = (self.image_subsize, self.image_subsize)
        inpts = torch.stack([img for (img,_) in self.samples if img.shape[1:] == shape], axis=0).detach()
        targs = torch.stack([trg for (_,trg) in self.samples if trg.shape[1:] == shape], axis=0).detach()
        preds = model(inpts).detach()
        if model.logits:
            preds = torch.sigmoid(preds)
        return (
            torch.permute(inpts, (0,2,3,1)),
            preds[:, 0, ...],
            torch.permute(targs, (0,2,3,1)))
    def pred_images(self, model):
        """Returns predicted segmentations for all images in
        the dataset."""
        inpts, preds, targs = self.pred_all(model)
    def extract_temp(self, model):
        """Extracts temperature for each pixel predicted plant segmentation
        separates into what was in the plant segmentation "thermal_inseg"
        and non plant "thermal_outseg" """
        results = []
        for ((input, _), sdata) in zip(self.samples, self.sample_data.values()):
            thermal_im = sdata[-1]
            pred = model(input[None,...]) #need none bc model expecting batch, so gives it a batch dimension
            # prediction is > 0.5 if pred is > 0 because pred is in logits and 
            # sigmoid() converts 0 to 0.5.
            # So the line that follows is equivalent to:
            # pred = torch.sigmoid(pred[0]) > 0.5
            pred = pred[0, 0, ...] > 0
            #print(pred.shape, type(thermal_im), input)
            thermal_inseg = thermal_im[pred].flatten()
            thermal_outseg = thermal_im[~pred].flatten()
            results.append((thermal_inseg, thermal_outseg))
        return results
    def image_temps(self, model):
        """For each file, compiles the data from all image patches
        and returns a DataFrame with images names matched to the mean temperatures
        of the plant material "plant_temp" and none plant material
        "none_temp" """
        (_,preds,_) = self.pred_all(model)
        result = {}
        for (pred, ((file,r,c),sdata)) in zip(preds, self.sample_data.items()):
            thermal = sdata[-1]
            plant_temp = torch.sum(pred*thermal)
            notpred = 1-pred
            none_temp = torch.sum(notpred*thermal)
            if file not in result: 
                result[file] = []
            result[file].append((r, c, plant_temp, none_temp, pred, notpred))
        df = []
        for file, patches in result.items():
            plant_temp = torch.sum(
                torch.stack([t for (_,_,t,_,_,_) in patches]))
            plant_temp /= torch.sum(
                torch.stack([w for (_,_,_,_,w,_) in patches]))
            none_temp = torch.sum(
                torch.stack([t for (_,_,_,t,_,_) in patches]))
            none_temp /= torch.sum(
                torch.stack([w for (_,_,_,_,_,w) in patches]))
            df.append(
                {"file":file,
                 "plant_temp": float(plant_temp.detach()),
                 "none_temp":  float(none_temp.detach())})
        return pd.DataFrame(df)
