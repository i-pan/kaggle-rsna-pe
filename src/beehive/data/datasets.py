import gc
import glob
import os, os.path as osp
import pydicom
import random
import numpy as np
import torch
try:
    import vtk
    from vtk.util import numpy_support
except:
    print('module `vtk` is unavailable !')
from scipy.ndimage.interpolation import zoom
from torch.utils import data


NONETYPE = type(None)


class DICOMDataset(data.Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 weight=None,
                 window=None,
                 resize=None,
                 augment=None,
                 crop=None,
                 preprocess=None,
                 flip=False,
                 random_hu=0,
                 verbose=True,
                 test_mode=False,
                 return_name=False,
                 return_weight=False):
        self.inputs = inputs
        self.labels = labels
        self.weight = weight
        self.resize = resize
        self.window = window
        self.augment = augment
        self.crop = crop 
        self.preprocess = preprocess
        self.flip = flip
        self.random_hu = random_hu
        self.verbose = verbose
        self.test_mode = test_mode
        self.return_name = return_name
        self.return_weight = return_weight

    def __len__(self): return len(self.inputs)

    def process_image(self, X):
        if self.stacked:
            if self.resize: 
                X = np.asarray([self.resize(image=_)['image'] for _ in X])
                X = X.transpose(1, 2, 0)
        else:
            if self.resize: X = self.resize(image=X)['image']
        if self.random_hu > 0 and not self.test_mode:
            X = X + np.random.normal(loc=0, scale=self.random_hu)
        if type(self.window) != NONETYPE:
            X = self.apply_window(X)
        if self.augment: X = self.augment(image=X)['image']
        if self.crop: X = self.crop(image=X)['image']
        if self.preprocess: X = self.preprocess(X)
        return X.transpose(2, 0, 1)

    @staticmethod
    def flip_array(X):
        # X.shape = (C, H, W)
        if random.random() > 0.5:
            X = X[:, :, ::-1]
        if random.random() > 0.5:
            X = X[:, ::-1, :]
        if random.random() > 0.5 and X.shape[-1] == X.shape[-2]:
            X = X.transpose(0, 2, 1)
        X = np.ascontiguousarray(X)
        return X

    @staticmethod
    def read_dicom(dcmfile):
        if 'placeholder' in dcmfile:
            return np.zeros((512,512))-1000
        D = pydicom.dcmread(dcmfile)
        m = float(D.RescaleSlope)
        b = float(D.RescaleIntercept)
        D = D.pixel_array.astype('float')*m
        D = D.astype('float')+b
        return D

    def apply_window(self, X):
        # [50, 350] conventional mediastinal window
        WL, WW = self.window
        upper, lower = WL+WW/2, WL-WW/2
        X = np.clip(X, lower, upper)
        X = X - np.min(X) 
        X = X / np.max(X)
        X = (X*255.0).astype('uint8')
        return X 

    def get(self, i):
        try:
            if isinstance(self.inputs[i], list):
                X = [self.read_dicom(j) for j in self.inputs[i]]
                assert len(X) == 3, 'Only 3 stacked images are currently supported'
                self.stacked = True
            else:
                X = self.read_dicom(self.inputs[i])
                X = np.repeat(np.expand_dims(X, axis=-1), 1, axis=-1)
                self.stacked = False
            return X
        except Exception as e:
            if self.verbose: print(e)
            return None

    def __getitem__(self, i):
        X = self.get(i)
        while type(X) == NONETYPE:
            if self.verbose: print('Failed to read {} !'.format(self.inputs[i]))
            i = np.random.randint(len(self))
            X = self.get(i)

        X = self.process_image(X)

        if self.flip and not self.test_mode:
            X = self.flip_array(X)

        X = torch.tensor(X).float()
        y = torch.tensor(self.labels[i])
        if self.return_name: 
            return X, y, self.inputs[i]
        
        if self.return_weight: 
            return X, (y,torch.tensor(self.weight[i]).float())
            
        return X, y


class SeriesDataset(DICOMDataset):

    def __init__(self, *args, **kwargs):
        self.num_slices = kwargs.pop('num_slices', 64)
        self.repeat_3ch = kwargs.pop('repeat_3ch', True)
        self.stack_3ch = kwargs.pop('stack_3ch', False)
        assert self.repeat_3ch + self.stack_3ch <= 1
        self.bzchw = kwargs.pop('bzchw', False)
        self.truncate = kwargs.pop('truncate', None)
        super().__init__(*args, **kwargs)

    def process_image(self, X):
        # X.shape = (Z, H, W)
        if len(X) > self.num_slices:
            scale = self.num_slices / float(len(X))
            X = zoom(X, [scale, 1., 1.], order=0, prefilter=False)
        elif len(X) < self.num_slices:
            padding = np.expand_dims(np.zeros_like(X[0]), axis=0)
            padding[...] = np.min(X)
            padding = np.concatenate([padding]*(self.num_slices-len(X)), axis=0)
            X = np.concatenate([X, padding], axis=0)
        if self.stack_3ch:
            X = np.asarray([X[i:i+3] for i in range(len(X)-2)])
            # X.shape = (Z-2, C, H, W)
            X = X.transpose(0, 2, 3, 1)
        if self.resize: 
            X = np.asarray([self.resize(image=_)['image'] for _ in X])
        if self.random_hu > 0:
            X = X + np.random.normal(loc=0, scale=self.random_hu)
        if type(self.window) != NONETYPE:
            X = self.apply_window(X)
        if self.repeat_3ch: X = np.repeat(np.expand_dims(X, axis=-1), 3, axis=-1)
        # X.shape (Z, H, W, C)
        if self.augment: 
            to_augment = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_augment.update({'image': X[0]})
            augmented = self.augment(**to_augment)
            X = np.asarray([augmented['image']] + [augmented['image{}'.format(_)] for _ in range(1,len(X))])
        if self.crop: 
            to_crop = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_crop.update({'image': X[0]})
            cropped = self.crop(**to_crop)
            X = np.asarray([cropped['image']] + [cropped['image{}'.format(_)] for _ in range(1,len(X))])
        if self.preprocess: X = self.preprocess(X)        # X.shape = (Z, H, W)
        if X.ndim == 3: X = np.expand_dims(X, axis=-1)
        if self.bzchw:
            X = X.transpose(0, 3, 1, 2)
            # X.shape (Z, C, H, W)
        else:
            X = X.transpose(3, 0, 1, 2)
            # X.shape (C, Z, H, W)
        return X

    def get(self, i):
        try:
            dicom_files = self.inputs[i]
            if self.truncate:
                dicom_files = dicom_files[:int(self.truncate*len(dicom_files))]
            X = np.asarray([self.read_dicom(j) for j in dicom_files])
            return X
        except Exception as e:
            if self.verbose: print(e)
            return None

    @staticmethod
    def flip_array(X):
        # X.shape = (C, Z, H, W)
        if random.random() > 0.5:
            X = X[:, :, :, ::-1]
        if random.random() > 0.5:
            X = X[:, :, ::-1, :]
        if random.random() > 0.5:
            X = X[:, ::-1, :, :]
        if random.random() > 0.5 and X.shape[-1] == X.shape[-2]:
            X = X.transpose(0, 1, 3, 2)
        X = np.ascontiguousarray(X)
        return X

    def __getitem__(self, i):
        X = self.get(i)
        while type(X) == NONETYPE:
            if self.verbose: print('Failed to read {} !'.format(self.inputs[i]))
            i = np.random.randint(len(self))
            X = self.get(i)

        X = self.process_image(X)
        if self.flip and not self.test_mode and not self.bzchw:
            X = self.flip_array(X)

        X = torch.tensor(X).float()
        if self.return_name:
            return X, self.inputs[i]

        y = torch.tensor(self.labels[i])
        return X, y



class SimpleSeries(DICOMDataset):
    """For use w/ feature extraction.
    """
    def __init__(self, *args, **kwargs):
        self.add_padding = kwargs.pop('add_padding', True)
        _ = kwargs.pop('repeat_3ch', None)
        _ = kwargs.pop('num_slices', None)
        super().__init__(*args, **kwargs)

    def process_image(self, X):
        if self.resize: 
            X = np.asarray([self.resize(image=_)['image'] for _ in X])
        if self.random_hu > 0 and not self.test_mode:
            X = X + np.random.normal(loc=0, scale=self.random_hu)
        if type(self.window) != NONETYPE:
            X = self.apply_window(X)
        if self.augment: X = self.augment(image=X)['image']
        if self.crop: 
            X = np.asarray([self.crop(image=_)['image'] for _ in X])
        if self.preprocess: X = self.preprocess(X)
        # X.shape = (Z, H, W)
        return X

    def get(self, i):
        try:
            X = np.asarray([self.read_dicom(j) for j in self.inputs[i]])
            if self.add_padding:
                # Pad 1 empty slice top and bottom
                padding = np.expand_dims(np.zeros_like(X[0])-1000, axis=0)
                X = np.concatenate([padding, X, padding])
            # X.shape = (Z, H, W)
            return X
        except Exception as e:
            if self.verbose: print(e)
            return None

    def __getitem__(self, i):
        X = self.get(i)
        while type(X) == NONETYPE:
            if self.verbose: print('Failed to read {} !'.format(self.inputs[i]))
            i = np.random.randint(len(self))
            X = self.get(i)

        X = self.process_image(X)

        X = torch.tensor(X).float()
        return X, self.inputs[i]


class SeriesPredict(data.Dataset):

    def __init__(self,
                 inputs,
                 use_vtk=False):
        self.inputs = inputs
        if use_vtk:
            print('Using VTK to read DICOMs ...')
            self.load_dicom_array = self.load_with_vtk
            self.reader = vtk.vtkDICOMImageReader()
        else:
            print('Using pydicom to read DICOMS ...')
            self.load_dicom_array = self.load_with_pydicom

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def load_with_pydicom(f):
        dicom_files = np.asarray(glob.glob(osp.join(f, '*')))
        dicoms = [pydicom.dcmread(d) for d in dicom_files]
        M = float(dicoms[0].RescaleSlope)
        B = float(dicoms[0].RescaleIntercept)
        # Assume all images are axial
        z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
        dicoms = np.asarray([d.pixel_array for d in dicoms])
        dicoms = dicoms[np.argsort(z_pos)]
        dicoms = dicoms * M
        dicoms = dicoms + B
        return dicoms, dicom_files[np.argsort(z_pos)]

    def load_slice_with_vtk(self, fp):
        self.reader.SetFileName(fp)
        self.reader.Update()
        _extent = self.reader.GetDataExtent()
        ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
        ConstPixelSpacing = self.reader.GetPixelSpacing()
        imageData = self.reader.GetOutput()
        pointData = imageData.GetPointData()
        arrayData = pointData.GetArray(0)
        ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
        ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
        if ArrayDicom.ndim == 3 and ArrayDicom.shape[-1] == 1:
            ArrayDicom = ArrayDicom[...,0]
        # Rotate to match orientation when loading w/ pydicom
        ArrayDicom = np.rot90(ArrayDicom)
        return ArrayDicom, self.reader.GetImagePositionPatient()[-1]

    def load_with_vtk(self, f):
        dicom_files = np.asarray(glob.glob(osp.join(f, '*')))
        dicoms = [self.load_slice_with_vtk(d) for d in dicom_files]
        z_pos = [d[1] for d in dicoms]
        dicoms = np.asarray([d[0] for d in dicoms])
        dicoms = dicoms[np.argsort(z_pos)]
        return dicoms, dicom_files[np.argsort(z_pos)]

    def __getitem__(self, i):
        array, sorted_files = self.load_dicom_array(self.inputs[i])
        padding = np.expand_dims(np.zeros_like(array[0]), axis=0)
        padding[...] = np.min(array)
        array = np.concatenate([padding, array, padding])
        return array, list(sorted_files)


class FeatureDataset(data.Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 seq_len,
                 z_pos=None,
                 exam_level_label=False,
                 resample=False,
                 test_mode=False,
                 return_name=False,
                 reverse=False,
                 noise=None,
                 inv_sigmoid=False,
                 use_first_only=False,
                 return_weight=False):
        self.inputs = inputs
        self.labels = labels
        self.seq_len = seq_len
        self.z_pos = z_pos
        self.exam_level_label = exam_level_label
        self.resample = resample
        self.reverse = reverse
        self.test_mode = test_mode
        self.noise = noise
        self.return_name = return_name
        self.inv_sigmoid = inv_sigmoid
        self.use_first_only = use_first_only
        self.return_weight = return_weight

    def __len__(self): return len(self.inputs)

    def __getitem__(self, i):
        X, y = np.load(self.inputs[i]), self.labels[i]
        if self.use_first_only:
            X = np.expand_dims(X[:,0], axis=-1)
        if self.inv_sigmoid:
            X = np.log(1/(1-X))
        if not isinstance(self.z_pos, NONETYPE):
            z = self.z_pos[i]
            z = [z[j+1]-z[j] for j in range(len(z)-1)]
            z.append(z[-1])
            z = np.repeat(np.expand_dims(np.asarray(z), axis=1), 4, axis=1) / 10.
            X = np.concatenate((X,z), axis=1)

        if not self.test_mode and self.reverse and random.random()>0.5:
            X = np.ascontiguousarray(X[::-1])
            y = np.ascontiguousarray(y[::-1])

        wt = np.mean(self.labels[i])
        if self.noise and not self.test_mode:
            X *= np.random.normal(loc=1, scale=self.noise, size=X.shape)
        truncate_or_resample = not self.test_mode or self.resample
        if len(X) > self.seq_len and truncate_or_resample:
            if self.resample:
                # Resample using nearest interpolation
                scale = self.seq_len/len(X)
                X = zoom(X, [scale, 1.], order=0, prefilter=False)
                if not self.exam_level_label:
                    y = zoom(y, [scale], order=0, prefilter=False)
            else:
                # Truncate
                start = np.random.randint(0, len(X)-self.seq_len)
                X = X[start:start+self.seq_len]
                if not self.exam_level_label:
                    y = y[start:start+self.seq_len]

        if len(X) < self.seq_len and truncate_or_resample:
            diff = self.seq_len-len(X)
            mask = np.asarray([1]*len(X) + [0]*diff)
            padding = np.zeros_like(X[0])
            padding = np.expand_dims(padding, axis=0)
            padding = np.concatenate([padding]*diff, axis=0)
            X = np.concatenate([X, padding], axis=0)
            if not self.exam_level_label:
                y = np.concatenate([y, [0]*diff])
        else:
            mask = np.asarray([1]*len(X))

        X = torch.tensor(X).float()
        y = torch.tensor(y)
        mask = torch.tensor(mask).long()

        if self.return_name: 
            return (X,mask), self.inputs[i]

        if self.exam_level_label:
            return (X,mask), y

        if self.return_weight:
            return (X,mask), (y,mask,wt)

        return (X,mask), (y,mask)


class RVLVFeatures(data.Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 probas=None,
                 test_mode=False):
        self.inputs = inputs
        self.labels = labels
        self.probas = probas
        self.test_mode = test_mode

    def __len__(self): return len(self.inputs)

    def __getitem__(self, i):
        X, p, y = np.load(self.inputs[i]), self.probas[i], self.labels[i]

        X = torch.tensor(X).float()
        p = torch.tensor(p).float()
        y = torch.tensor(y)

        return (X,p), y


class ProbDataset(data.Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 test_mode,
                 **kwargs):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return torch.tensor(self.inputs[i]).float(), torch.tensor(self.labels[i])


class PEProbDataset(data.Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 test_mode=False,
                 return_name=False,
                 return_weight=True):
        self.inputs = inputs
        self.labels = labels
        self.test_mode = test_mode
        self.return_name = return_name
        self.return_weight = return_weight

    def __len__(self): return len(self.inputs)

    def __getitem__(self, i):
        X = np.load(self.inputs[i])
        y, w = X[-2], X[-1]
        X = X[:-2]

        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        w = torch.tensor(w).float()

        if self.return_name: 
            return X, self.inputs[i]

        if self.return_weight:
            if self.test_mode:
                return X, torch.tensor([y,w])
            return X, (y,w)

        return X,y

