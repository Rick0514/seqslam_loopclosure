from pyseqslam.utils import AttributeDict
import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.image as mpimg
from PIL import Image 
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SeqSLAM():
    params = None
    
    def __init__(self, params):
        self.params = params

    def run(self):
        # begin with preprocessing of the images
        if self.params.DO_PREPROCESSING:
            results = self.doPreprocessing()
        
        # image difference matrix             
        if self.params.DO_DIFF_MATRIX:
            results = self.doDifferenceMatrix(results)
        
        # contrast enhancement
        if self.params.DO_CONTRAST_ENHANCEMENT:
            results = self.doContrastEnhancement(results)        
        else:
            if self.params.DO_DIFF_MATRIX:
                results.DD = results.D
        
        # find the matches
        if self.params.DO_FIND_MATCHES:
            results = self.doFindMatches(results)
        return results


    def findLoopClosure(self):
        # begin with preprocessing of the images
        if self.params.DO_PREPROCESSING:
            results = self.doPreprocessing()

        # image difference matrix
        if self.params.DO_DIFF_MATRIX:
            results = self.doDifferenceMatrix(results)

        # contrast enhancement
        if self.params.DO_CONTRAST_ENHANCEMENT:
            results = self.doContrastEnhancement(results)
        else:
            if self.params.DO_DIFF_MATRIX:
                results.DD = results.D

        # find the matches
        if self.params.DO_FIND_MATCHES:
            results = self.doFindLoop(results)
        return results

    def doPreprocessing(self):
        results = AttributeDict()
        results.dataset = []
        for i in range(len(self.params.dataset)):
            # shall we just load it?
            filename = '%s/preprocessing-%s%s.mat' % (self.params.savePath, self.params.dataset[i].saveFile, self.params.saveSuffix)
            if self.params.dataset[i].preprocessing.load and os.path.exists(filename):
                r = loadmat(filename)
                print('Loading file %s ...' % filename)
                d = AttributeDict()
                d.preprocessing = r['results_preprocessing']
                results.dataset.append(d)
            else:
                # or shall we actually calculate it?
                p = deepcopy(self.params)    
                p.dataset = self.params.dataset[i]
                d = AttributeDict()
                d.preprocessing = np.copy(SeqSLAM.preprocessing(p))
                results.dataset.append(d)
    
                if self.params.dataset[i].preprocessing.save:
                    results_preprocessing = results.dataset[i].preprocessing
                    savemat(filename, {'results_preprocessing': results_preprocessing})

        return results
    
    @staticmethod
    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    
    @staticmethod
    def preprocessing(params):
        print('Preprocessing dataset %s, indices %d - %d ...' % (params.dataset.name, params.dataset.imageIndices[0], params.dataset.imageIndices[-1]))
        # allocate memory for all the processed images
        n = len(params.dataset.imageIndices)
        m = params.downsample.size[0]*params.downsample.size[1] 
        
        if len(params.dataset.crop) > 0:
            c = params.dataset.crop
            m = (c[2]-c[0]+1) * (c[3]-c[1]+1)
        
        images = np.zeros((m,n), 'uint8')
        j=0
        
        # for every image ....
        for i in (params.dataset.imageIndices):
            filename = ('%s/%s%' + params.dataset.dataFormat + 'd%s%s') % (params.dataset.imagePath,
                                          params.dataset.prefix,
                                          i,
                                          params.dataset.suffix,
                                          params.dataset.extension)
            
            img = Image.open(filename)
            
            # convert to grayscale
            if params.DO_GRAYLEVEL:
                img = img.convert('L') #LA to include alpha
                # img = SeqSLAM.rgb2gray(np.asarray(img))
            
            # resize the image
            if params.DO_RESIZE:
                img = img.resize(params.downsample.size, params.downsample.method)
            
            img = np.copy(np.asarray(img))
            #img.flags.writeable = True
            
            # crop the image if necessary
            if len(params.dataset.crop) > 0:
                img = img[params.dataset.crop[1]:params.dataset.crop[3], params.dataset.crop[0]:params.dataset.crop[2]]            
            
            # do patch normalization
            if params.DO_PATCHNORMALIZATION:
                img = SeqSLAM.patchNormalize(img, params)            

            # shall we save the result?
            if params.DO_SAVE_PREPROCESSED_IMG:
                pass
            
            images[:,j] = img.flatten()
            j += 1
            
        return images
    
    
    @staticmethod
    def patchNormalize(img, params):
        s = params.normalization.sideLength    
        
        n = range(0, img.shape[0]+2, s)
        m = range(0, img.shape[1]+2, s)
            
        for i in range(len(n)-1):
            for j in range(len(m)-1):
                p = img[n[i]:n[i+1], m[j]:m[j+1]]
                
                pp=np.copy(p.flatten())

                # try:
                if params.normalization.mode != 0:
                    pp=pp.astype(float)
                    std = np.std(pp, ddof=1)
                    mean = pp-np.mean(pp)
                    img[n[i]:n[i+1], m[j]:m[j+1]] = 127+np.reshape(np.round(mean/(std+1e-6)), (s, s))
                else:
                    f = 255.0/np.max((1, np.max(pp) - np.min(pp)))
                    img[n[i]:n[i+1], m[j]:m[j+1]] = np.round(f * (p-np.min(pp)))

                # except RuntimeWarning:
                #     print(std)
                #print str((n[i], n[i+1], m[j], m[j+1]))
        img[img < 0] = 0
        img[img > 255] = 255
        return img
    
    def getDifferenceMatrix(self, data0preproc, data1preproc):
        # TODO parallelize
        n = data0preproc.shape[1]
        m = data1preproc.shape[1]
        D = np.zeros((n, m))   
    
        #parfor?
        for i in range(n):
            d = data1preproc - np.tile(data0preproc[:,i],(m, 1)).T
            D[i,:] = np.sum(np.abs(d), 0)/n
            
        return D
    
    def doDifferenceMatrix(self, results):
        filename = '%s/difference-%s-%s%s.mat' % (self.params.savePath, self.params.dataset[0].saveFile, self.params.dataset[1].saveFile, self.params.saveSuffix)  
    
        if self.params.differenceMatrix.load and os.path.exists(filename):
            print('Loading image difference matrix from file %s ...' % filename)
    
            d = loadmat(filename)
            results.D = d['D']
        else:
            if len(results.dataset)<2:
                print('Error: Cannot calculate difference matrix with less than 2 datasets.')
                return None
    
            print('Calculating image difference matrix ...')
    
            results.D=self.getDifferenceMatrix(results.dataset[0].preprocessing, results.dataset[1].preprocessing)
            
            # save it
            if self.params.differenceMatrix.save:                   
                savemat(filename, {'D':results.D})
            
        return results
    
    def enhanceContrast(self, D):
        # TODO parallelize
        DD = np.zeros(D.shape)
    
        #parfor?
        for i in range(D.shape[0]):
            a=np.max((0, i-self.params.contrastEnhancement.R/2))
            b=np.min((D.shape[0], i+self.params.contrastEnhancement.R/2+1))                                                        
            v = D[int(a):int(b), :]
            DD[i,:] = (D[i,:] - np.mean(v, 0)) / np.std(v, 0, ddof=1)  
        
        return DD-np.min(np.min(DD))
    
    def doContrastEnhancement(self, results):
        
        filename = '%s/differenceEnhanced-%s-%s%s.mat' % (self.params.savePath, self.params.dataset[0].saveFile, self.params.dataset[1].saveFile, self.params.saveSuffix)  
        
        if self.params.contrastEnhanced.load and os.path.exists(filename):
            print('Loading contrast-enhanced image distance matrix from file %s ...' % filename)
            dd = loadmat(filename)
            results.DD = dd['DD']
        else:
            print('Performing local contrast enhancement on difference matrix ...')
               
            # let the minimum distance be 0
            results.DD = self.enhanceContrast(results.D)

            # save it?
            if self.params.contrastEnhanced.save:                        
                DD = results.DD
                savemat(filename, {'DD':DD})
                
        return results
    
    def doFindMatches(self, results):
     
        filename = '%s/matches-%s-%s%s.mat' % (self.params.savePath, self.params.dataset[0].saveFile, self.params.dataset[1].saveFile, self.params.saveSuffix)  
         
        if self.params.matching.load and os.path.exists(filename):
            print('Loading matchings from file %s ...' % filename)
            m = loadmat(filename)
            results.matches = m['matches']
        else:
            print('Searching for matching images ...')
            # make sure ds is dividable by two
            self.params.matching.ds = self.params.matching.ds + np.mod(self.params.matching.ds,2)
        
            matches = self.getMatches(results.DD)
            # save it
            if self.params.matching.save:
                savemat(filename, {'matches':matches})
            results.matches = matches

        return results

    def doFindLoop(self, results):
        filename = '%s/LoopClosure-%s-%s%s.mat' % (self.params.savePath, self.params.dataset[0].saveFile, self.params.dataset[1].saveFile, self.params.saveSuffix)
        if self.params.matching.load and os.path.exists(filename):
            print('Loading LoopClosure from file %s ...' % filename)
            m = loadmat(filename)
            results.matches = m['Loop']
        else:
            print('Searching for matching images ...')
            # make sure ds is dividable by two
            self.params.matching.ds = self.params.matching.ds + np.mod(self.params.matching.ds,2)

            matches = self.getLoopClosure(results.DD)
            # save it
            if self.params.matching.save:
                savemat(filename, {'Loop':matches})
            results.matches = matches

        return results

    def getMatches(self, DD):
        # TODO parallelize
        matches = np.nan*np.ones((DD.shape[1],2))    
        # parfor?
        for N in range(int(self.params.matching.ds/2), int(DD.shape[1]-self.params.matching.ds/2)):
            # find a single match
            
            # We shall search for matches using velocities between
            # params.matching.vmin and params.matching.vmax.
            # However, not every vskip may be neccessary to check. So we first find
            # out, which v leads to different trajectories:
                
            move_min = self.params.matching.vmin * self.params.matching.ds    
            move_max = self.params.matching.vmax * self.params.matching.ds    
            
            move = np.arange(int(move_min), int(move_max)+1)
            v = move.astype(float) / self.params.matching.ds
            
            idx_add = np.tile(np.arange(0, self.params.matching.ds+1), (len(v),1))
            idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T)
            
            # this is where our trajectory starts
            n_start = N + 1 - self.params.matching.ds/2    
            x= np.tile(np.arange(n_start , n_start+self.params.matching.ds+1), (len(v), 1))    
            
            # TODO idx_add and x now equivalent to MATLAB, dh 1 indexing
            # add a line of inf costs so that we penalize running out of data
            DD=np.vstack((DD, np.infty*np.ones((1,DD.shape[1]))))

            # I think y_max should be min(N + self.param, DD.shape[0])
            y_max = min(N + self.params.matching.ds//2, DD.shape[0])
            # y_max = DD.shape[0]     # n+1
            xx = (x-1) * y_max

            score = np.zeros(y_max)
            # score = np.zeros(DD.shape[0])
            flatDD = DD.flatten('F')
            for s in range(1, y_max):   # 1~n+1
                y = np.copy(idx_add+s)
                y[y>y_max]=y_max     
                idx = (xx + y).astype(int)
                ds = np.sum(flatDD[idx-1],1)
                score[s-1] = np.min(ds)

            # for vmin~vmax find a vel that score is minimum
            
            # find min score and 2nd smallest score outside of a window
            # around the minimum 
            
            min_idx = np.argmin(score)
            min_value=score[min_idx]
            window = np.arange(np.max((0, min_idx-self.params.matching.Rwindow/2)), np.min((len(score), min_idx+self.params.matching.Rwindow/2)))
            not_window = list(set(range(len(score))).symmetric_difference(set(window))) #xor
            if len(not_window) == 0:
                continue
            min_value_2nd = np.min(score[not_window])
            match = [min_idx + self.params.matching.ds/2, min_value / min_value_2nd]
            matches[N,:] = match
                
        return matches
    

    def getLoopClosure(self, DD):
        # change a littlt bit: loop means can't not match the index itself
        n, n = DD.shape
        move_min = int(self.params.matching.vmin * self.params.matching.ds)
        move_max = int(self.params.matching.vmax * self.params.matching.ds)
        move = np.arange(move_min, move_max + 1)
        v = move.astype(float) / self.params.matching.ds

        if n < move_min + 2*self.params.matching.Rwindow + self.params.matching.no_seq + self.params.matching.ds or n < self.params.matching.ds:
            return None

        matches = np.nan * np.ones((n, 2))
        for N in range(move_min + 2*self.params.matching.Rwindow + self.params.matching.no_seq + self.params.matching.ds -1, n):

            # 0 ~ ds-1 --> ds items
            idx_add = np.tile(np.arange(0, self.params.matching.ds), (len(v), 1))
            idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T)

            idx_add1 = np.tile(idx_add[:,-1], (self.params.matching.ds, 1)).T - idx_add

            # this is where our trajectory starts
            n_start = N + 1 - self.params.matching.ds
            if n_start < self.params.matching.no_seq:
                continue

            x = np.tile(np.arange(n_start, n_start + self.params.matching.ds), (len(v), 1))

            # I think y_max should be min(N + self.param, DD.shape[0])
            y_max = n_start - self.params.matching.no_seq
            xx = x * n

            score = []
            final_idx = []
            flatDD = DD.flatten('F')
            for s in range(y_max+1 - move_min + 1):  # 0 ~ N+1-move_min
                y = np.copy(idx_add + s)
                y1 = np.copy(idx_add1 + s)
                y[y > y_max] = y_max
                y1[y1 > y_max] = y_max
                idx = (xx + y).astype(int)
                idx1 = (xx + y1).astype(int)
                ds = np.sum(flatDD[idx], 1)
                ds1 = np.sum(flatDD[idx1], 1)
                tmp = np.argmin(ds)
                tmp1 = np.argmin(ds1)
                if tmp <= tmp1:
                    score.append(ds[tmp])
                    final_idx.append(min(n-1, s + int(idx_add[tmp, -1])))
                else:
                    score.append(ds1[tmp1])
                    final_idx.append(s)

            score = np.array(score)
            min_idx = np.argmin(np.array(score))
            min_value = score[min_idx]
            window = np.arange(np.max((0, min_idx - self.params.matching.Rwindow // 2)),
                               np.min((len(score), min_idx + self.params.matching.Rwindow // 2)))
            not_window = list(set(range(len(score))).symmetric_difference(set(window)))  # xor
            if len(not_window) < self.params.matching.Rwindow:
                continue
            min_value_2nd = np.min(score[not_window])
            match = [final_idx[min_idx], min_value / min_value_2nd]
            matches[N, :] = match

        return matches
