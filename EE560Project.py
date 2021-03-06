# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 07:13:18 2017

@author: JChauvin
"""

import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

class ImageHandler:
    
    def __init__(self, Pv):
        self.trainingDatabase = [] # Filled with (orig, noisy) image pairs
        self.Pv = Pv
        
    def addImpulsiveNoise(self, img):
        numRows, numCols, nChannels = img.shape
        noisyImg = img.copy()
        # If Pv is the probability of a pixel being corrupted, then we need
        # to find the probability of an individual channel being corrupted
        Pv_i = 1 - np.cbrt(1 - self.Pv)
        # Loop over pixels
        for row in range(numRows):
            for col in range(numCols):
                # Loop over color channels
                for i in range(3):
                    if np.random.rand() < Pv_i:
                        # Channel value has 50%/50% chance of going to white or black
                        noisyImg[row, col, i] = 255 if np.random.rand() < 0.5 else 0
        return noisyImg
    
    def resizeImage(self, img):
        desiredSize = 256
        # Resize image to 256 rows while preserving aspect ratio
        if img.shape[0] == desiredSize:
            return img
        # Copied from https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
        r = desiredSize / float(img.shape[1])
        dim = (desiredSize, int(img.shape[0]*r))
        resizedImg = cv2.resize(img, dim)
        return resizedImg
    
    def createTrainingDatabase(self, imagePaths):
        for image in imagePaths:
            rawImg = cv2.imread(image)   # BGR order
            origImg = self.resizeImage(rawImg)
            noisyImg = self.addImpulsiveNoise(origImg)
            self.trainingDatabase.append((origImg, noisyImg))
        return
    
    def showTrainingDatabase(self):
        for imgPair in self.trainingDatabase:
            origImg = imgPair[0]
            noisyImg = imgPair[1]
            cv2.imshow('Original', origImg)
            cv2.waitKey(0)
            cv2.imshow('Noisy', noisyImg)
            cv2.waitKey(0)
        return
    
    def createDifferenceImage(self, origImg, filteredImg):
        diffImg = np.zeros((origImg.shape[0], origImg.shape[1]), np.uint8)
        for row in range(origImg.shape[0]):
            for col in range(origImg.shape[1]):
                diffImg[row, col] = np.linalg.norm(filteredImg[row,col,:].astype(float) - \
                                                   origImg[row,col,:].astype(float)) / float(origImg.shape[2])
        return diffImg.astype(np.uint8)
    
    def calcImageNorms(self, img):
        numRows, numCols, nChannels = img.shape
        imgNorms = np.zeros((numRows, numCols))
        for row in range(numRows):
            for col in range(numCols):
                imgNorms[row, col] = np.linalg.norm(np.array(img[row, col]))
        return imgNorms
        
    def calcAngle(self, pix1, pix2, pix1Norm, pix2Norm):
        # Copied from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
        # Account for measuring angles with black (i.e., [0,0,0]) pixels
        if pix1Norm == 0 or pix2Norm == 0:
            return 0
        u1 = np.array(pix1) / pix1Norm
        u2 = np.array(pix2) / pix2Norm
        return abs(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        #return np.abs(self.arccosApprox(np.dot(u1, u2)))
    
    def calculatePixelToPixelAngles(self, noisyImg, patchDims):
        # Determine array shapes
        nImgRows, nImgCols, nChannels = noisyImg.shape
        nPatchRows, nPatchCols = patchDims[0], patchDims[1]
        totalPixelNum = noisyImg.shape[0] * noisyImg.shape[1]
        borderLen = int((nPatchRows - 1) / 2)
        
        # Pad the input image to allow filtering of edge pixels
        paddedImg = cv2.copyMakeBorder(noisyImg, borderLen, borderLen, 
                                       borderLen, borderLen, 
                                       cv2.BORDER_REFLECT)
        
        # Calculate pixel norms for image
        imgNorms = self.calcImageNorms(paddedImg)
        
        # Initialize arrays
        pixToPixAngles = np.full(totalPixelNum * (nPatchRows*nPatchCols)**2, np.nan) 
        
        # For each pixel, calculate all of required the pixel-to-pixel angles
        angleIndex = 0
        for row in range(borderLen, nImgRows+borderLen):
            for col in range(borderLen, nImgCols+borderLen):
                patch = paddedImg[(row-borderLen):(row+borderLen+1), (col-borderLen):(col+borderLen+1)]
                patch1D = np.reshape(patch, (nPatchRows*nPatchCols, nChannels))
                patchNorms = imgNorms[(row-borderLen):(row+borderLen+1), (col-borderLen):(col+borderLen+1)]
                patchNorms1D = patchNorms.ravel()
                angleMatrix = np.full((nPatchRows*nPatchCols, nPatchRows*nPatchCols), np.nan)
                for i in range(nPatchRows*nPatchCols):
                    angleMatrix[i, i] = 0
                    for j in range(i+1, nPatchRows*nPatchCols):
                        angleMatrix[i, j] = self.calcAngle(patch1D[i], patch1D[j], patchNorms1D[i], patchNorms1D[j])
                        angleMatrix[j, i] = angleMatrix[i, j]
                pixToPixAngles[angleIndex:angleIndex+(nPatchRows*nPatchCols)**2] = angleMatrix.ravel()
                angleIndex += (nPatchRows*nPatchCols)**2
                     
        return pixToPixAngles
    
######################
class WVDF:
    
    def __init__(self, weightArrays):
        if len(np.array(weightArrays[0]).shape) == 1:
            dim = int(np.sqrt(len(weightArrays[0])))
            num = len(weightArrays)
            self.weightArrays = np.reshape(weightArrays, (num, dim, dim))
        else:
            self.weightArrays = weightArrays
        self.meanDiffs = np.zeros(len(self.weightArrays))
        self.bestWeights = None
    
    def applyFiltersAndCompareToOriginal(self, noisyImg, origImg, pixToPixAngles):
        numImgRows, numImgCols, nChannels = noisyImg.shape
        nPatchRows, nPatchCols = self.weightArrays[0].shape
        totalPixelNum = numImgRows * numImgCols
        borderLen = int((len(self.weightArrays[0]) - 1) / 2)
        # Pad the input image to allow filtering of edge pixels
        paddedImg = cv2.copyMakeBorder(noisyImg, borderLen, borderLen, 
                                       borderLen, borderLen, 
                                       cv2.BORDER_REFLECT)
        
        # Loop over pixels in padded image
        angleIndex = 0
        for row in range(borderLen, numImgRows+borderLen):
            for col in range(borderLen, numImgCols+borderLen):
                patch = paddedImg[(row-borderLen):(row+borderLen+1), (col-borderLen):(col+borderLen+1)]
                patch1D = np.reshape(patch, (nPatchRows*nPatchCols, nChannels))
                origPixVal = origImg[row-borderLen, col-borderLen]
                angleMatrix = np.reshape(pixToPixAngles[angleIndex:angleIndex+len(patch1D)**2], (nPatchRows**2, nPatchCols**2))
                # Loop over the weight arrays
                for w in range(len(self.weightArrays[:,0])):
                    weights = self.weightArrays[w,:]
                    weights1D = weights.ravel()
                    # Calculate the betas
                    betaArr = np.dot(angleMatrix, weights1D)
                    # Find the pixel corresponding to the minimum beta value
                    ind = list(np.where(betaArr == np.min(betaArr))[0])[0]
                    pixVal = patch1D[ind]
                    # Calculate mean pixel difference by dividing by nRows*nCols*nChannels
                    self.meanDiffs[w] += np.mean(np.abs(np.array(pixVal).astype(float) - np.array(origPixVal).astype(float))) / float(totalPixelNum)
                # Update angle index
                angleIndex += (nPatchRows * nPatchCols)**2
        return
    
    def applyBestFilterToImage(self, img, weights, pixToPixAngles):
        numPatchRows, numPatchCols = weights.shape
        numImgRows, numImgCols, nChannels = img.shape
        filteredImg = np.zeros((numImgRows, numImgCols, nChannels), np.uint8)
        borderLen = int((len(weights[0]) - 1) / 2)
        # Pad the input image to allow filtering of edge pixels
        paddedImg = cv2.copyMakeBorder(img, borderLen, borderLen, 
                                       borderLen, borderLen, 
                                       cv2.BORDER_REFLECT)
        
        # Loop over pixels in padded image
        angleIndex = 0
        for row in range(borderLen, numImgRows+borderLen):
            for col in range(borderLen, numImgCols+borderLen):
                #Loop over pixels within window
                patch = paddedImg[(row-borderLen):(row+borderLen+1), (col-borderLen):(col+borderLen+1)]
                patch1D = np.reshape(patch, (numPatchRows*numPatchCols, nChannels))
                angleMatrix = np.reshape(pixToPixAngles[angleIndex:angleIndex+(numPatchRows*numPatchCols)**2], (nPatchRows**2, nPatchCols**2))
                # Calculate the betas
                betaArr = np.dot(angleMatrix, weights.ravel())
                # Find the pixel corresponding to the minimum beta value
                ind = list(np.where(betaArr == np.min(betaArr))[0])[0]
                pixVal = patch1D[ind]
                # Update filtered image with resulting value
                filteredImg[row-borderLen, col-borderLen] = pixVal
                angleIndex += (numPatchRows*numPatchCols)**2
        return filteredImg

    
###############################################################################
if __name__ == '__main__':
    
    # Assign probability of impulse noise
    Pv = 0.10
    
    # Create ImageHandler object
    IH = ImageHandler(Pv)
    
    # Create training database
    #imageLoc = r'C:\Users\jchauvin\Dropbox (Physical Optics)\Grad School\EE 560\project\images\*'
    imageLoc = r'C:\Users\chauv\Dropbox (Physical Optics)\Grad School\EE 560\project\images\*'
    imagePaths = glob.glob(imageLoc)
    IH.createTrainingDatabase(imagePaths)
    
    # Display training database
    #IH.showTrainingDatabase()
    
    # Test image filtering
    nPatchRows, nPatchCols = (3,3)
    weights = np.ones((1, nPatchRows*nPatchCols))
    origImg = IH.trainingDatabase[0][0]
    noisyImg = IH.trainingDatabase[0][1]
    
    angleStartTime = time.clock()
    pixToPixAngles = IH.calculatePixelToPixelAngles(noisyImg, (nPatchRows, nPatchCols))
    angleEndTime = time.clock()
    print('Time to calculate angles: %.3f minutes' % ((angleEndTime-angleStartTime)/60.))
    filt = WVDF(weights)
    
    startTime = time.clock()
    filt.applyFiltersAndCompareToOriginal(noisyImg, origImg, pixToPixAngles)
    bestWeights = filt.weightArrays[list(filt.meanDiffs).index(min(filt.meanDiffs))]
    filteredImg = filt.applyBestFilterToImage(noisyImg, bestWeights, pixToPixAngles)
    endTime = time.clock()
    print('Time to apply weight array, evaluate, and apply best to image: %.3f minutes' % ((endTime-startTime)/60.))
    
    # Plot results
    plt.figure(1)#, figsize=(100,100))
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(origImg, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title('Original Image')
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(noisyImg, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title('Noisy Image')
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(filteredImg, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title('Filtered Image')
    plt.subplot(224)
    diffImg = IH.createDifferenceImage(origImg, filteredImg)
    plt.imshow(diffImg, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Image Difference')
    plt.show()

    print('Difference: %.3f' % (np.mean(np.abs(origImg.astype(float)-filteredImg.astype(float)))))
