# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 08:37:53 2017

@author: chauv
"""
import os
import glob
import cv2
import time
import numpy as np
import datetime
import EE560Project as ee
import matplotlib.pyplot as plt

class GenAlg:
    
    def __init__(self, Ngenes, Np, Pc, Pm, Re, Pv, numIter):
        self.Ngenes = Ngenes # Equals length of weight array
        self.Np = Np
        self.Pc = Pc
        self.Pm = Pm
        self.Re = Re
        self.Pv = Pv
        self.numIter = numIter
        self.maxMAE = None
        self.numBits = 10
        
        self.fitnessArr = np.zeros(Np)
        self.numElites = self.myRound(self.Re*self.Np, base=2)
        self.lastGeneration = np.full((self.Np, self.Ngenes), np.nan)
        
        # Create the initial population
        self.createInitialPopulation()
        
    def myRound(self, x, base=2):
        # Copied from https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
        return int(base * round(float(x)/base))
    
    def encode(self, realNum):
        # Assuming real number accuracy of 10 bits
        newNum = int(1000 * realNum)
        binNum = bin(newNum)[2:].zfill(self.numBits)
        return binNum
    
    def encodeGene(self, realGene):
        # realGene is an array of real values
        binGene = ''
        for gene in realGene:
            binNum = self.encode(gene)
            binGene += binNum
        return binGene
    
    def decode(self, binNum):
        # Assuming real number accuracy of 10 bits
        realNum = int(binNum, 2) * (1. / float(2^self.numBits))
        return realNum
    
    def decodeChromosome(self, chromosome):
        realChromosome = []
        for i in range(0, self.Ngenes, self.numBits):
            binGene = chromosome[i:i+self.numBits]
            realGene = self.decode(binGene)
            realChromosome.append(realGene)
        return realChromosome
    
    def bitFlip(self, binNum, i):
        bit = '1' if binNum[i] == '0' else '0'
        newBin = binNum[2+i] + bit + binNum[i+1:] if i < len(binNum)-2 else binNum[2+i] + bit
        return newBin
        
    def createInitialPopulation(self):
        self.population = np.zeros((self.Np, self.Ngenes))
        for i in range(self.Np):
            chromosome = np.random.rand(self.Ngenes)
            self.population[i,:] = chromosome / np.sum(chromosome)
        return
            
    def performTournamentSelection(self):
        # Selects parents for the next generation based on binary tournament selection
        parentsArr = np.zeros((self.Np, self.Ngenes))
        # Perform binary tournament selection
        for i in range(self.Np):
            inds = np.random.randint(0, high=self.Np, size=2)
            parentsArr[i,:] = self.population[inds[0],:] if self.fitnessArr[inds[0]] > self.fitnessArr[inds[1]] else self.population[inds[1],:]
        return parentsArr
        
    def createNextGeneration(self, parentsArr):
        # Copy current generation into last generation
        self.lastGeneration = np.copy(self.population)
        # Perform elitism
        self.population[:self.numElites] = self.performElitism()
        # Permute the parents array
        parentsArr = np.random.permutation(parentsArr)
        # Create children
        for i in range(self.numElites, self.Np, 2):
            parent1, parent2 = parentsArr[i], parentsArr[i+1]
            if np.random.rand() < self.Pc:
                self.population[i], self.population[i+1] = self.performCrossover(parent1, parent2)
            else:
                self.population[i], self.population[i+1] = parent1, parent2
            # Apply mutation
            self.population[i] = self.performMutation(self.population[i])
            self.population[i+1] = self.performMutation(self.population[i+1])
        return
    
    def performElitism(self):
        # Use argsort with -1 times the fitness array to sort in descending order
        sortIndex = np.argsort(-self.fitnessArr)
        return self.lastGeneration[sortIndex[:self.numElites]]
    
    def performCrossover(self, parent1, parent2):
        if np.random.rand() < self.Pc:
            crossoverIndex = np.random.randint(0, high=self.Ngenes)
            child1 = list(parent1[:crossoverIndex]) + list(parent2[crossoverIndex:])
            child2 = list(parent2[:crossoverIndex]) + list(parent1[crossoverIndex:])
        else:
            child1 = parent1
            child2 = parent2
        return child1, child2
    
    def performMutation(self, child):
        for gene in range(len(child)):
            if np.random.rand() < self.Pm:
                child[gene] = np.random.rand()
                child = child/np.sum(child)
        return child
        
    def calcFitness(self, origImg, filteredImg):
        if self.maxMAE == None:
            self.maxMAE = np.mean(origImg) * (self.Pv)
        MAE = np.mean(np.abs(filteredImg.ravel().astype(float) - origImg.ravel().astype(float)))
        fitness = 1. - MAE / self.maxMAE
        return fitness
    
    def calcFitnesses(self, meanDiffs):
        if self.maxMAE == None:
            # Calculate the maximum possible error by assuming an error equal to
            # the mean pixel value times the total probability of impulse divided
            # by the number of channels
            self.maxMAE = np.mean(origImg.ravel()) * (self.Pv)
        self.fitnessArr = 1. - meanDiffs / self.maxMAE
        return
        
##############################################################################
if __name__ == '__main__':
    
    # Assign probability of impulse noise
    Pv = 0.10
    
    # Create ImageHandler object
    IH = ee.ImageHandler(Pv)
    
    # Create training database
    #folder = r'C:\Users\jchauvin\Dropbox (Physical Optics)\Grad School\EE 560\project'
    folder = r'C:\Users\chauv\Dropbox (Physical Optics)\Grad School\EE 560\project'
    imageLoc = folder + '\images\*'
    imagePaths = glob.glob(imageLoc)
    IH.createTrainingDatabase(imagePaths)
    
    # Perform GA optimization
    nPatchRows, nPatchCols = (3,3)
    weights = np.ones((nPatchRows, nPatchCols))
    Ngenes = len(weights.ravel())
    Np = 100
    Pc = 0.70
    Pm = 0.05
    Re = 0.10
    numIter = 40
    GA = GenAlg(Ngenes, Np, Pc, Pm, Re, Pv, numIter)
    
    # Create output folder
    outFolder = os.path.join(folder, 'results', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_Np_'+str(Np)+'_numIter_'+str(numIter))
    os.makedirs(outFolder)
    
    # Use genetic algorithm to find global optimum
    origImg = IH.trainingDatabase[0][0]
    noisyImg = IH.trainingDatabase[0][1]
    pixToPixAngles = IH.calculatePixelToPixelAngles(noisyImg, (nPatchRows, nPatchCols))
    fitnessArr = np.zeros(Np)
    bestFitnessArr = np.zeros(numIter)
    
    startTime = time.clock()
    for k in range(numIter):
        iterStartTime = time.clock()
        print('Iteration %d...' % k)
        filt = ee.WVDF(GA.population)
        filt.applyFiltersAndCompareToOriginal(noisyImg, origImg, pixToPixAngles)
        GA.calcFitnesses(filt.meanDiffs)
        iterEndTime = time.clock()
        print('Iteration %d time: %.3f minutes' % (k, (iterEndTime-iterStartTime)/60.))
        parentsArr = GA.performTournamentSelection()
        GA.createNextGeneration(parentsArr)
        bestFitnessArr[k] = np.max(GA.fitnessArr)
    endTime = time.clock()
    
    # Calculate fitness for the starting noisy image
    noisyImgFitness = GA.calcFitness(origImg, noisyImg)
        
    plt.figure(1)
    plt.plot(range(numIter), bestFitnessArr)
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.show()
    plt.savefig(outFolder + '\\Np_'+str(Np)+'_numIter_'+str(numIter)+'_diffPerIter.png')
        
    # Create final filtered image
    print('Producing final filtered image...')
    bestWeights = filt.weightArrays[list(filt.meanDiffs).index(min(filt.meanDiffs))]
    filteredImg = filt.applyBestFilterToImage(noisyImg, bestWeights, pixToPixAngles)
    
    # Calculate fitness for resulting image
    filteredImgFitness = GA.calcFitness(origImg, filteredImg)
    
    # Plot results
    plt.figure(2)#, figsize=(100,100))
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
    plt.savefig(outFolder + '\\Np_'+str(Np)+'_numIter_'+str(numIter)+'_results.png')

    print('Original Difference: %.3f' % (np.mean(np.abs(origImg.astype(float)-noisyImg.astype(float)))))
    print('Filtered Difference: %.3f' % (np.mean(np.abs(origImg.astype(float)-filteredImg.astype(float)))))
    print('Noisy image fitness: %.3f' % noisyImgFitness)
    print('Filtered image fitness: %.3f' % filteredImgFitness)
    print('Total runtime: %.3f minutes' % ((endTime-startTime)/60.))
    
    
    with open(outFolder+'\\Np_'+str(Np)+'_numIter_'+str(numIter)+'.txt', 'w') as f:
        f.write('Population size: %d\n' % Np)
        f.write('Number of iterations: %d\n' % numIter)
        f.write('Optimal weights:\n')
        f.write(str(bestWeights)+'\n')
        f.write('Difference: %.3f\n' % (np.mean(np.abs(origImg.astype(float)-filteredImg.astype(float)))))
        f.write('Noisy image fitness: %.3f\n' % noisyImgFitness)
        f.write('Filtered image fitness: %.3f\n' % filteredImgFitness)
        f.write('Total runtime: %.3f minutes\n' % ((endTime-startTime)/60.))
