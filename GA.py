# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 08:37:53 2017

@author: chauv
"""
import glob
import cv2
import time
import numpy as np
import EE560Project as ee
import matplotlib.pyplot as plt

class GenAlg:
    
    def __init__(self, Ngenes, Np, Pc, Pm, Re, numIter):
        self.Ngenes = Ngenes # Equals length of weight array
        self.Np = Np
        self.Pc = Pc
        self.Pm = Pm
        self.Re = Re
        self.numIter = numIter
        self.maxMAE = None
        
        self.fitnessArr = np.zeros(Np)
        
        # Create the population
        self.createPopulation()
        
    def createPopulation(self):
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
        # Create children
        for i in range(0, self.Np, 2):
            parent1, parent2 = parentsArr[i], parentsArr[i+1]
            if np.random.rand() < self.Pc:
                self.population[i], self.population[i+1] = self.performCrossover(parent1, parent2)
            else:
                self.population[i], self.population[i+1] = parent1, parent2
        # Apply mutation
        self.population[i] = self.performMutation(self.population[i])
        self.population[i+1] = self.performMutation(self.population[i+1])
        return
    
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
            self.maxMAE = np.mean(origImg)
        MAE = np.mean(np.abs(filteredImg.astype(float) - origImg.astype(float)))
        fitness = 1. - MAE / self.maxMAE
        return fitness
        
##############################################################################
if __name__ == '__main__':
    
    # Assign probability of impulse noise
    Pv = 0.10
    
    # Create ImageHandler object
    IH = ee.ImageHandler(Pv)
    
    # Create training database
    folder = r'C:\Users\jchauvin\Dropbox (Physical Optics)\Grad School\EE 560\project'
    #folder = r'C:\Users\chauv\Dropbox (Physical Optics)\Grad School\EE 560\project'
    outFolder = folder + '\\results'
    imageLoc = folder + '\images\*'
    imagePaths = glob.glob(imageLoc)
    IH.createTrainingDatabase(imagePaths)
    
    # Perform GA optimization
    weights = np.ones((3,3))
    Ngenes = len(weights.ravel())
    Np = 30
    Pc = 0.70
    Pm = 0.05
    Re = 0.10
    numIter = 7 
    GA = GenAlg(Ngenes, Np, Pc, Pm, Re, numIter)
    
    # Use genetic algorithm to find global optimum
    origImg = IH.trainingDatabase[0][0]
    noisyImg = IH.trainingDatabase[0][1]
    fitnessArr = np.zeros(Np)
    bestFitnessArr = np.zeros(numIter)
    
    agents = 5
    chunksize = 3
    startTime = time.clock()
    for k in range(numIter):
        print('Iteration %d...' % k)
        for i in range(Np):
            print(GA.population[i,:])
            filt = ee.WVDF(GA.population[i,:])
            filteredImg = filt.applyFilterToImage(noisyImg)
            GA.fitnessArr[i] = GA.calcFitness(origImg, filteredImg)
        parentsArr = GA.performTournamentSelection()
        GA.createNextGeneration(parentsArr)
        bestFitnessArr[k] = np.max(GA.fitnessArr)
    endTime = time.clock()
        
    plt.figure(1)
    plt.plot(range(numIter), bestFitnessArr)
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.savefig(outFolder + '\\Np_'+str(Np)+'_numIter_'+str(numIter)+'_diffPerIter.png')
        
    # Calculate fitnesses of final population to find optimal weights
    print('Calculating optimal weights...')
#    for i in range(Np):
#        filt = ee.WVDF(GA.population[i,:])
#        filteredImg = filt.applyFilterToImage(noisyImg)
#        GA.fitnessArr[i] = GA.calcFitness(origImg, filteredImg)
    optInd = np.where(GA.fitnessArr == np.max(GA.fitnessArr))[0][0]
    bestWeights = GA.population[optInd,:]
    print(bestWeights)
    # Create final filtered image
    filt = ee.WVDF(bestWeights)
    filteredImg = filt.applyFilterToImage(noisyImg)
    
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

    print('Difference: %.3f' % (np.mean(np.abs(origImg.astype(float)-filteredImg.astype(float)))))
    print('Total runtime: %.3f minutes' % ((endTime-startTime)/60.))
    
    
    with open(outFolder+'\\Np_'+str(Np)+'_numIter_'+str(numIter)+'.txt', 'w') as f:
        f.write('Population size: %d\n' % Np)
        f.write('Number of iterations: %d\n' % numIter)
        f.write('Optimal weights:\n')
        f.write(str(bestWeights)+'\n')
        f.write('Difference: %.3f\n' % (np.mean(np.abs(origImg.astype(float)-filteredImg.astype(float)))))
        f.write('Total runtime: %.3f minutes\n' % ((endTime-startTime)/60.))
