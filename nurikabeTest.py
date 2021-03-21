import matplotlib.pyplot as mpl
from board import boards
from board import board
import copy
import random
import sys
import numpy
import pandas as pd
import numpy as np
import matplotlib
import time
import multiprocessing
from multiprocessing import cpu_count
from lyingPool import NonDaemonPool as Pool
import os
matplotlib.use('Agg')
debug = False
perfDebug = False
numThreads = cpu_count()
neighborhoodThreads = 4


def threadedNeighborhoods(n, inputBoard, timeVar):
    start = time.time()
    if n == 1:
        inputBoard.surplusIslandSwap()
        timeVar += time.time()-start
    elif n == 2:
        inputBoard.swapBlackBlock()
        timeVar += time.time()-start
    elif n == 3:
        inputBoard.regenIsland()
        timeVar += time.time()-start
    elif n == 4:
        inputBoard.fixWall()
        timeVar += time.time()-start
    return inputBoard, timeVar


def processWrapper(start, fitEvalBoardName, fitEvalIterationNum, fitEvalUpdateTime, fitEvalUpdateValue, tempFitEval, tempFitTime,
                   evalFitForConvergGraph, timeForConvergGraph, fitPopID, fitIterNum, initialBoardFit, boardSize, popSize, fitData, solvedBoard,
                   pop, outputcsvFileName, startIndex, minFit, numIterations, getFitEvalNum, j, n, popStart):
    subPool = Pool(neighborhoodThreads)
    startLoc = int((popSize//numThreads)*popStart)
    if debug:
        print("starting search at index ", startLoc)
    bestBoard = pop[startLoc]
    startTime = time.time()
    endLoc = int((popSize//numThreads)*(popStart+1))
    popToConsider = pop[startLoc:endLoc]
    shouldSort = random.randint(1, 3)
    if (shouldSort == 1):
        popToConsider.sort(key=lambda x: x.getFitness())
    elif (shouldSort == 2):
        popToConsider.sort(key=lambda x: x.getFitness(), reverse=True)
    for i in range(endLoc-startLoc-1):
        fitData.append([])
        startFit = popToConsider[i].getFitness()
        if popToConsider[i].getFitness() < minFit:
            fitEvalBoardName.append(j+startIndex)
            fitEvalIterationNum.append(n)
            fitEvalUpdateTime.append(getFitEvalNum)
            fitEvalUpdateValue.append(popToConsider[i].getFitness())
            tempFitEval.append(popToConsider[i].getFitness())
            tempFitTime.append(getFitEvalNum)
            evalFitForConvergGraph.append(popToConsider[i].getFitness())
            timeForConvergGraph.append(getFitEvalNum)
            fitIterNum.append(0)
            fitPopID.append(i)
            initialBoardFit.append(startFit)
            boardSize.append(popToConsider[i].getSize())
            bestBoard = popToConsider[i]
            minFit = bestBoard.getFitness()
        iterStartTime = time.time()
        totalSearchTime = 0.0
        meanSearchTime = 0.0
        noImprovement = 0
        timeArr = [0.0, 0.0, 0.0, 0.0]
        for k in range(numIterations):
            aggArr = [(1, popToConsider[i], timeArr[0]), (2, popToConsider[i], timeArr[1]),
                      (3, popToConsider[i], timeArr[2]), (4, popToConsider[i], timeArr[3])]
            threadSearchStartTime = time.time()
            retBoards = []
            res = subPool.starmap(threadedNeighborhoods, aggArr)
            for b, tup in enumerate(res):
                retBoards.append(tup[0])
                timeArr[b] = tup[1]
            totalSearchTime += time.time()-threadSearchStartTime
            minFitness = popToConsider[i].getFitness()
            for board in retBoards:
                if (board.getFitness() < minFitness) or random.randint(1, 100) < 3:
                    popToConsider[i] = board
                    minFitness = board.getFitness()
            noImprovement += 1
            if popToConsider[i].getFitness() < minFit:
                noImprovement = 0
                fitEvalBoardName.append(j+startIndex)
                fitEvalIterationNum.append(n)
                fitEvalUpdateTime.append(getFitEvalNum)
                fitEvalUpdateValue.append(popToConsider[i].getFitness())
                tempFitEval.append(popToConsider[i].getFitness())
                tempFitTime.append(getFitEvalNum)
                evalFitForConvergGraph.append(popToConsider[i].getFitness())
                timeForConvergGraph.append(getFitEvalNum)
                fitIterNum.append(k)
                fitPopID.append(i)
                initialBoardFit.append(startFit)
                boardSize.append(popToConsider[i].getSize())
                minFit = popToConsider[i].getFitness()
                bestBoard = copy.deepcopy(popToConsider[i])
            fitData[i].append(popToConsider[i].getFitness())
            if noImprovement >= numIterations//7:
                popToConsider[i].shake()
            if bestBoard.getFitness() <= 0.0:
                break
        if perfDebug:
            print("\niteration took", round(time.time()-iterStartTime, 2), "seconds \nof that ",
                  round(totalSearchTime, 2), "seconds was spend on searching\nThe mean time per search was ", totalSearchTime/numIterations)
            print("Fix wall ate up ",
                  ((timeArr[0]/sum(timeArr))*100), " percent of the search time")
            print("swap ate up ",
                  ((timeArr[1]/sum(timeArr))*100), "many seconds")
            print("regen ate up ",
                  ((timeArr[2]/sum(timeArr))*100), "many seconds")
            print("surplus ate up ",
                  ((timeArr[3]/sum(timeArr))*100), "many seconds")
            print("", flush=True)
        if bestBoard.getFitness() <= 0.0:
            bestBoard.plotBoard(' '+str(j+startIndex))
            if debug:
                print("solution found!")
            break
    totalTime = time.time()-startTime
    plot_ga(evalFitForConvergGraph, str(j+startIndex)+' ' +
            str(n)+' '+str(os.getpid()), timeForConvergGraph)
    df = pd.DataFrame({'Board': fitEvalBoardName, 'ReCalc Number': fitEvalIterationNum, 'fitEvalTime': fitEvalUpdateTime, 'fitEvalValue': fitEvalUpdateValue,
                       'initial board fitness': initialBoardFit, 'iteration number': fitIterNum, 'Population ID': fitPopID, 'Board Size': boardSize})
    df.to_csv(outputcsvFileName, mode='a', header=False)
    return (bestBoard, totalTime)


def minFitFunc(fitData, iters):
    retArr = []
    maxLen = 0
    for i in range(iters):
        temp = []
        for arr in fitData:
            try:
                temp.append(arr[i])
            except:
                continue
        try:
            retArr.append(min(temp))
        except:
            retArr.append(0)
    return retArr


def avgFit(fitData, iters):
    retArr = []
    maxLen = 0
    for i in range(iters):
        temp = []
        for arr in fitData:
            try:
                temp.append(arr[i])
            except:
                continue
        total = 0
        for item in temp:
            total += item
        try:
            retArr.append(total/len(temp))
        except:
            retArr.append(0)
    return retArr


def plot_ga(fitUpdateValues, boardName, iterations):
    # print(fitUpdateValues)
    y_max = max(fitUpdateValues)
    fig = mpl.figure()
    ax = mpl.subplot(111)
    ax.plot(iterations, fitUpdateValues,
            label='Min fitness per func eval number')
    title = "GA Convergence" + str(boardName)
    mpl.xticks(iterations)
    mpl.title(title)
    mpl.ylim(0, y_max)
    mpl.xlabel('func evalpo number')
    mpl.ylabel('fitness value')
    ax.legend()
    mpl.savefig("convg"+boardName+".png")
    mpl.close(fig)
    return


def solveNurikabe(listOfBoards, outputcsvFileName, whichAmI, startIndex):
    debug = False
    numTries = 1
    solved = 0
    solvedBoards = []
    fitEvalBoardName = []
    fitEvalIterationNum = []
    fitEvalUpdateTime = []
    fitEvalUpdateValue = []
    fitPopID = []
    fitIterNum = []
    initialBoardFit = []
    boardSize = []
    df = pd.DataFrame({'Board': fitEvalBoardName, 'ReCalc Number': fitEvalIterationNum, 'fitEvalTime': fitEvalUpdateTime, 'fitEvalValue': fitEvalUpdateValue,
                       'initial board fitness': initialBoardFit, 'iteration number': fitIterNum, 'Population ID': fitPopID, 'Board Size': boardSize})
    df.to_csv(outputcsvFileName)
    pool = Pool(numThreads)
    for j, board in enumerate(listOfBoards):
        popSize = max(3000, (board.getSize()**4)*10)
        numIterations = max(2000, ((board.getSize()**3)*10))
        evalFitForConvergGraph = []
        timeForConvergGraph = []
        hits = 0
        for n in range(numTries):
            tempFitEval = []
            tempFitTime = []
            random.seed(n)
            refBoard = copy.deepcopy(board)
            pop = []
            bestBoard = copy.deepcopy(refBoard)
            boardSet = set()
            for i in range(popSize):
                temp = copy.deepcopy(refBoard)
                temp.shake()
                shakeTries = 0
                while temp.hash() in boardSet and shakeTries < 30:
                    temp.shake()
                    shakeTries += 1
                boardSet.add(temp.hash())
                temp.calculateFitness()
                if temp.getFitness() < bestBoard.getFitness():
                    bestBoard = temp
                pop.append(temp)
            print("number of unique boards for board number", j +
                  whichAmI, ":", len(set([b.hash() for b in pop])))
            minFit = bestBoard.getFitness()
            fitData = []
            solvedBoard = False

            getFitEvalNum = 0
            start = time.time()
            fitEvalBoardName.append(j+startIndex)
            fitEvalIterationNum.append(n)
            fitEvalUpdateTime.append(0)
            fitEvalUpdateValue.append(refBoard.getFitness())
            tempFitEval.append(refBoard.getFitness())
            tempFitTime.append(0)
            evalFitForConvergGraph.append(refBoard.getFitness())
            timeForConvergGraph.append(0)
            fitPopID.append(0)
            fitIterNum.append(0)
            initialBoardFit.append(refBoard.getFitness())
            boardSize.append(refBoard.getSize())
            popStart = 0
            manager = multiprocessing.Manager()
            args = [start, fitEvalBoardName, fitEvalIterationNum, fitEvalUpdateTime, fitEvalUpdateValue, tempFitEval, tempFitTime, evalFitForConvergGraph, timeForConvergGraph,
                    fitPopID, fitIterNum, initialBoardFit, boardSize, popSize, fitData, solvedBoard, pop, outputcsvFileName, startIndex, minFit, numIterations, getFitEvalNum, j, n, 0]
            argArr = []
            for b in range(numThreads):
                argCopy = copy.copy(args)
                argCopy[-1] = b
                argArr.append(tuple(argCopy))
            startTime = time.time()
            print('now solving board number ', j+whichAmI)
            boardsNTimes = pool.starmap(processWrapper, argArr)
            if debug:
                print(boardsNTimes)
            times = []
            hadHit = False
            for boardNTime in boardsNTimes:
                if boardNTime[0].getFitness() == 0.0:
                    times.append(boardNTime[1])
                    if not hadHit:
                        hits += 1
                    hadHit = True
            fitEvalBoardName = []
            fitEvalIterationNum = []
            fitEvalUpdateTime = []
            fitEvalUpdateValue = []
            initialBoardFit = []
            fitIterNum = []
            fitPopID = []
            boardSize = []
            if hadHit:
                print('solution found for board ', j+whichAmI, 'in ',
                      min(times), 'seconds on attempt number', n)
            else:
                print("solution not found for board ", j +
                      whichAmI, " on attempt number ", n)
        print('hit ratio for board', j+startIndex, ': ', hits/numTries)


def main():
    # clean out previous runs
    try:
        os.system("rm *.png")
    except:
        pass
    try:
        os.system("rm boardData.csv")
    except:
        pass
    # parameters
    boardFile = 'nurikabeRep.csv'
    outputcsvFileName = 'boardData.csv'
    print("executing on a system with", cpu_count(), "cpus")
    if not debug:
        boardType = input(
            "Enter 1 for big boards, 2 for representative subset, 3 for all boards: ")
        if boardType == '1':
            boardFile = 'bigBoards.csv'
        elif boardType == '2':
            boardFile = 'nurikabeRep.csv'
        elif boardType == '3':
            boardFile = "nurikabeFull.csv"
        else:
            raise ValueError("Please enter either 1 or 2 for data options")
    if debug:
        sys.stdout = open('debugNurikabeLarge.txt', 'w')
    listOfBoards = boards(boardFile)
    startIndex = 0
    whichAmI = 0
    for board in listOfBoards:
        board.plotBoard('Initial Board ')
    solveNurikabe(listOfBoards, outputcsvFileName, whichAmI, startIndex)


main()
