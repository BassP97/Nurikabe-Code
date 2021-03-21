from collections import deque
import matplotlib.pyplot as plt
import random
import time
import uuid
import copy
import sys
import numpy as np
import matplotlib
import time
matplotlib.use('Agg')
debug = False

def neighbors(x, y, dimension):
    retArr = []
    if x-1 > -1:
        retArr.append((x-1, y))
    if x+1 < dimension:
        retArr.append((x+1, y))
    if y-1 > -1:
        retArr.append((x, y-1))
    if y+1 < dimension:
        retArr.append((x, y+1))
    return retArr


class boards:
    def __init__(self, filename):
        self.games = {}
        self.gameNums = []
        self.currGame = 0
        map_file = open(filename, "r", encoding='utf-8')
        for j, line in enumerate(map_file):
            game = line.split(',')
            gameNum = int(game[0])
            dimension = int(game[1])
            integer_list = []
            coordinate_list = []
            for i in range(2, len(game), 3):
                if(game[i] != ''):
                    coordinate_list.append([int(game[i]), int(game[i+1])])
                    integer_list.append(int(game[i+2]))
            game = board(dimension, coordinate_list, integer_list, j)
            if debug:
                print('total fitness:', game.getFitness())
                game.showBoard()
                print('----------------------------------')
            self.gameNums.append(gameNum)
            self.games[gameNum] = game

    def getGame(self):
        try:
            return self.games[gameNum]
        except:
            raise KeyError("Game number invalid, game not found")

    def __iter__(self):
        return boardIterator(self)


class boardIterator:
    def __init__(self, boards):
        self._boards = boards
        self._index = 0

    def __next__(self):
        if self._index < (len(self._boards.gameNums)):
            result = self._boards.games[self._boards.gameNums[self._index]]
            self._index += 1
            return result
        raise StopIteration

# board key:
# -\d are integer cells
# -1 serve as black cells
# 0 serve as white cells


class board:
    def __init__(self, dimension, coordinates, integers, _boardID):
        self.fitNum = 0
        self.boardID = _boardID
        self.numWhiteCells = sum(integers)
        self.maxSize = dimension
        self.allIslands = set()
        self.islandsWithTooFew = set()
        self.islandsWithTooMany = set()
        self.ints = integers
        if debug:
            print("placing", integers)
        self.coor = coordinates
        whiteCellsLeft = self.numWhiteCells
        whiteCellsLeft = self.resetBoard(dimension, coordinates, integers)
        while True:
            if self.createBoard(whiteCellsLeft):
                break
            self.resetBoard(dimension, coordinates, integers)
        if debug:
            print("fitness from init")
        self.calculateFitness()

    def createBoard(self, whiteCellsLeft):
        for i in range(whiteCellsLeft):
            placed = False
            numIterations = 0
            while not placed:
                numIterations += 1
                placed = self.placeWhiteCell(
                    self.gameBoard, random.randint(0, self.maxSize-1))
                if numIterations > self.maxSize*50000:
                    return False
        return True

# ------------------------------ UTILITY METHODS ------------------------------

    def hash(self):
        modifiedGameBoard = []
        for i in range(len(self.gameBoard)):
            modifiedGameBoard.append([])
            for j in range(len(self.gameBoard[i])):
                modifiedGameBoard[i].append(self.gameBoard[i][j][0])
        #kinda gross, but easiest way to get a hash of a gameboard I guess
        #maybe fix later?
        return str(modifiedGameBoard)

    def resetBoard(self, dimension, coordinates, integers):
        self.gameBoard = []
        whiteCellsLeft = self.numWhiteCells
        for i in range(dimension):
            self.gameBoard.append([])
            for j in range(dimension):
                self.gameBoard[i].append([-1, -1])
        for coordinate, integer in zip(coordinates, integers):
            self.gameBoard[coordinate[0]][coordinate[1]] = [
                integer, uuid.uuid1()]
            whiteCellsLeft -= 1
        return whiteCellsLeft

    def placeWhiteCell(self, board, row):
        for i in range(len(board[row])):
            neighborCells = neighbors(row, i, self.maxSize)
            canPlace = False
            for n in neighborCells:
                if board[n[0]][n[1]][0] >= 0 and not canPlace:
                    canPlace = True
                    island = board[n[0]][n[1]][1]
                elif board[n[0]][n[1]][0] >= 0 and canPlace and board[n[0]][n[1]][1] != island:
                    canPlace = False
                    break
            if canPlace:
                if board[row][i][0] < 0:
                    board[row][i] = [0, island]
                    return True
        return False

    def getSize(self):
        return self.maxSize

    def getID(self):
        return self.boardID

    def plotBoard(self, addlStr):
        temp = []
        for i in range(self.maxSize):
            temp.append([])
            for j in range(self.maxSize):
                temp[i].append(self.gameBoard[i][j][0])

        H = np.array(temp)
        size = self.maxSize
        x_start = 3.0
        x_end = 9.0
        y_start = 6.0
        y_end = 12.0

        extent = [x_start, x_end, y_start, y_end]

        # The normal figure
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        im = ax.imshow(H, extent=extent, origin='lower',
                       interpolation='None', cmap='gray')

        # Add the text
        jump_x = (x_end - x_start) / (2.0 * size)
        jump_y = (y_end - y_start) / (2.0 * size)
        x_positions = np.linspace(
            start=x_start, stop=x_end, num=size, endpoint=False)
        y_positions = np.linspace(
            start=y_start, stop=y_end, num=size, endpoint=False)

        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = H[y_index, x_index]
                text_x = x + jump_x
                text_y = y + jump_y
                ax.text(text_x, text_y, label, color='black',
                        ha='center', va='center')

        fig.colorbar(im)
        plt.savefig(str(self.boardID)+addlStr+'.png')
        plt.close()

    def showBoard(self):
        print('\n'.join([''.join(['{:4}'.format(item[0])
                                  for item in row]) for row in self.gameBoard]))

    def getWhitesFromIsland(self, x, y):
        if self.gameBoard[x][y][0] <= 0:
            self.showBoard()
            raise KeyError("Provided loc ", x, y,
                           " is not a numbered cell - see above board")
        queue = [(x, y)]
        visited = set()
        total = []
        while queue:
            temp = queue.pop()
            neighborCells = neighbors(temp[0], temp[1], self.maxSize)
            for n in neighborCells:
                if self.gameBoard[n[0]][n[1]][0] == 0 and (n[0], n[1]) not in visited:
                    queue.append((n[0], n[1]))
                    visited.add((n[0], n[1]))
                    total.append((n[0], n[1]))
        return total

    def checkIslands(self):
        numWhites = 0
        for i in range(self.maxSize):
            for j in range(self.maxSize):
                if self.gameBoard[i][j][0] >= 0:
                    numWhites += 1
                if self.gameBoard[i][j][0] == 0:
                    neighborCells = neighbors(i, j, self.maxSize)
                    isolated = True
                    for neighbor in neighborCells:
                        if neighbor[0] >= 0:
                            isolated = False
                            break
                    if isolated:
                        return False
        if numWhites < sum(self.ints):
            print("too few white cells")
            return False
        elif numWhites > sum(self.ints):
            print("too many white cells")
            return False
        for co in self.coor:
            if self.gameBoard[co[0]][co[1]][0] <= 0:
                if debug:
                    print(co[0], co[1], " messed up")
                return False
        return True

# ------------------------------ FITNESS METHODS ------------------------------
    def calculateFitness(self):
        self.fitNum += 1
        if debug:
            print("______________NEW CALC____________")
            self.showBoard()
        fourByFourBlocks = self.fourByFour()
        numFragments = self.wallFragments()
        islandSizeDiff = self.islandSize()
        rawFit = [fourByFourBlocks, numFragments, islandSizeDiff]
        weights = [3, 4, 1]
        weightedFit = [a*b for a, b in zip(rawFit, weights)]
        if debug:
            """
            print("\nfour by four: ", fourByFourBlocks)
            print("numFrags: ", numFragments)
            print("Island Size Difference: ", islandSizeDiff)
            """
            print("islands with too many:", [
                  island for island in list(self.islandsWithTooMany)])
            print("islands with too few:", [
                  island for island in list(self.islandsWithTooFew)], '\n')
        totalFitness = sum(weightedFit)
        self.fitness = totalFitness

    # checks if something is a part of a 4x4 square
    def isSurrounded(self, i, j):
        bl = False
        tl = False
        br = False
        tr = False
        # right squares
        if i+1 < self.maxSize:
            if self.gameBoard[i+1][j][0] == -1:
                # top right
                if j+1 < self.maxSize:
                    tr = self.gameBoard[i][j +
                                           1][0] == self.gameBoard[i+1][j+1][0] == -1
                # bottom right
                if j-1 >= 0:
                    br = self.gameBoard[i][j -
                                           1][0] == self.gameBoard[i+1][j-1][0] == -1
        # left squares
        if i-1 >= 0:
            if self.gameBoard[i-1][j][0] == -1:
                # top left
                if j+1 < self.maxSize:
                    tl = self.gameBoard[i][j +
                                           1][0] == self.gameBoard[i-1][j+1][0] == -1
                # bottom left
                if j-1 >= 0:
                    bl = self.gameBoard[i][j -
                                           1][0] == self.gameBoard[i-1][j-1][0] == -1
        return (tr or br or tl or bl)

    def fourByFour(self):
        totalSquares = 0
        for i in range(self.maxSize):
            for j in range(self.maxSize):
                if self.gameBoard[i][j][0] == -1 and self.isSurrounded(i, j):
                    totalSquares += 1
        return totalSquares/4

    def wallFragments(self):
        wallCoordList = []
        for i in range(self.maxSize):
            for j in range(self.maxSize):
                if self.gameBoard[i][j][0] == -1:
                    wallCoordList.append((i, j))
        wallCoordSet = set(wallCoordList)
        numFrags = 0

        while len(wallCoordSet) > 0:
            start = random.sample(wallCoordSet, 1)
            visited = set()
            queue = deque(start)
            while queue:
                curr = queue.pop()
                visited.add(curr)
                neighborCells = neighbors(curr[0], curr[1], self.maxSize)
                for neighbor in neighborCells:
                    if neighbor not in visited and self.gameBoard[neighbor[0]][neighbor[1]][0] == -1:
                        queue.append(neighbor)
            wallCoordSet = wallCoordSet.difference(visited)
            numFrags += 1
        return numFrags - 1

    def getIslandSize(self, x, y):
        visited = set()
        queue = deque()
        queue.append((x, y))
        while queue:
            curr = queue.pop()
            visited.add(curr)
            neighborCells = neighbors(curr[0], curr[1], self.maxSize)
            for neighbor in neighborCells:
                if neighbor not in visited and self.gameBoard[neighbor[0]][neighbor[1]][0] == 0:
                    queue.append(neighbor)
        if debug:
            print("island white squares for coordinates", x, y, ":")
            print(visited)
        return len(visited)

    def islandSize(self):
        self.islandsWithTooMany = set()
        self.islandsWithTooFew = set()
        totalDifference = 0
        intCoords = zip(self.ints, self.coor)
        leftover = set()
        for integer, coords in list(intCoords):

            intIsleSize = self.getIslandSize(coords[0], coords[1])
            if debug:
                print("\ncoords and integer: ", coords, integer)
                print("island size: ", intIsleSize)
                print(self.islandsWithTooMany)
                print(self.islandsWithTooFew)
            # keep track of islands with too few white squares
            if integer < intIsleSize:
                if debug:
                    print("island too large")
                self.islandsWithTooMany.add(tuple(coords))

            # keep track of islands with too many white squares
            elif integer > intIsleSize:
                if debug:
                    print("island too small")
                self.islandsWithTooFew.add(tuple(coords))

            # keep track of islands with enough white squares (goldilocks)
            else:
                leftover.add(tuple(coords))
            totalDifference += max(integer, intIsleSize) - \
                min(integer, intIsleSize)

        self.allIslands = self.islandsWithTooMany.union(
            self.islandsWithTooFew).union(leftover)
        return totalDifference

# ------------------------------ GETTER METHODS ------------------------------
    def getFitness(self):
        return self.fitness

    def getFitEvalNum(self):
        return self.fitNum

# ------------------------------ NEIGHBORHOOD METHODS -------------------------
    # choose a random island with more white cells, turn one of its edge cells black
    # then add a white cell to an island with fewer white cells

    def surplusIslandSwap(self):
        if len(self.islandsWithTooMany) == 0 or len(self.islandsWithTooFew) == 0:
            return
        numMaxIterations = 20
        for i in range(numMaxIterations):
            tooFew = random.sample(self.islandsWithTooFew, 1)
            tooMany = random.sample(self.islandsWithTooMany, 1)

            tooMany = (tooMany[0][0], tooMany[0][1])
            tooFew = (tooFew[0][0], tooFew[0][1])

            tooManyWhite = self.findWhiteThatDoesntSplitIsland(
                tooMany[0], tooMany[1])
            if tooManyWhite != -1:
                temp = self.gameBoard[tooManyWhite[0]][tooManyWhite[1]]
                self.gameBoard[tooManyWhite[0]][tooManyWhite[1]] = [-1, -1]
                # find a black cell that is only adjacent to one white
                tooFewID = self.gameBoard[tooFew[0]][tooFew[1]][1]
                tooFewWhite = self.findBlackNextToOneWhite(tooFewID)
                # if no such black cell exists, reset the board
                if tooFewWhite == -1:
                    self.gameBoard[tooManyWhite[0]][tooManyWhite[1]] = temp
                else:
                    self.gameBoard[tooFewWhite[0]
                                   ][tooFewWhite[1]] = [0, tooFewID]
                    break
        if debug:
            print("fitness from surplus")
        self.calculateFitness()

    def findWhiteThatDoesntSplitIsland(self, i, j):
        if self.getIslandSize(i, j) == 1:
            return -1
        visited = set()
        queue = deque()
        queue.append((i, j))
        while queue:
            random.shuffle(queue)
            curr = queue.pop()
            visited.add(curr)
            neighborCells = neighbors(curr[0], curr[1], self.maxSize)
            random.shuffle(neighborCells)
            for neighbor in neighborCells:
                if neighbor not in visited and self.gameBoard[neighbor[0]][neighbor[1]][0] == 0:
                    queue.appendleft(neighbor)
        return curr

    def findBlackNextToOneWhite(self, idNum):
        for i in range(self.maxSize):
            for j in range(self.maxSize):
                if self.gameBoard[i][j][0] == -1 and random.randint(0, 100) > 30:
                    foundNeighbor = self.notAdjMoreThanOneWhite(i, j, idNum)
                    if foundNeighbor:
                        return (i, j)
        return -1

    def notAdjMoreThanOneWhite(self, i, j, idNum=0):
        neighborCells = neighbors(i, j, self.maxSize)
        foundNeighbor = False
        for n in neighborCells:
            if self.gameBoard[n[0]][n[1]][0] != -1:
                if idNum == 0:
                    idNum = self.gameBoard[n[0]][n[1]][1]
                if idNum == self.gameBoard[n[0]][n[1]][1]:
                    foundNeighbor = True
            if self.gameBoard[n[0]][n[1]][0] != -1 and self.gameBoard[n[0]][n[1]][1] != idNum:
                foundNeighbor = False
                break
        return foundNeighbor

    def swapBlackBlock(self):
        if self.fourByFour() == 0 or not self.islandsWithTooMany:
            return
        for i in range(self.maxSize):
            for j in range(self.maxSize):
                if self.gameBoard[i][j][0] == 0:
                    idNum = self.gameBoard[i][j][1]
                    neighborCells = neighbors(i, j, self.maxSize)
                    for n in neighborCells:
                        if self.gameBoard[n[0]][n[1]][0] == -1 and self.isSurrounded(n[0], n[1]) and self.notAdjMoreThanOneWhite(n[0], n[1], idNum):
                            self.gameBoard[n[0]][n[1]] = [0, idNum]
                            try:
                                tooMany = random.sample(
                                    self.islandsWithTooMany, 1)
                            except:
                                self.gameBoard[n[0]][n[1]] = [-1, -1]
                                return
                            tooManyStart = (tooMany[0][0], tooMany[0][1])
                            tooManyWhite = self.findWhiteThatDoesntSplitIsland(
                                tooManyStart[0], tooManyStart[1])
                            self.gameBoard[tooManyWhite[0]
                                           ][tooManyWhite[1]] = [-1, -1]
                            self.calculateFitness()
                            return

    def fixWall(self):
        if self.wallFragments == 1:
            return
        blackWallSum = 0
        wallCoordList = []
        for i in range(self.maxSize):
            for j in range(self.maxSize):
                if self.gameBoard[i][j][0] == -1:
                    blackWallSum += 1
                    wallCoordList.append((i, j))

        wallCoordSet = set(wallCoordList)
        wallFrags = []
        while len(wallCoordSet) > 0:
            start = random.sample(wallCoordSet, 1)
            visited = set()
            queue = deque(start)
            while queue:
                curr = queue.pop()
                visited.add(curr)
                neighborCells = neighbors(curr[0], curr[1], self.maxSize)
                for neighbor in neighborCells:
                    if neighbor not in visited and self.gameBoard[neighbor[0]][neighbor[1]][0] == -1:
                        queue.append(neighbor)
            wallFrags.append(list(visited))
            wallCoordSet = wallCoordSet.difference(visited)

        sortedWalls = sorted(wallFrags, key=len)
        wallWeight = []
        for i in range(len(sortedWalls)):
            wallWeight.append(
                (blackWallSum-len(list(sortedWalls))/blackWallSum))
        # smallest wall is now a misnomer - the wall is randomly selected BUT
        smallestWall = random.choices(
            population=sortedWalls, weights=wallWeight, k=1)[0]
        swapID = 1
        blackSwap = -1
        for block in smallestWall:
            if self.notAdjMoreThanOneWhite(block[0], block[1]):
                blackSwap = block
                neighborCells = neighbors(block[0], block[1], self.maxSize)
                for n in neighborCells:
                    if self.gameBoard[n[0]][n[1]][0] == 0:
                        swapID = self.gameBoard[n[0]][n[1]][1]
                    break
        if blackSwap == -1:
            return
        intsAndCoords = zip(self.ints, self.coor)

        whiteSwap = random.sample(self.allIslands, 1)

        swapStart = (whiteSwap[0][0], whiteSwap[0][1])
        swapLoc = self.findWhiteThatDoesntSplitIsland(
            swapStart[0], swapStart[1])
        if swapLoc == -1:
            return
        self.gameBoard[swapLoc[0]][swapLoc[1]] = [-1, -1]
        self.gameBoard[blackSwap[0]][blackSwap[1]] = [0, swapID]
        self.calculateFitness()
        if debug:
            print("fitness from fix wall")

    def regenIsland(self):
        regen = random.sample(self.coor, 1)[0]
        toRegen = self.getWhitesFromIsland(regen[0], regen[1])
        whiteBudget = len(toRegen)
        idNum = self.gameBoard[regen[0]][regen[1]][1]
        for coo in toRegen:
            self.gameBoard[coo[0]][coo[1]] = [-1, -1]
        boardCopy = [[item for item in x] for x in self.gameBoard]
        queue = [regen]
        foundIsland = False
        while queue:
            random.shuffle(queue)
            currCell = queue.pop()
            neighborCells = neighbors(
                currCell[0], currCell[1], self.maxSize)
            random.shuffle(neighborCells)
            for n in neighborCells:
                if self.notAdjMoreThanOneWhite(n[0], n[1], idNum):
                    if whiteBudget > 0:
                        boardCopy[n[0]][n[1]] = [0, idNum]
                        whiteBudget -= 1
                        queue.append(n)
            if whiteBudget == 0:
                foundIsland = True
                break
        if foundIsland:
            self.gameBoard = boardCopy
        if debug:
            print("fitness from regen")
        self.calculateFitness()

    def shake(self):
        whiteCellsLeft = self.resetBoard(self.maxSize, self.coor, self.ints)
        while True:
            if self.createBoard(whiteCellsLeft):
                break
            self.resetBoard(self.maxSize, self.coor, self.ints)
        self.calculateFitness()
