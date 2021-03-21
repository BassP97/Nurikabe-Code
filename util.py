def getBoards(map_file):
    boards = []
    for line in enumerate(map_file):
        print(line)
        game = line.split(',')
        dimension = int(game[1])
        board = []
        for i in range(dimension):
            board.append([])
            for j in range(dimension):
                board[i].append(-1)
        for i in range(2, len(game), 3):
            if(game[i]!= ''):
                board[int(game[i])][int(game[i+1])] = int(game[i+2])
        boards.append(board)
    return boards