def Func():
    filename = "movie_ids.txt"
    movieList = []
    with open(filename, 'rb') as f:
        for line in f:
            words = line.strip().split(maxsplit=1)
            movieList.append([words[0], words[1]])    
    # Store all movies in cell array movie{}
    return movieList