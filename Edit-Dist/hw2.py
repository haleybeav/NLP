import numpy as np
import sys
import random as rdm

# FUNCTION DEFINITIONS #########################################################
# calculates levenshtein edit distance between a target and source string
def distance(target, source, insertcost, deletecost, replacecost):
	n = len(target)+1
	m = len(source)+1
	# set up dist and initialize values
	dist = [[0 for j in range(m)] for i in range(n)]
	for i in range (1,n):
		dist[i][0] = dist[i-1][0] + insertcost
	for j in range (1,m):
		dist[0][j] = dist[0][j-1] + deletecost
	# align source and target strings
	for j in range(1,m):
		for i in range(1,n):
			inscost = insertcost + dist[i-1][j]
			delcost = deletecost + dist[i][j-1]
			if (source[j-1] == target[i-1]):
				add = 0
			else:
				add = replacecost
			substcost = add + dist[i-1][j-1]
			dist[i][j] = min(inscost, delcost, substcost)

			# increment ops decision made at this (i,j)
			opscost = [inscost, delcost, substcost]

	#return min edit distance
	return dist[n-1][m-1]

################################################################################
# recursively finds all paths through edit distance matrix of determined min edit distance
def backtrace(editTable, i, j):
	if i == 0 and j == 0:
		return [[(i, j)]]

	val  = editTable[i  ][j  ]
	left = editTable[i  ][j-1]
	up   = editTable[i-1][j  ]
	diag = editTable[i-1][j-1]

	paths = []
	if left <= val and j > 0:
		tpaths = backtrace(editTable, i, j-1)[:]
		tpaths = [path[:] for path in tpaths]
		[path.append((i, j)) for path in tpaths]
		paths += tpaths
	if up <= val and i > 0:
		tpaths = backtrace(editTable, i-1, j)[:]
		tpaths = [path[:] for path in tpaths]
		[path.append((i, j)) for path in tpaths]
		paths += tpaths
	if diag <= val and i > 0 and j > 0:
		tpaths = backtrace(editTable, i-1, j-1)[:]
		tpaths = [path[:] for path in tpaths]
		[path.append((i, j)) for path in tpaths]
		paths += tpaths

	return paths

################################################################################
# visual representation of alignment given coordinates in path and edit distance
def align(dist, path, target, source):
	# indices in source and target
	a = 0
	b = 0

	# visual display of alignment
	upr = ""
	mid = ""
	dwn = ""

	itrs = 0
	last_coor = False
	while not last_coor:
		coor_1 = path[itrs]
		i_1 = coor_1[0]
		j_1 = coor_1[1]

		coor_2 = path[itrs+1]
		i_2 = coor_2[0]
		j_2 = coor_2[1]

		cost = dist[i_2][j_2]-dist[i_1][j_1]

		# substitution
		if   cost == 2:
			upr += target[a]
			mid += " "
			dwn += source[b]
			a+=1
			b+=1
		# insertion or deletion
		elif cost == 1:
			# insert
			if j_1 == j_2:
				upr += target[a]
				mid += " "
				dwn += "_"
				a+=1
			# delete
			if i_1 == i_2:
				upr += "_"
				mid += " "
				dwn += source[b]
				b+=1
		# no operation
		else:
			upr += target[a]
			mid += "|"
			dwn += source[b]
			a+=1
			b+=1

		# update looping variable
		itrs+=1
		last_coor = (itrs == len(path)-1)

	# printing alignment
	upr+="\n"
	mid+="\n"

	return(upr+mid+dwn)

################################################################################
# calculates cost given coordinates in path, and edit distance
def cost(dist, path):
	last_coor = False
	itrs = 0
	cost = 0

	while not last_coor:
		coor_1 = path[itrs]
		i_1 = coor_1[0]
		j_1 = coor_1[1]

		coor_2 = path[itrs+1]
		i_2 = coor_2[0]
		j_2 = coor_2[1]

		cost += dist[i_2][j_2]-dist[i_1][j_1]

		# update looping variable
		itrs+=1
		last_coor = (itrs == len(path)-1)

	return cost

################################################################################
# constructs edit distance matrix
def construct(target, source):
	dimtarget = len(target)+1
	dimsource = len(source)+1

	# stores edit distances
	dist = np.zeros((dimtarget,dimsource))

	for i in range(dimtarget):
		for j in range(dimsource):
			dist[i,j] = distance(target[:i],source[:j],1,1,2)

	return (dist)

# CALCULATIONS #################################################################

# user input parameters
target = sys.argv[1]
source = sys.argv[2]

# number of alignments to print
nalign = 50;
if len(sys.argv) == 4:
	nalign = int(sys.argv[3])

# constructing edit distance matrix
dist = construct(target,source)

# finding all paths with previously calculated min levenshtein edit distance
dimx = dist.shape[0]-1
dimy = dist.shape[1]-1
paths = backtrace(dist, dimx, dimy)

# at corner-most coordinate in edit distance matrix
levenshtein_distance = dist[dimx][dimy]

# visual representation of a randomly chosen path through min edit distance
# matrix of previously calculated min levenshtein edit distance
aligned = align(dist, paths[rdm.randint(0,len(paths)-1)], target, source)

# EXERCISES ####################################################################
# 1
# printing a possible alignment and the min levenshtein distance
print("Levenshtein Distance = " + str(levenshtein_distance))
print(aligned)

print("-----------------------------------------------------------------------")

# 2
# printing n possible valid paths through min edit distance matrix of
# previously calculated min levenshtein edit distance, by default first 50 paths
for n in range(nalign):
	print("PATH #" + str(n+1) + " " + str(paths[n]))
