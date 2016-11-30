import math 
import data
import numpy as np

truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0

trnSetTrue = []
trnSetFalse = []

#function 정의
def gaussianDistribution(meanArrTrue, stdArrTrue, meanArrFalse, stdArrFalse, value):
	pi = math.pi
	resultTrue = 1
	resultFalse = 1
	i=0
	while i < len(meanArrTrue):
		resultTrue *= 1/(math.sqrt(2*pi)*stdArrTrue[i])*math.exp(-1*math.pow(meanArrTrue[i]-value[i],2)/(2*math.pow(stdArrTrue[i],2)))
		i+=1
	i=0
	while i < len(meanArrFalse):
		resultFalse *= 1/(math.sqrt(2*pi)*stdArrFalse[i])*math.exp(-1*math.pow(meanArrFalse[i]-value[i],2)/(2*math.pow(stdArrFalse[i],2)))

		i+=1

	resultTrue *= len(meanArrTrue)/(len(meanArrTrue)+len(meanArrFalse))
	resultTrue *= len(meanArrFalse)/(len(meanArrTrue)+len(meanArrFalse))

	if resultTrue > resultFalse:
		return 1
	else:
		return 0

def multivariateDistribution(meanArrTrue, meanArrFalse, covTrue, covFalse, value):
    pi = math.pi
    valueT = value.transpose()
    covTrueInverse = inv(covTrue)
    covFalseInverse = inv(covFalse)
    convTrueDet = np.linalg.det(covTrue)
    convFalseDet = np.linalg.det(covFalse)
    
    resultTrue = 1/(pow(math.sqrt(2*pi),len(value))*math.sqrt(convTrueDet))
    resultTrue *= math.exp(-1/2*np.dot(np.dot(value-meanArrTrue,covTrueInverse),(value-meanArrTrue).transpose()))

    resultFalse = 1/(pow(math.sqrt(2*pi),len(value))*math.sqrt(convFalseDet))
    resultFalse *= math.exp(-1/2*np.dot(np.dot(value-meanArrFalse,covFalseInverse),(value-meanArrFalse).transpose()))

    if resultTrue > resultFalse:
        return 1
    else:
        return 0

#trnSet read 시작
print("trnSet read")
f = open('test.txt', 'r')

lines = f.readlines()
for line in lines:
	tmp=line.split()
	#trn.txt파일에 형식에 맞지 않는 data들이 있어서 걸러낸다...
	if len(tmp)!=14:
		continue
	result = int(tmp.pop())
	dataList = []
	for i in tmp:
		dataList.append(float(i))
	
	if result == 1:
		trnSetTrue.append(data.data(dataList, result))
	else:
		trnSetFalse.append(data.data(dataList, result))


#trnSet np table로 받기
tmp = []
for data in trnSetTrue:
	tmp.append(data.dataList)

dataInTrue = np.array(tmp)
dataInTrueTranspose = dataInTrue.transpose()
meanInTrue = []
stdInTrue = []

tmp.clear()
"""
for data in trnSetFalse:
	tmp.append(data.dataList)

dataInFalse = np.array(tmp)
dataInFalseTranspose = dataInFalse.transpose()
meanInFalse = []
stdInFalse = []

tmp.clear()
"""

#mean, std 값 얻기
for list in dataInTrueTranspose:
	meanInTrue.append(np.mean(list))
	stdInTrue.append(np.std(list))
"""
for list in dataInFalseTranspose:
	meanInFalse.append(np.mean(list))
	stdInFalse.append(np.std(list))
"""

print("mean, std")



#공분산 계산 시작
print("------------------------------")
print(dataInTrue)
print("------------------------------")
print(np.cov(dataInTrueTranspose))
print("------------------------------")


x = np.array([[4.0, 2.0, 0.60],
              [4.2, 2.1, 0.59],
              [3.9, 2.0, 0.58],
              [4.3, 2.1, 0.62],
              [4.1, 2.2, 0.63]])
y = x.transpose()
print(np.cov(x))
print(y)
for list in y:
    print(list.mean())
    print(list.std())


print(np.cov(y))
print(np.inv(np.cov(y)))

