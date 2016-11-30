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
		"""
		print(meanArrFalse[i])
		print(stdArrFalse[i])
		print(value[i])
		print("중간 계산1 : "+str(1/(math.sqrt(2*pi)*stdArrFalse[i])))
		print("중간 계산2 : "+str(-1*math.pow(meanArrFalse[i]-value[i],2)/(2*math.pow(stdArrFalse[i],2))))
		print("중간 계산3 : "+str(math.exp(-1*math.pow(meanArrFalse[i]-value[i],2)/(2*math.pow(stdArrFalse[i],2)))))
		print(resultFalse)
		"""
		i+=1

	resultTrue *= len(meanArrTrue)/(len(meanArrTrue)+len(meanArrFalse))
	resultTrue *= len(meanArrFalse)/(len(meanArrTrue)+len(meanArrFalse))

	if resultTrue > resultFalse:
		return 1
	else:
		return 0

def multivariateDistribution(meanArrTrue, meanArrFalse, covTrueInverse, covFalseInverse, convTrueDet, convFalseDet, value, k):
   
    
    resultTrue = 1/(pow(math.sqrt(2*math.pi),len(value)/2)*math.sqrt(convTrueDet))
    resultTrue *= math.exp(-1/2*np.dot(np.dot(np.array(value)-np.array(meanArrTrue),covTrueInverse),(np.array(value)-np.array(meanArrTrue)).transpose()))
    resultTrue += k

    resultFalse = 1/(pow(math.sqrt(2*math.pi),len(value)/2)*math.sqrt(convFalseDet))
    resultFalse *= math.exp(-1/2*np.dot(np.dot(np.array(value)-np.array(meanArrFalse),covFalseInverse),(np.array(value)-np.array(meanArrFalse)).transpose()))
#    resultFalse += k

    if resultTrue > resultFalse:
        return 1
    else:
        return 0

#function 선언부 끝

#trnSet read 시작
print("trnSet 읽기 시작")
f = open('trn.txt', 'r')

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


print("trnSet 읽기 완료")

#trnSet np table로 받기
tmp = []
for data in trnSetTrue:
	tmp.append(data.dataList)

dataInTrue = np.array(tmp)
dataInTrueTranspose = dataInTrue.transpose()
meanInTrue = []
stdInTrue = []

tmp.clear()

for data in trnSetFalse:
	tmp.append(data.dataList)

dataInFalse = np.array(tmp)
dataInFalseTranspose = dataInFalse.transpose()
meanInFalse = []
stdInFalse = []

tmp.clear()


#mean, std 값 얻기
for list in dataInTrueTranspose:
	meanInTrue.append(np.mean(list))
	stdInTrue.append(np.std(list))

for list in dataInFalseTranspose:
	meanInFalse.append(np.mean(list))
	stdInFalse.append(np.std(list))

print("mean, std값 추출 완료")




#실제 test할 데이터 읽기 시작
print("test 데이터 읽기 시작")
f = open('tst.txt', 'r')
lines = f.readlines()

"""
print("Guassian Distribution")
for line in lines:
	
	tmp=line.split()

	#tst.txt파일에 형식에 맞지 않는 data들이 있어서 걸러낸다...
	if len(tmp)!=14:
		continue
	result = int(tmp.pop())
	dataList = []
	for i in tmp:
		dataList.append(float(i))
	prediction = gaussianDistribution(meanInTrue, stdInTrue, meanInFalse, stdInFalse, dataList)
	#print("실제값 : "+str(result)+" 예상값 : "+str(prediction))
	if result==0 and prediction==0:
		trueNeg += 1
	elif result==1 and prediction==0:
		falseNeg += 1
	elif result==0 and prediction==1:
		falsePos += 1
	else:
		truePos += 1

print("true positive : "+str(truePos))
print("false negative: "+str(falseNeg))
print("false positive : "+str(falsePos))
print("true negative : "+str(trueNeg))

print("precision : "+str(truePos/(truePos+falsePos)))
print("recall : "+str(truePos/(truePos+falseNeg)))
print("accurancy : "+str((truePos+trueNeg)/(truePos+trueNeg+falsePos+falseNeg)))
"""

print("Multivariable Distribution")

covTrueInverse = np.linalg.inv(np.cov(dataInTrueTranspose))
covFalseInverse = np.linalg.inv( np.cov(dataInFalseTranspose))
convTrueDet = np.linalg.det(np.cov(dataInTrueTranspose))
convFalseDet = np.linalg.det( np.cov(dataInFalseTranspose))


for line in lines:	
	tmp=line.split()
	#tst.txt파일에 형식에 맞지 않는 data들이 있어서 걸러낸다...
	if len(tmp)!=14:
		continue
	result = int(tmp.pop())
	dataList = []
	for i in tmp:
		dataList.append(float(i))
	prediction =multivariateDistribution(meanInTrue, meanInFalse, covTrueInverse, covFalseInverse, convTrueDet, convFalseDet, dataList, 0)
	#print("실제값 : "+str(result)+" 예상값 : "+str(prediction))
	if result==0 and prediction==0:
		trueNeg += 1
	elif result==1 and prediction==0:
		falseNeg += 1
	elif result==0 and prediction==1:
		falsePos += 1
	else:
		truePos += 1
print("true positive : "+str(truePos))
print("false negative: "+str(falseNeg))
print("false positive : "+str(falsePos))
print("true negative : "+str(trueNeg))

print("precision : "+str(truePos/(truePos+falsePos)))
print("recall : "+str(truePos/(truePos+falseNeg)))
print("accurancy : "+str((truePos+trueNeg)/(truePos+trueNeg+falsePos+falseNeg)))

print("the empirical error : "+str((falsePos+falseNeg)/(truePos+trueNeg+falsePos+falseNeg)))

# roc curve 그리기 위한 data 추출
constant=-0.0000000001
x_row = []
y_row = []
fnr = []
fpr = []
while constant<0.00000001:
    truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0
    for line in lines:	
	    tmp=line.split()
	    #tst.txt파일에 형식에 맞지 않는 data들이 있어서 걸러낸다...
	    if len(tmp)!=14:
		    continue
	    result = int(tmp.pop())
	    dataList = []
	    for i in tmp:
		    dataList.append(float(i))
	    prediction =multivariateDistribution(meanInTrue, meanInFalse, covTrueInverse, covFalseInverse, convTrueDet, convFalseDet, dataList, constant)
	    #print("실제값 : "+str(result)+" 예상값 : "+str(prediction))
	    if result==0 and prediction==0:
		    trueNeg += 1
	    elif result==1 and prediction==0:
		    falseNeg += 1
	    elif result==0 and prediction==1:
		    falsePos += 1
	    else:
		    truePos += 1
    y_row.append(truePos/(truePos+falseNeg))
    x_row.append(1-(trueNeg/(falsePos+trueNeg)))
    fnr.append(falseNeg/(falseNeg+truePos))
    fpr.append(falsePos/(falsePos+trueNeg))
    constant+=0.00000000005
"""
    print("constant : "+str(constant))
    print("true positive : "+str(truePos))
    print("false negative: "+str(falseNeg))
    print("false positive : "+str(falsePos))
    print("true negative : "+str(trueNeg))

    print("precision : "+str(truePos/(truePos+falsePos)))
    print("recall : "+str(truePos/(truePos+falseNeg)))
    print("accurancy : "+str((truePos+trueNeg)/(truePos+trueNeg+falsePos+falseNeg)))
    print("sensitivity : "+str(truePos/(truePos+falseNeg)))
    print("1-specifiicty: "+str(1-(trueNeg/(falsePos+trueNeg))))
    print("------------------------------------------------------")
"""    
    
f = open("result.txt", 'w')
count = 0
while count < len(x_row):
    data = str(x_row[count])+"\t"+str(y_row[count])+"\t"+str(fpr[count])+"\t"+str(fnr[count])+"\n"
    f.write(data)
    count += 1
f.close()

print("finished")
