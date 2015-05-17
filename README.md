Usage : 
cat Holmes_Training_Data/*.TXT | ./preprocessing.sh > training.txt

ref : 
https://code.google.com/p/pygoogle

Data preprocessing
./preprocessing <training data 路徑> <training data filename or -a（路徑下所有檔案）>
會產生一個train_pro.txt，可以給word2vec使用


###Issue need to solve :
1. 目前 outputLayer是以softMax做nonlinearity，但今天她代表的是機率，跟我們從word2vec得到的R^m vector是不同的東西
   (看docs上說word2vec是以hierarchical softMax實作，但今天他有正有負(log(prob)??)) 
   所以...是要照目前的方式，然後再加深層數之類的做deep RNN，還是我們要將所有的word做classification(dim炸裂囧)
   不過...可能是需要做regularization

2. 3D matrix manipulication??  -> tensordot  (solved?)
   張量內積那邊numpy的docs寫的還蠻清楚的(也沒有啦 例子很懂懂XDD)

3. nesterov grad不能用吧XD 目前是用rmsprop做為update method  (做實驗的好材料？) 可以試試其他方式

4. Regularization? 不要讓output炸開而是能壓在一定的range以下?(lasagne.regularization有包好~~但我覺得好像也應該是要跟word2vec做match?)

5. X\_train ,Y\_train 的給法
   e.g. Given wordSeq {w\_1,w\_2,....w\_n}
   Assume 5-gram : X\_train : {w\_1,w\_2,...w\_5}
  		   Y\_train : {w\_2,w\_3,...w\_6}   , another method ? 
