# Course 4. Convolutional Neural Networks
## Week 4.1. Neural Style Transfer
<br/>  <br/>  

### 4.1.1. What is face recognition?
- Verification (1:1)  
 - input: 사람 이미지, 이름  
 - outpit: 입력 이미지가 주어진 이름의 사람이 맞는가?(yes or no)  
- Recognition (1:k)  
 - k명의 사람들의 데이터베이스   
 - input: 사람 이미지   
 - output: 입력 이미지가 k명의 사람 중 누구?(특정 이름)  
 
 <br/><br/>
 
### 4.1.2. one shot learning <br/>
- 한 사람 당 하나의 사진을 input으로 받아서 학습
  - DB내에 있는 사람의 다른 각도에서 촬영한 사진을 "같은 사람"이라고 인식하게 만들 수 있도록
  - DB내에 없는 사람이라면 없는 사람임을 인지할 수 있도록
- output layer의 유닛 갯수는 DB내 사람의 수 + 1
<br/>
- similarity function  
![](images/2.jpg)  
<br/><br/>

- 다음으로 유사도 함수 d를 정의
  - ex) $d(img1, img2) = ||f(x^{[1]}) - f(x^{[2]})||_{2}^{2}$  

### 4.1.2. siamese Network <br/>
![](images/6.jpg)   
<br/>
신경망의 parameter들은 인코딩 f(x^{[i]})를 결정
- $x^{[i]}$와 $x^{[j]}$가 같은 사람이면 $||f(x^{[i]}) - f(x^{[j]})||_{2}^{2}$ 가 작도록   

- $x^{[i]}$와 $x^{[j]}$가 다른 사람이면 $||f(x^{[i]}) - f(x^{[j]})||_{2}^{2}$ 가 크도록   

<br/>
### 4.1.3. Triplet Loss <br/>

- Face Recognition 모델의 핵심 요소인 Similarity function과 Siamese Network에 대해서 학습하였다.
- 그러면 Face Recognition 모델의 Loss Function인 Triplet Loss에 대하여 알아보자. <br/><br/>
----
![APN](images/APN.JPG)
<br/><br/>
- 기준이 되는 사진을 Anchor(A), Anchor와 같은 사람의 다른 이미지를 Positive(P), 다른 사람의 이미지를 Negative(N)

- 원하는 것은 A와 P의 similarity는 높이고(거리 감소) A와 N의 similarity는 낮추는(거리 증가) 것. 두 similarity의 차이의 수준은 $a$(margin)
<br/>
----
<br/>
![Triplet Loss](images/Triplet Loss.JPG)
<br/>
- Loss Function은 3개의 image 쌍인 Triplet(A,P,N)에 대해서 위와 같이 정의
- Max를 하는 이유는 좌항이 0보다 작은 경우 $a$수준의 margin을 달성한 것이므로, 해당 loss에 대해서는 더 이상 계산하지 않는 다는 의미. 즉 모든 Triplet에 대해서 $a$이상의 수준을 달성하지만 $a$만 넘는다면 상관하지 않음.
- Loss에 시그마를 씌우면 Cost Function

<br/>
- 마지막으로 모델을 학습할 때 모든 Triplet을 사용하는 것은 너무 오래 걸리기 때문에 특정 Triplet들을 위주로 학습 시키는 것이 효율적. 여기서 특정 Triple이란 학습시키기 어려운 Triplet로서, (1) 같은 사람인데 거리가 먼 케이스 (2) 다른 사람인데 거리가 가까운 케이스, 두 가지로 구분 됨. (1)을 hard-positive, (2)를 hard-negative라 부름. 효율적인 학습을 위해 트레이닝 셋에서 hard-positive와 hard-negative를 추출하여 모델을 학습.


<br/>
### 4.1.4. Face Verification and Binary Classification
<br/>
- triplet loss가 아니라 일반적인 loss function을 이용하는 방법:  
<br/>
![](images/4.jpg)      <br/>
 - 얼굴 인식을 이진 분류 문제로 확장  
 - 두 개의 siamease network의 쌍으로 임베딩들을 계산 후 로지스틱 회귀 유닛에 입력해서 동일 유무를 예측  
 - 같은 사람일 경우 1, 다른 사람일 경우 0  
 - $ \hat{y} = \sigma{(\sum^{128}_{[k=1]} = w_{[i]}|f(x^{[i]})_{k} - f(x^{[j]})_{k} + b )} $    
 - 카이제곱 식을 사용하기도 함.   
 - 데이터베이스에서는 이미지를 저장하지 않고 미리 인코딩을 계산해 놓은 값을 활용.    
![](images/5.jpg) <br/>





