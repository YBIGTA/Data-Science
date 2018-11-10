## What is face recognition?
- Verification (1:1)  
 - input: 사람 이미지, 이름  
 - outpit: 입력 이미지가 주어진 이름의 사람이 맞는가?(yes or no)  
- Recognition (1:k)  
 - k명의 사람들의 데이터베이스   
 - input: 사람 이미지   
 - output: 입력 이미지가 k명의 사람 중 누구?(특정 이름)  
 
## one shot learning
- 한 사람 당 하나의 사진을 input으로 받아서 학습
  - DB내에 있는 사람의 다른 각도에서 촬영한 사진을 "같은 사람"이라고 인식하게 만들 수 있도록
  - DB내에 없는 사람이라면 없는 사람임을 인지할 수 있도록
- output layer의 유닛 갯수는 DB내 사람의 수 + 1

- similarity function  
![](course4_week4/2.jpg)  

## siamese Network
![](course4_week4/6.jpg)   
신경망의 parameter들은 인코딩 f(x^{[i]})를 결정
- x^{[i]}와 x^{[j]}가 같은 사람이면 ||f(x^{[i]}) - f(x^{[j]})||_{2}^{2}$$ 가 작도록   

- x^{[i]}와 x^{[j]}가 다른 사람이면 ||f(x^{[i]}) - f(x^{[j]})||_{2}^{2}$$ 가 크도록   


## Triplet loss
- 가장 먼저 해야 하는 일은 input image를 인코딩하는 작업  
![](course4_week4/3.jpg)    
  - 일련의 과정을 통해서 최종적으로 feature vector를 다음과 같이 산출함

- 다음으로 유사도 함수 d를 정의
  - ex) $$d(img1, img2) = ||f(x^{[1]}) - f(x^{[2]})||_{2}^{2}$$  
  
- 훈련 세트의 구성
  - 훈련을 위해서 임의적으로 사진을 선택할 경우 loss function의 목적 달성이 비교적 쉬워짐
  - 더 어려운 훈련 세트 구성 필요 : d(A,P)와 d(A, N)이 근접한 사진을 선택
  - 학습 알고리즘의 computational efficiency 가 높아짐
- cost function의 정의  


## Face Verification and Binary Classification

- triplet loss가 아니라 일반적인 loss function을 이용하는 방법:  
![](course4_week4/4.jpg)     
 - 얼굴 인식을 이진 분류 문제로 확장  
 - 두 개의 siamease network의 쌍으로 임베딩들을 계산 후 로지스틱 회귀 유닛에 입력해서 동일 유무를 예측  
 - 같은 사람일 경우 1, 다른 사람일 경우 0  
 - $/hat{y} = /sigma{(/sum^{128}_{[k=1]} = w_{[i]}|f(x^{[i]})_{k} - f(x^{[j]})_{k} + b )$    
 - 카이제곱 식을 사용하기도 함.   
 - 데이터베이스에서는 이미지를 저장하지 않고 미리 인코딩을 계산해 놓은 값을 활용.    
![](course4_week4/5.jpg)   
