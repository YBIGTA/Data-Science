### 2.1.3 <font color = 'Red'>**Trade-Off**</font> Between Prediction Accuracy(예측 정확도) & Model Interpretability(모델 해석력)

앞의 Years of Education과 Seniority를 통해 Income을 예측하는 두가지 예시 모델을 보자.

![Alt Text](https://github.com/YBIGTA/Data-Science/tree/master/docs/8_ISLR_NEW/image/1.PNG)

​                                                                                 *(그림 2.1)* 

​                                        ![Alt Text](https://github.com/YBIGTA/Data-Science/tree/master/docs/8_ISLR_NEW/image/2.PNG) 

​                                                                                 *(그림 2.2)*

(그림 2.2)의 Smooth thin-plate spline과 같은 방법들은 (그림 2.1)의 선형적 모델보다 훨씬 유연(flexible)하고, 보통의 경우 더 오차가 적은 예측을 보여준다.  

그럼 다음과 같은 의문이 생긴다. ***도대체 왜 매우 유연한 기법 대신에 더 제한적인(restrictive) 방법을 사용하는가?***

---

#### 모델의 유연성과 해석력에는 Trade - Off 관계가 있다!

![Alt Text](https://github.com/YBIGTA/Data-Science/tree/master/docs/8_ISLR_NEW/image/1.PNG)

​                    (그림 2.3) 몇 가지 method들에 대해 flexibility와 interpretability의 **Trade-Off** 관계

**우리가 *f*를 추론하고자 하는 이유 : 예측(Prediction) vs 추론(Inference)**

> **추론이 목적인 경우**

- **해석이 중요**
- 따라서 *f*의 추정이 복잡하여 각 X와 Y의 상관관계를 이해하기 어려운 유연한 모델보다,
- X와 Y의 상관관계를 매우 직관적으로 해석할 수 있는 제한적 모델이 better!(e.g. 선형모델)

>**예측이 목적인 경우**

- e.g. 주가 예측 알고리즘
- 적용 가능한 가장 유연한 모델이 최선?
- Not Always the case! (Overfitting)



### 2.1.4 Supervised VS Unsupervised Learning

> **Supervised Learning(지도학습)**

- 설명변수 *x*<sub>*i*</sub> (*i* = 1, ... , n) 에 대해 연관된 반응변수 *y*<sub>*i*</sub>가 존재
- 목적 : 예측(Prediction) 또는 추론(Inference)
- Linear Regression, Logistic Regression, GAM, Boosting, Support vector machines

> **Unsupervised Learning(비지도학습)**

- 설명변수 *x*<sub>*i*</sub> (*i* = 1, ... , n) 에 대해 연관된 반응변수 *y*<sub>*i*</sub>가 존재하지 않음
- 목적 :  그룹을 식별하는 것
- Clustering Analysis

> **Semi-Supervised Learning(준지도학습)**

- *n*개 관측치의 집합이 있을때
- *m*(<*n*)개의 관측치는 설명변수에 대한 반응변수가 존재
- *m*-*n*개의 관측치는 설명변수에 대해 반응변수가 존재하지 않는 경우
- ISLR에서는 다루지 않음.

### 2.1.5 Regression VS Classification Problems

> **Regression Problem**

- 양적(*quantitative*) 반응변수(*y*)를 가지는 문제
- 양적 반응 변수란, 수치 값을 취하는 것 
- e.g. 사람의 나이, 키, 수입, 집 값, 주식 가격 등

> **Classification Problem**

- 질적(*qualitative, categorical*) 반응변수를 가지는 문제
- 질적 반응변수란, K개의 다른 클래스(혹은 카테고리) 중의 하나를 값으로 가지는 것.
- e.g. 사람의 성별(남 or 여), 성적(A,B,C,D or F), 채무 지불 여부(연체 or 완납) 



​       **이 구분이 항상 명확한 것은 아님**.

e.g. ) 

**Least sqares linear regression**(최소제곱선형회귀) : 양적 반응변수와 사용
**Logistic Regression**(로지스틱 회귀) : **질적 반응변수와 사용** but 클래스 확률(양적)을 추정하므로 회귀로 생각될 수도 있음





## 2.2 Assessing Model Accuracy(모델의 정확도 평가)

***왜 하나의 최고의 방법 대신 많은 통계학습 기법을 소개하는가?***

- 통계에서 모든 자료에 대해 어떤 한 방법이 다른 방법들보다 지배적으로 나은 경우 X
- 임의의 주어지 자료에 대해 최고의 기법을 선택하는 것이 중요



### 2.2.1 Measuring the Quality of Fit



### 2.2.2 The Bias-Variance <font color = 'Red'>***Trade-Off***</font>



### 2.2.3 The Classification Setting

