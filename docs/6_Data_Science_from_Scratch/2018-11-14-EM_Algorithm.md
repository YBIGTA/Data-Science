## Expectation Maximization

출처: https://www.nature.com/articles/nbt1406.pdf

번역글 참고 : https://csstudy.wordpress.com/2013/07/03/210/

보충 수식 설명(한글): http://sanghyukchun.github.io/70/  //  http://norman3.github.io/prml/docs/chapter09/4.html



원래 공부하려던건 clustering 알고리즘이었으나 하단에 이미 잘 설명되어있어서, 

https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68

그 중 네번째 GMM에서 나오는 EM algorithm에 대해 알아보기로 했다.



수식으로 이해하긴 좀 어려워서 nature지에 쉽게 설명된 예제를 통해 설명해보겠다.



## EM algorithm 이란?

 모델이 잠재변수(latent variable)에 영향을 받는 경우(incomplete data), MLE 혹은 MAP를 구할 수 있게 해주는 것이다. 즉, hideen Markov model이나 Bayesian network와 같은 확률 모델에서 파라미터 추정을 가능하게 해준다.

예제를 보며 이해해보자.

**1) 잠재변수란? **

지겨울 수도 있겠지만 동전 뒤집기로 생각해보자.

 $\theta_A$ 가 동전 A가 앞면이 나올 확률, $\theta_B$ 는 동전 B가 앞면이 나올 확률이라고 하자. 동전 앞면 개수 맞추기로 내기를 했다면 이 $\theta$ 가 무엇인지 알고 싶겠지만, 주어지지 않은 경우에는 이 모수들을 추정해야 한다. 가장 쉬운 방법은, 

![image](https://user-images.githubusercontent.com/32008883/48476085-71872280-e841-11e8-8f3f-50f9706ee3ef.png)

상단과 같이 동전 A,B 던지기 결과가 주어졌다면, 

$\hat{\theta_A} $ = # of heads using coin A/ total # of flips using coin A ,

$\hat{\theta_B} $ = # of heads using coin B/ total # of flips using coin B

로 모수를 추정할 수 있다. 이게 바로 MLE 다. 



그런데 만약? 동전 던지기의 결과가 동전 A인지 동전 B인지 알 수 없다면?

동전의 종류가 잠재변수(latent variable)이 된다. 그러면 더 모수를 추정하기 어려워진다.

이를 해결하고자 EM algorithm의 아이디어가 나온다. 







**2) EM algorithm의 아이디어 **

![image](https://user-images.githubusercontent.com/32008883/48524564-18140780-e8c4-11e8-9c0f-857032d1f680.png)







i. 초기 모수를 설정한다. $\hat{\theta^{(t)} } =( \hat{\theta_A^{(t)} }, \hat{\theta_B^{(t)} } ) $  

$ \hat{\theta^{(0)} }= (0.6, 0.5) $



ii. 앞에서 설정한 모수를 바탕으로 P(결과|동전A), P(결과|동전B)를 구한다. 

 HTTTHHTHTH 결과가 주어졌다면,  P(HTTTHHTHTH|A) 과 P(HTTTHHTHTH|B) 를 구한다.

베이즈 정리에 의하여, 

P(HTTTHHTHTH|A) $\propto$ P(H|A) * P(T|A) * ... * P(H|A) * P(A) = 0.6 * 0.4 * ….. * 0.4 * 0.5 = 0.0003981 

P(HTTTHHTHTH|B) $\propto$ P(H|B) * P(T|B) * …. * P(T|B) * P(B) = 0.5 * 0.5*… * 0.5 * 0.5 =  0.00048825  

normalize하면 

P(HTTTHHTHTH|A) = 0.45,  P(HTTTHHTHTH|B)=0.55이다.



iii. (E-Step)현재의 모수  $\hat{\theta^{(t)} }$로  확률 분포를 추정한다.

즉, 앞에서 구한 P(결과|동전 A), P(동전 B| 결과)를 바탕으로 각 동전에 따른 Head와 Tail 개수의  expectation을 구한다.

H: 앞면 나온 횟수, Y: 동전의 종류 라고 확률 변수를 정하면,

E(H=5|Y=A) = 5*P(H=5|Y=A) = 5 * 0.45 = 2.2

같은 방법으로 뒷면에 대해서도 expectation을 구할 수 있다.



iv. (M-Step) E-step에서의 expectation table을 바탕으로 다시 MLE을 구한다. (새로운 $\hat{\theta^{(t+1)} }$ )

Expecation을 구한 표에서 각 동전의 경우 Head가 나온 갯수, Tail이 나온 갯수를 다 더한 후에 MLE을 계산한다,

$\hat{\theta_A^{1} } = 21.3 / (21.3 + 8.6)$ 임을 알 수 있다.



v. 수렴할 때까지 앞의 과정을 반복한다.





**3) EM alogrithm 의 의의 **

incomplete data데이터가 불충분한 경우(hidden variable이 존재하는 경우) 파라미터 추정을 할 때 간단하고 robust하다. gradient descent와 Newton-Raphson도 비슷한 역할을 하지만 EM algorithm이 구현하기 쉬우며 robust하다.

실제 연구에 쓰이는 예시로, 만약 유전자 데이터가 있다면 그 유전자들의 cluster를 설정하여 분석하고 싶은 경우가 있다. (각 유전자들은 해당 클러스터의 Gaussian dist를 따른다고 가정) 이 경우 우리가 추정해야하는 모수는 Gaussian dist의 mean과 covariance matrice가 되며, hidden variable은 클러스터 종류?이다. 즉, E-step과 M-step을 거치며 유전자들을 cluster에 배치하게 된다.
