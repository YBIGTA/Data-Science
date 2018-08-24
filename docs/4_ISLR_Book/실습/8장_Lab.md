R Notebook
================

Lab: Decision Trees
===================

8.3.1 Fitting Classification Trees
----------------------------------

분류 / 회귀 Tree 모델은 tree library을 통해 사용할 수 있다.

``` r
library(tree)
```

Carseats에서 sales가 높은지, 낮은지 분류하는 bianry classifiaction을 시행하기 전에, sales 변수가 수치형이므로 이를 바꿔준다.

``` r
library(ISLR)
attach(Carseats)
High = ifelse(Sales>8,"yes","no")
Carseats = data.frame(Carseats,High)
```

이제 tree() func을 사용하여 분류를 시행해보자. 구문은 lm() func와 비슷하다.

``` r
fit.tree = tree(High~.-Sales,data=Carseats)
summary(fit.tree)
```

    ## 
    ## Classification tree:
    ## tree(formula = High ~ . - Sales, data = Carseats)
    ## Variables actually used in tree construction:
    ## [1] "ShelveLoc"   "Price"       "Income"      "CompPrice"   "Population" 
    ## [6] "Advertising" "Age"         "US"         
    ## Number of terminal nodes:  27 
    ## Residual mean deviance:  0.4575 = 170.7 / 373 
    ## Misclassification error rate: 0.09 = 36 / 400

training error rate가 9%인 것을 확인할 수 있다. 분류 tree에서는 엔트로피와 유사한 식(책 325쪽 참조)으로 training error가 계산된다.

tree 모델의 가장 좋은 장점 중 하나는 그래프로 나타낼 수 있다는 것이다. pretty=0은 범주형 자료에 대해 각 범주를 포함하라는 옵션이다.

``` r
plot(fit.tree)
text(fit.tree,pretty=0)
```

![](8장_Lab_files/figure-markdown_github/unnamed-chunk-4-1.png)

``` r
 #fit.tree
#을 입력하면 나무의 가지에 해당하는 결과(분류)를 보여준다.
```

이제 training data와 test data로 나눠서 tree 모델을 적용해보자.

``` r
set.seed(2)
train=sample(1:nrow(Carseats),200)
Carseats.test = Carseats[-train,]
High.test = High[-train]
fit.tree = tree(High~.-Sales , data=Carseats, subset=train)
tree.pred = predict(fit.tree,newdata=Carseats.test, type="class")
table(tree.pred, High.test)
```

    ##          High.test
    ## tree.pred no yes
    ##       no  86  27
    ##       yes 30  57

``` r
(88+56)/200
```

    ## [1] 0.72

이제 tree model을 prune하는 것이 향상된 결과를 가져오는지 확인한다. cv.tree() func은 최적의 tree 복잡도를 결정하기 위해 CV를 실시한다. FUN=prune.misclass는 classification error를 원할 때 쓴다.

``` r
set.seed(3)
cv.carseats = cv.tree(fit.tree,FUN=prune.misclass)
names(cv.carseats)
```

    ## [1] "size"   "dev"    "k"      "method"

``` r
cv.carseats
```

    ## $size
    ## [1] 19 17 14 13  9  7  3  2  1
    ## 
    ## $dev
    ## [1] 55 55 53 52 50 56 69 65 80
    ## 
    ## $k
    ## [1]       -Inf  0.0000000  0.6666667  1.0000000  1.7500000  2.0000000
    ## [7]  4.2500000  5.0000000 23.0000000
    ## 
    ## $method
    ## [1] "misclass"
    ## 
    ## attr(,"class")
    ## [1] "prune"         "tree.sequence"

여기서 size는 terminal nodes의 수(size), 그에 따른 cv-error(dev), cost-complexity parameter의 값(여기서는 k인데 앞서서는 알파)을 출력한다. 결과를 살펴보면 9개의 terminal nodes가 있을 때, cv-error가 가장 낮음을 알 수 있다.

``` r
par(mfrow=c(1,2))
plot(cv.carseats$size,cv.carseats$dev,type="b")
plot(cv.carseats$k,cv.carseats$dev,type="b")
```

![](8장_Lab_files/figure-markdown_github/unnamed-chunk-8-1.png)

이제 어디까지 prune을 해야하는지 cv를 통해서 알아봤으므로 직접 prune.misclass을 통해 prune을 해보자.

``` r
prune.carseats = prune.misclass(fit.tree,best=9)
plot(prune.carseats)
text(prune.carseats, pretty=0)
```

![](8장_Lab_files/figure-markdown_github/unnamed-chunk-9-1.png)

prune이 끝난 tree model으로 test data에 대해서 예측을 해보자.

``` r
prune.pred = predict(prune.carseats,newdata=Carseats.test,type="class")
table(prune.pred,High.test)
```

    ##           High.test
    ## prune.pred no yes
    ##        no  94  24
    ##        yes 22  60

``` r
(94+60)/200
```

    ## [1] 0.77

prune이 된 tree model이 해석하기도 더 편할 뿐만 아니라 정확도도 더 올라갔다!

8.3.2 Fitting Regression Trees
------------------------------

Boston data에 대해서 regression tree을 적합시켜보자. 먼저 training data와 test data로 나눈다.

``` r
library(MASS)
attach(Boston)
set.seed(1)
train = sample(1:nrow(Boston),nrow(Boston)/2)
Boston.test = Boston[-train,]
medv.test = medv[-train]
fit.tree = tree(medv~.,data=Boston,subset=train)
summary(fit.tree)
```

    ## 
    ## Regression tree:
    ## tree(formula = medv ~ ., data = Boston, subset = train)
    ## Variables actually used in tree construction:
    ## [1] "lstat" "rm"    "dis"  
    ## Number of terminal nodes:  8 
    ## Residual mean deviance:  12.65 = 3099 / 245 
    ## Distribution of residuals:
    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ## -14.10000  -2.04200  -0.05357   0.00000   1.96000  12.60000

regression tree에서는 deviance가 tree에 대한 sum of squared errors라고 보면 된다.

``` r
plot(fit.tree)
text(fit.tree,pretty=0)
```

![](8장_Lab_files/figure-markdown_github/unnamed-chunk-12-1.png)

이제 prune을 통해 결과를 더 향상시킬 수 있을지 살펴보자.

``` r
cv.boston = cv.tree(fit.tree)
cv.boston
```

    ## $size
    ## [1] 8 7 6 5 4 3 2 1
    ## 
    ## $dev
    ## [1]  5226.322  5228.360  6462.626  6692.615  6397.438  7529.846 11958.691
    ## [8] 21118.139
    ## 
    ## $k
    ## [1]      -Inf  255.6581  451.9272  768.5087  818.8885 1559.1264 4276.5803
    ## [8] 9665.3582
    ## 
    ## $method
    ## [1] "deviance"
    ## 
    ## attr(,"class")
    ## [1] "prune"         "tree.sequence"

``` r
plot(cv.boston$size,cv.boston$dev,type="b")
```

![](8장_Lab_files/figure-markdown_github/unnamed-chunk-14-1.png)

prune을 해보자

``` r
prune.boston = prune.tree(fit.tree,best=7)
plot(prune.boston)
text(prune.boston,pretty=0)
```

![](8장_Lab_files/figure-markdown_github/unnamed-chunk-15-1.png)

prune을 할 때, classification tree에서는 prune.misclass()이었지만 regression tree에서는 prune.tree()을 사용한다.

prune을 거친 tree model을 통해 test mse을 구해보자.

``` r
prune.pred = predict(prune.boston,newdata=Boston.test)
mean((medv.test-prune.pred)^2)
```

    ## [1] 25.72341

prune을 하기 전, tree model을 통해 test mse을 구해보자.

``` r
unprune.pred = predict(fit.tree,newdata=Boston.test)
mean((medv.test-unprune.pred)^2)
```

    ## [1] 25.04559

prune을 한 tree model이 test mse가 살짝 더 높은 것을 알 있다.

8.3.3 Bagging and Random Forests
--------------------------------

bagging과 random forests는 randomForest package을 이용하여 실행할 수 있다.

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

bagging은 randomforest의 특별한 경우다. 즉, bagging은 m=k인 경우다.따라서 randomForest()로 bagging, randomforest 모두를 시행할 수 있다.

먼저 bagging을 해보자.

``` r
set.seed(1)
fit.bag = randomForest(medv~.,data=Boston,subset=train,mytry=ncol(Boston)-1,importance=TRUE)
fit.bag
```

    ## 
    ## Call:
    ##  randomForest(formula = medv ~ ., data = Boston, mytry = ncol(Boston) -      1, importance = TRUE, subset = train) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 4
    ## 
    ##           Mean of squared residuals: 12.48435
    ##                     % Var explained: 84.88

``` r
names(fit.bag)
```

    ##  [1] "call"            "type"            "predicted"      
    ##  [4] "mse"             "rsq"             "oob.times"      
    ##  [7] "importance"      "importanceSD"    "localImportance"
    ## [10] "proximity"       "ntree"           "mtry"           
    ## [13] "forest"          "coefs"           "y"              
    ## [16] "test"            "inbag"           "terms"

``` r
fit.bag$importance # 중요도 살펴보기.
```

    ##            %IncMSE IncNodePurity
    ## crim     7.2905442    1247.87271
    ## zn       0.2570270      84.97987
    ## indus    5.3449029    1308.70340
    ## chas     0.2279444      98.53459
    ## nox      6.6837255    1168.44941
    ## rm      30.6389694    5636.39110
    ## age      3.7418214     660.34542
    ## dis      6.0070861    1453.81733
    ## rad      1.5398313     161.38834
    ## tax      3.6917809     736.79049
    ## ptratio  5.1991608    1191.12896
    ## black    1.6716001     418.42770
    ## lstat   60.8139498    6179.11950

이 bagging model에 대한 test mse를 구해보자.

``` r
bag.pred = predict(fit.bag,newdata=Boston.test)
mean((medv.test-bag.pred)^2)
```

    ## [1] 11.6076

decision tree의 test mse보다 훨씬 줄어들었다. 또한 tree의 갯수를 조절할 수도 있다.

``` r
fit.bag = randomForest(medv~. , data=Boston, subset=train, mytry=ncol(Boston)-1, ntree=25)
bag.pred = predict(fit.bag, newdata=Boston.test)
mean((medv.test - bag.pred)^2)
```

    ## [1] 11.65373

randomforest을 시행하는 것은 mytry을 바꾸는 것 이외에 모두 동일하다. 기본값으로, randomForest() func은 분류 문제에 대해서 root(p)개의 변수를, regression 문제에 대해서는 p/3개의 변수를 사용한다.

``` r
set.seed(1)
fit.rf = randomForest(medv~. , data=Boston, subset=train, mytry=6, importance=TRUE)
rf.pred = predict(fit.rf, newdata=Boston.test)
mean((medv.test-rf.pred)^2)
```

    ## [1] 11.6076

각 변수중 중요도를 살펴보자.

``` r
importance(fit.rf)
```

    ##           %IncMSE IncNodePurity
    ## crim    12.712371    1247.87271
    ## zn       3.340046      84.97987
    ## indus   10.611878    1308.70340
    ## chas     1.733390      98.53459
    ## nox     13.843947    1168.44941
    ## rm      29.465247    5636.39110
    ## age      6.565646     660.34542
    ## dis     12.830180    1453.81733
    ## rad      4.679142     161.38834
    ## tax      9.195031     736.79049
    ## ptratio 11.521584    1191.12896
    ## black    8.385275     418.42770
    ## lstat   26.313510    6179.11950

왼쪽은 해당 변수가 빠졌을 때, 예측변수의 정확도 하락의 평균이다. 즉 이 수치가 크다면 이 변수가 빠질 시에 정확도가 크게 내려간다는 뜻이고 그만큼 중요하다는 뜻이다. 오른쪽은 그 변수에서 split했을 때 node impurity 총 하락을 측정것 것이다. 결국 얘가 크다는 것은 그만큼 중요하다는 뜻이다. varImpPlot()을 통해서 plot으로 그려보자.

``` r
varImpPlot(fit.rf)
```

![](8장_Lab_files/figure-markdown_github/unnamed-chunk-24-1.png)

rm과 lstat 변수가 가장 중요함을 알 수 있다!

8.3.4 Boosting
--------------

boosting model을 적용하기 위해 gbm package의 gbm() func을 이용한다.

``` r
library(gbm)
```

    ## Loading required package: survival

    ## Loading required package: lattice

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.3

``` r
set.seed(1)
fit.boost = gbm(medv~. , data=Boston[train,], distribution = "gaussian", n.trees=5000, interaction.depth=4) #gbm은 subset을 쓰면 error가 뜸.
names(fit.boost)
```

    ##  [1] "initF"             "fit"               "train.error"      
    ##  [4] "valid.error"       "oobag.improve"     "trees"            
    ##  [7] "c.splits"          "bag.fraction"      "distribution"     
    ## [10] "interaction.depth" "n.minobsinnode"    "num.classes"      
    ## [13] "n.trees"           "nTrain"            "train.fraction"   
    ## [16] "response.name"     "shrinkage"         "var.levels"       
    ## [19] "var.monotone"      "var.names"         "var.type"         
    ## [22] "verbose"           "data"              "Terms"            
    ## [25] "cv.folds"          "call"              "m"

``` r
summary(fit.boost)
```

![](8장_Lab_files/figure-markdown_github/unnamed-chunk-26-1.png)

    ##             var    rel.inf
    ## lstat     lstat 45.9627334
    ## rm           rm 31.2238187
    ## dis         dis  6.8087398
    ## crim       crim  4.0743784
    ## nox         nox  2.5605001
    ## ptratio ptratio  2.2748652
    ## black     black  1.7971159
    ## age         age  1.6488532
    ## tax         tax  1.3595005
    ## indus     indus  1.2705924
    ## chas       chas  0.8014323
    ## rad         rad  0.2026619
    ## zn           zn  0.0148083

rm과 lstat가 가장 중요한 변수임을 알 수 있다.

또한 이 두 개의 변수에 대한 partial dependence plots을 그릴 수도 있다. 이 plot은 다른 변수를 integrate out한 후에, 반응변수에 대한 선택된 변수의 marginal effect을 보여준다.

``` r
par(mfrow=c(1,2))
plot(fit.boost,i="rm")
plot(fit.boost,i="lstat")
```

![](8장_Lab_files/figure-markdown_github/unnamed-chunk-27-1.png)

이제 boost model의 test mse를 구해보자.

``` r
boost.pred = predict(fit.boost, newdata=Boston.test, n.trees=5000)
mean((medv.test-boost.pred)^2)
```

    ## [1] 11.84434

만약에 원한다면, 다른 shrinkage parameter lambda로 boosting을 할 수도 있다. 기본 값은 0.001이며 여기서는 0.2로 해보았다.

``` r
fit.boost = gbm(medv~., data=Boston[train,], distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2)
boost.pred = predict(fit.boost, newdata=Boston.test, n.trees=5000)
mean((medv.test-boost.pred)^2)
```

    ## [1] 11.51109
