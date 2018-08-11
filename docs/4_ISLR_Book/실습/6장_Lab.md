R Notebook
================

6.5.1 Best Subset Selection
---------------------------

Hitters data로 best subset selection 진행. Salary 칼럼을 다른 변수들로 예측하는 상황. 우선 Salary 칼럼에 결측치가 있으므로 해당 행을 지우자.

``` r
library(ISLR)
names(Hitters)
```

    ##  [1] "AtBat"     "Hits"      "HmRun"     "Runs"      "RBI"      
    ##  [6] "Walks"     "Years"     "CAtBat"    "CHits"     "CHmRun"   
    ## [11] "CRuns"     "CRBI"      "CWalks"    "League"    "Division" 
    ## [16] "PutOuts"   "Assists"   "Errors"    "Salary"    "NewLeague"

``` r
dim(Hitters)
```

    ## [1] 322  20

``` r
sum(is.na(Hitters$Salary))
```

    ## [1] 59

``` r
hitters = na.omit(Hitters)
dim(hitters)
```

    ## [1] 263  20

leaps library에 있는 regsubsets() function이 best subset selection을 시행해준다. 문법은 lm과 동일하고 summary()는 각 모델에 대한 best set of variables을 출력해준다.

``` r
library(leaps)
mod.best = regsubsets(Salary~. , hitters)
summary(mod.best)
```

    ## Subset selection object
    ## Call: regsubsets.formula(Salary ~ ., hitters)
    ## 19 Variables  (and intercept)
    ##            Forced in Forced out
    ## AtBat          FALSE      FALSE
    ## Hits           FALSE      FALSE
    ## HmRun          FALSE      FALSE
    ## Runs           FALSE      FALSE
    ## RBI            FALSE      FALSE
    ## Walks          FALSE      FALSE
    ## Years          FALSE      FALSE
    ## CAtBat         FALSE      FALSE
    ## CHits          FALSE      FALSE
    ## CHmRun         FALSE      FALSE
    ## CRuns          FALSE      FALSE
    ## CRBI           FALSE      FALSE
    ## CWalks         FALSE      FALSE
    ## LeagueN        FALSE      FALSE
    ## DivisionW      FALSE      FALSE
    ## PutOuts        FALSE      FALSE
    ## Assists        FALSE      FALSE
    ## Errors         FALSE      FALSE
    ## NewLeagueN     FALSE      FALSE
    ## 1 subsets of each size up to 8
    ## Selection Algorithm: exhaustive
    ##          AtBat Hits HmRun Runs RBI Walks Years CAtBat CHits CHmRun CRuns
    ## 1  ( 1 ) " "   " "  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 2  ( 1 ) " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 3  ( 1 ) " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 4  ( 1 ) " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 5  ( 1 ) "*"   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 6  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "  
    ## 7  ( 1 ) " "   "*"  " "   " "  " " "*"   " "   "*"    "*"   "*"    " "  
    ## 8  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   " "    " "   "*"    "*"  
    ##          CRBI CWalks LeagueN DivisionW PutOuts Assists Errors NewLeagueN
    ## 1  ( 1 ) "*"  " "    " "     " "       " "     " "     " "    " "       
    ## 2  ( 1 ) "*"  " "    " "     " "       " "     " "     " "    " "       
    ## 3  ( 1 ) "*"  " "    " "     " "       "*"     " "     " "    " "       
    ## 4  ( 1 ) "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 5  ( 1 ) "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 6  ( 1 ) "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 7  ( 1 ) " "  " "    " "     "*"       "*"     " "     " "    " "       
    ## 8  ( 1 ) " "  "*"    " "     "*"       "*"     " "     " "    " "

\*은 그 변수가 모델에 포함되었다는 뜻이다. 예를 들어 1개의 변수를 사용하는 애는 Hits와 CRBI를 포함한다는 뜻이다. regsubsets()은 기본값으로 변수가 8개까지 포함된 best subset model을 출력해주는데 바꾸고 싶으면 nvmax 을 조정하면 된다.

``` r
mod.best = regsubsets(Salary~. ,data=hitters,nvmax=19)
summary(mod.best)
```

    ## Subset selection object
    ## Call: regsubsets.formula(Salary ~ ., data = hitters, nvmax = 19)
    ## 19 Variables  (and intercept)
    ##            Forced in Forced out
    ## AtBat          FALSE      FALSE
    ## Hits           FALSE      FALSE
    ## HmRun          FALSE      FALSE
    ## Runs           FALSE      FALSE
    ## RBI            FALSE      FALSE
    ## Walks          FALSE      FALSE
    ## Years          FALSE      FALSE
    ## CAtBat         FALSE      FALSE
    ## CHits          FALSE      FALSE
    ## CHmRun         FALSE      FALSE
    ## CRuns          FALSE      FALSE
    ## CRBI           FALSE      FALSE
    ## CWalks         FALSE      FALSE
    ## LeagueN        FALSE      FALSE
    ## DivisionW      FALSE      FALSE
    ## PutOuts        FALSE      FALSE
    ## Assists        FALSE      FALSE
    ## Errors         FALSE      FALSE
    ## NewLeagueN     FALSE      FALSE
    ## 1 subsets of each size up to 19
    ## Selection Algorithm: exhaustive
    ##           AtBat Hits HmRun Runs RBI Walks Years CAtBat CHits CHmRun CRuns
    ## 1  ( 1 )  " "   " "  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 2  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 3  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 4  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 5  ( 1 )  "*"   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 6  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "  
    ## 7  ( 1 )  " "   "*"  " "   " "  " " "*"   " "   "*"    "*"   "*"    " "  
    ## 8  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   "*"    "*"  
    ## 9  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"  
    ## 10  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"  
    ## 11  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"  
    ## 12  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"  
    ## 13  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"  
    ## 14  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    " "   " "    "*"  
    ## 15  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    "*"   " "    "*"  
    ## 16  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"  
    ## 17  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"  
    ## 18  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   " "    "*"  
    ## 19  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   "*"    "*"  
    ##           CRBI CWalks LeagueN DivisionW PutOuts Assists Errors NewLeagueN
    ## 1  ( 1 )  "*"  " "    " "     " "       " "     " "     " "    " "       
    ## 2  ( 1 )  "*"  " "    " "     " "       " "     " "     " "    " "       
    ## 3  ( 1 )  "*"  " "    " "     " "       "*"     " "     " "    " "       
    ## 4  ( 1 )  "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 5  ( 1 )  "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 6  ( 1 )  "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 7  ( 1 )  " "  " "    " "     "*"       "*"     " "     " "    " "       
    ## 8  ( 1 )  " "  "*"    " "     "*"       "*"     " "     " "    " "       
    ## 9  ( 1 )  "*"  "*"    " "     "*"       "*"     " "     " "    " "       
    ## 10  ( 1 ) "*"  "*"    " "     "*"       "*"     "*"     " "    " "       
    ## 11  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     " "    " "       
    ## 12  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     " "    " "       
    ## 13  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 14  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 15  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 16  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 17  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    "*"       
    ## 18  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    "*"       
    ## 19  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    "*"

``` r
names(summary(mod.best))
```

    ## [1] "which"  "rsq"    "rss"    "adjr2"  "cp"     "bic"    "outmat" "obj"

``` r
summary(mod.best)$adjr2
```

    ##  [1] 0.3188503 0.4208024 0.4450753 0.4672734 0.4808971 0.4972001 0.5007849
    ##  [8] 0.5137083 0.5180572 0.5222606 0.5225706 0.5217245 0.5206736 0.5195431
    ## [15] 0.5178661 0.5162219 0.5144464 0.5126097 0.5106270

``` r
summary(mod.best)$rsq
```

    ##  [1] 0.3214501 0.4252237 0.4514294 0.4754067 0.4908036 0.5087146 0.5141227
    ##  [8] 0.5285569 0.5346124 0.5404950 0.5426153 0.5436302 0.5444570 0.5452164
    ## [15] 0.5454692 0.5457656 0.5459518 0.5460945 0.5461159

RSS, adj R^2, Cp, BIC를 그래프로 그려보는 것은 어떤 모델을 선택할지 도움을 준다.

``` r
library(ggplot2)
sum.mat = data.frame('rsq' = summary(mod.best)$rsq,'adjr2'=summary(mod.best)$adjr2,'bic'=summary(mod.best)$bic,'num.var' = 1:19)
#var 갯수에 따른 결정계수 그래프
ggplot(sum.mat,aes(x=num.var,y=rsq)) + geom_line(color='red')+ labs(title='rsq and ajd rsq with best subset selection',caption='red: rsq, blue: adj rsq') + geom_line(color='blue',aes(x=num.var,y=adjr2)) + geom_vline(xintercept =sum.mat$num.var[which.max(sum.mat$adjr2)],color='green')
```

![](6장_Lab_files/figure-markdown_github/unnamed-chunk-4-1.png)

``` r
#var 갯수에 따른 adj bic
ggplot(sum.mat,aes(x=num.var, y=bic)) + geom_line(color='red') + labs(title='bic with best subset selection') + geom_vline(xintercept = sum.mat$num.var[which.min(sum.mat$bic)],color='green')
```

![](6장_Lab_files/figure-markdown_github/unnamed-chunk-4-2.png)

Forward and Backward stepwise selection
=======================================

best subset selection을 실행하는 regsubsets func에서 method = 'forward' 또는 'backward'라고 하면 된다.

``` r
mod.fwd = regsubsets(Salary~. , data=hitters, nvmax = 19, method = 'forward')
summary(mod.fwd)
```

    ## Subset selection object
    ## Call: regsubsets.formula(Salary ~ ., data = hitters, nvmax = 19, method = "forward")
    ## 19 Variables  (and intercept)
    ##            Forced in Forced out
    ## AtBat          FALSE      FALSE
    ## Hits           FALSE      FALSE
    ## HmRun          FALSE      FALSE
    ## Runs           FALSE      FALSE
    ## RBI            FALSE      FALSE
    ## Walks          FALSE      FALSE
    ## Years          FALSE      FALSE
    ## CAtBat         FALSE      FALSE
    ## CHits          FALSE      FALSE
    ## CHmRun         FALSE      FALSE
    ## CRuns          FALSE      FALSE
    ## CRBI           FALSE      FALSE
    ## CWalks         FALSE      FALSE
    ## LeagueN        FALSE      FALSE
    ## DivisionW      FALSE      FALSE
    ## PutOuts        FALSE      FALSE
    ## Assists        FALSE      FALSE
    ## Errors         FALSE      FALSE
    ## NewLeagueN     FALSE      FALSE
    ## 1 subsets of each size up to 19
    ## Selection Algorithm: forward
    ##           AtBat Hits HmRun Runs RBI Walks Years CAtBat CHits CHmRun CRuns
    ## 1  ( 1 )  " "   " "  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 2  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 3  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 4  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 5  ( 1 )  "*"   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 6  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "  
    ## 7  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "  
    ## 8  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    "*"  
    ## 9  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"  
    ## 10  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"  
    ## 11  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"  
    ## 12  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"  
    ## 13  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"  
    ## 14  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    " "   " "    "*"  
    ## 15  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    "*"   " "    "*"  
    ## 16  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"  
    ## 17  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"  
    ## 18  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   " "    "*"  
    ## 19  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   "*"    "*"  
    ##           CRBI CWalks LeagueN DivisionW PutOuts Assists Errors NewLeagueN
    ## 1  ( 1 )  "*"  " "    " "     " "       " "     " "     " "    " "       
    ## 2  ( 1 )  "*"  " "    " "     " "       " "     " "     " "    " "       
    ## 3  ( 1 )  "*"  " "    " "     " "       "*"     " "     " "    " "       
    ## 4  ( 1 )  "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 5  ( 1 )  "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 6  ( 1 )  "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 7  ( 1 )  "*"  "*"    " "     "*"       "*"     " "     " "    " "       
    ## 8  ( 1 )  "*"  "*"    " "     "*"       "*"     " "     " "    " "       
    ## 9  ( 1 )  "*"  "*"    " "     "*"       "*"     " "     " "    " "       
    ## 10  ( 1 ) "*"  "*"    " "     "*"       "*"     "*"     " "    " "       
    ## 11  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     " "    " "       
    ## 12  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     " "    " "       
    ## 13  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 14  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 15  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 16  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 17  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    "*"       
    ## 18  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    "*"       
    ## 19  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    "*"

``` r
mod.bwd = regsubsets(Salary~. , data=hitters, nvmax = 19, method = 'forward')
summary(mod.bwd)
```

    ## Subset selection object
    ## Call: regsubsets.formula(Salary ~ ., data = hitters, nvmax = 19, method = "forward")
    ## 19 Variables  (and intercept)
    ##            Forced in Forced out
    ## AtBat          FALSE      FALSE
    ## Hits           FALSE      FALSE
    ## HmRun          FALSE      FALSE
    ## Runs           FALSE      FALSE
    ## RBI            FALSE      FALSE
    ## Walks          FALSE      FALSE
    ## Years          FALSE      FALSE
    ## CAtBat         FALSE      FALSE
    ## CHits          FALSE      FALSE
    ## CHmRun         FALSE      FALSE
    ## CRuns          FALSE      FALSE
    ## CRBI           FALSE      FALSE
    ## CWalks         FALSE      FALSE
    ## LeagueN        FALSE      FALSE
    ## DivisionW      FALSE      FALSE
    ## PutOuts        FALSE      FALSE
    ## Assists        FALSE      FALSE
    ## Errors         FALSE      FALSE
    ## NewLeagueN     FALSE      FALSE
    ## 1 subsets of each size up to 19
    ## Selection Algorithm: forward
    ##           AtBat Hits HmRun Runs RBI Walks Years CAtBat CHits CHmRun CRuns
    ## 1  ( 1 )  " "   " "  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 2  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 3  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 4  ( 1 )  " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 5  ( 1 )  "*"   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
    ## 6  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "  
    ## 7  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "  
    ## 8  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    "*"  
    ## 9  ( 1 )  "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"  
    ## 10  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"  
    ## 11  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   "*"    " "   " "    "*"  
    ## 12  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"  
    ## 13  ( 1 ) "*"   "*"  " "   "*"  " " "*"   " "   "*"    " "   " "    "*"  
    ## 14  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    " "   " "    "*"  
    ## 15  ( 1 ) "*"   "*"  "*"   "*"  " " "*"   " "   "*"    "*"   " "    "*"  
    ## 16  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"  
    ## 17  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   " "   "*"    "*"   " "    "*"  
    ## 18  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   " "    "*"  
    ## 19  ( 1 ) "*"   "*"  "*"   "*"  "*" "*"   "*"   "*"    "*"   "*"    "*"  
    ##           CRBI CWalks LeagueN DivisionW PutOuts Assists Errors NewLeagueN
    ## 1  ( 1 )  "*"  " "    " "     " "       " "     " "     " "    " "       
    ## 2  ( 1 )  "*"  " "    " "     " "       " "     " "     " "    " "       
    ## 3  ( 1 )  "*"  " "    " "     " "       "*"     " "     " "    " "       
    ## 4  ( 1 )  "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 5  ( 1 )  "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 6  ( 1 )  "*"  " "    " "     "*"       "*"     " "     " "    " "       
    ## 7  ( 1 )  "*"  "*"    " "     "*"       "*"     " "     " "    " "       
    ## 8  ( 1 )  "*"  "*"    " "     "*"       "*"     " "     " "    " "       
    ## 9  ( 1 )  "*"  "*"    " "     "*"       "*"     " "     " "    " "       
    ## 10  ( 1 ) "*"  "*"    " "     "*"       "*"     "*"     " "    " "       
    ## 11  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     " "    " "       
    ## 12  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     " "    " "       
    ## 13  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 14  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 15  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 16  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    " "       
    ## 17  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    "*"       
    ## 18  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    "*"       
    ## 19  ( 1 ) "*"  "*"    "*"     "*"       "*"     "*"     "*"    "*"

변수가 7개일 때 어떤 변수들이 어떤 값으로 선택되었는지 알고 싶다면?

``` r
coef(mod.bwd,7)
```

    ##  (Intercept)        AtBat         Hits        Walks         CRBI 
    ##  109.7873062   -1.9588851    7.4498772    4.9131401    0.8537622 
    ##       CWalks    DivisionW      PutOuts 
    ##   -0.3053070 -127.1223928    0.2533404

Choosing among models using the validation set approach and cross-validation
============================================================================

train data와 test data의 index을 정한다.

``` r
set.seed(1)
train = sample(c(TRUE,FALSE),nrow(hitters),replace=TRUE)
test = !train
```

train data로 best subset selection을 해보자.

``` r
mod.best = regsubsets(Salary~. , data=hitters[train,],nvmax=19)
```

test data로부터 model matrix을 만든다.

``` r
test.mat = model.matrix(Salary~. , data=hitters[test,])
```

이제 validation error을 계산해보자.

``` r
val.error = rep(0,19)
for (i in 1:19){
  coefi = coef(mod.best,id=i)
  pred = test.mat[,names(coefi)] %*% coefi # %*%은 matrix multiplication.
  val.error[i] = mean((hitters$Salary[test]-pred)^2)
}
which.min(val.error) # test error가 가장 낮은 애는 9개의 변수를 포함하는 모델
```

    ## [1] 10

``` r
coef(mod.best,9)
```

    ##  (Intercept)        AtBat         Hits        Walks       CAtBat 
    ## -116.8513468   -1.5669672    7.6177014    3.5505374   -0.1888594 
    ##        CHits       CHmRun       CWalks      LeagueN      PutOuts 
    ##    1.1121891    1.3421445   -0.7221434   84.0143083    0.2433223

regsubsets()에 대한 predict 함수없기 때문에 위와같이 했다. 그렇다면 함수를 직접 만들어보자.

``` r
predict.regsubsets = function(object,newdata,id,...){
  form = as.formula(object$call[[2]]) # object은 regsubsets() object
  mat = model.matrix(form,newdata) # model.matrix(model fit한 formula, data)
  coefi = coef(object, id=id) # regsubsets() 결과에서 var이 ~개일 때의 coef 저장하기
  xvars = names(coefi) # coefi 칼럼명 뽑아내기
  mar[,xvars]%*%coefi
}
```

Ridge Regression
----------------

ridge와 lasso을 시행하는 함수 glmnet()은 glmnet 패키지 안에 있으며 다른 모델 적합 함수와는 문법이 약간 다르다. matrix 형태의 x와 vector 형태의 y를 쓴다.

``` r
x = model.matrix(Salary~. , hitters)[,-1]
y = hitters$Salary
```

model.matrix()는 x를 만드는데 유용한 함수이다. 19개의 예측변수를 포함하는 행렬을 만들분만 아니라 범주형 자료를 자동으로 더미화시켜준다. glmnet() 함수는 수치형 자료만 받기 때문에 이렇더 더미화 해주는 것이 중요하다.

``` r
library(glmnet)
```

    ## Loading required package: Matrix

    ## Loading required package: foreach

    ## Loaded glmnet 2.0-16

``` r
grid = 10^seq(10,-2,length=100)
mod.ridge = glmnet(x,y,alpha=0,lambda=grid) 
```

glmnet()은 alpha = 0이면 ridge이고 = 1이면 lasso을 시행한다. 원래 glmnet()은 자동적으로 람다를 선택하는데 여기서는 10^10부터 10^2까지 해봄. 또한 glmnet()은 자동적으로 변수를 표준화(standardize) 해준다(lasso, ridge을 할때 변수를 표준화해주라고 본문에 나와있음.) 이 기능을 없애고 싶으면 standardize=FALSE 해주면 됨. 모델 적합 결과는 람다 값에 따른 matrix 형태이다. 여기서는 변수가 20개 있고 람다 값을 100개 설정했으므로 20 \* 100 matrix이다.

``` r
dim(coef(mod.ridge))
```

    ## [1]  20 100

본문에서 살펴봤듯이, 람다가 커질수록(l2 norm이 작아질수록), penalty를 많이 준다는 뜻이고, 계수들이 작아질 것이다.

``` r
mod.ridge$lambda[50] # 람다가 11498일때
```

    ## [1] 11497.57

``` r
coef(mod.ridge)[,50] # 계수들을 살펴보자
```

    ##   (Intercept)         AtBat          Hits         HmRun          Runs 
    ## 407.356050200   0.036957182   0.138180344   0.524629976   0.230701523 
    ##           RBI         Walks         Years        CAtBat         CHits 
    ##   0.239841459   0.289618741   1.107702929   0.003131815   0.011653637 
    ##        CHmRun         CRuns          CRBI        CWalks       LeagueN 
    ##   0.087545670   0.023379882   0.024138320   0.025015421   0.085028114 
    ##     DivisionW       PutOuts       Assists        Errors    NewLeagueN 
    ##  -6.215440973   0.016482577   0.002612988  -0.020502690   0.301433531

``` r
sqrt(sum(coef(mod.ridge)[-1,50]^2)) # 절편 제외 l2 norm은??
```

    ## [1] 6.360612

``` r
mod.ridge$lambda[60] # 람다가 705일때
```

    ## [1] 705.4802

``` r
coef(mod.ridge)[,60] # 계수들을 살펴보자
```

    ##  (Intercept)        AtBat         Hits        HmRun         Runs 
    ##  54.32519950   0.11211115   0.65622409   1.17980910   0.93769713 
    ##          RBI        Walks        Years       CAtBat        CHits 
    ##   0.84718546   1.31987948   2.59640425   0.01083413   0.04674557 
    ##       CHmRun        CRuns         CRBI       CWalks      LeagueN 
    ##   0.33777318   0.09355528   0.09780402   0.07189612  13.68370191 
    ##    DivisionW      PutOuts      Assists       Errors   NewLeagueN 
    ## -54.65877750   0.11852289   0.01606037  -0.70358655   8.61181213

``` r
sqrt(sum(coef(mod.ridge)[-1,60]^2)) # 절편 제외 l2 norm은??
```

    ## [1] 57.11001

l2 norm이 람다가 작아짐에 따라 커졌음을 확인할 있다!!

predict() func을 새로운 람다에 대해서 ridge reg을 할 때에도 사용할 수 있다. 예를 들어 람다 50에 대해서는,

``` r
predict(mod.ridge,s=50,type='coefficients')[1:20,]
```

    ##   (Intercept)         AtBat          Hits         HmRun          Runs 
    ##  4.876610e+01 -3.580999e-01  1.969359e+00 -1.278248e+00  1.145892e+00 
    ##           RBI         Walks         Years        CAtBat         CHits 
    ##  8.038292e-01  2.716186e+00 -6.218319e+00  5.447837e-03  1.064895e-01 
    ##        CHmRun         CRuns          CRBI        CWalks       LeagueN 
    ##  6.244860e-01  2.214985e-01  2.186914e-01 -1.500245e-01  4.592589e+01 
    ##     DivisionW       PutOuts       Assists        Errors    NewLeagueN 
    ## -1.182011e+02  2.502322e-01  1.215665e-01 -3.278600e+00 -9.496680e+00

이제 train이랑 test으로 쪼개서 모델을 적합시키는 과정을 살펴본다. 보통 데이터를 이렇게 두개로 쪼개는 것은 두 가방 방법이 있는데 TRUE FALSE 벡터를 생성하는 것, 또는 1:n 중 train data index을 뽑는 것이다.

``` r
set.seed(1)
train = sample(1:nrow(x),nrow(x)/2)
test = -train
y.test = y[test]
```

이제 train data로 적합하 하고 test data로 성능평 평가해보자.

``` r
mod.ridge = glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh=1e-12)
ridge.pred = predict(mod.ridge,s=4,newx=x[test,])
mean((y.test - ridge.pred)^2)
```

    ## [1] 101036.8

절편만 포함하는 ridge는?

``` r
mean((mean(y[train])-y.test)^2)
```

    ## [1] 193253.1

이제 cross validation을 해보자! glmnet library안에 cv.glmnet()가 이미 내장되어 있다. 기본값은 10 folds이다. 바꾸고 싶으면 nfolds =.

``` r
set.seed(1)
cv.out = cv.glmnet(x[train,],y[train],alpha=0)
cv.out$lambda.min
```

    ## [1] 211.7416

가장 작은 cross validation error을 출력하는 람다는 212이다.그렇다면 이때 test mse는 몇일까?

``` r
ridge.pred = predict(mod.ridge,s=212,newx=x[test,])
mean((y[test]-ridge.pred)^2)
```

    ## [1] 96015.27

이제 cv를 이용해서 가장 작은 test mse를 출력하는 람다를 찾았으니 전체 train data로 적합을 해보자!

``` r
final.out = glmnet(x,y,alpha=0,lambda=212)
coef(final.out)
```

    ## 20 x 1 sparse Matrix of class "dgCMatrix"
    ##                       s0
    ## (Intercept)   9.81495442
    ## AtBat         0.03191025
    ## Hits          1.00790629
    ## HmRun         0.14204716
    ## Runs          1.11283540
    ## RBI           0.87320884
    ## Walks         1.80310238
    ## Years         0.14156997
    ## CAtBat        0.01114964
    ## CHits         0.06486802
    ## CHmRun        0.45137750
    ## CRuns         0.12884114
    ## CRBI          0.13722324
    ## CWalks        0.02928067
    ## LeagueN      27.15822017
    ## DivisionW   -91.58770575
    ## PutOuts       0.19138790
    ## Assists       0.04243399
    ## Errors       -1.81028711
    ## NewLeagueN    7.23220165

앞서 살펴봤듯이, ridge는 어떠한 계수도 0으로 만들지 않는다. 즉, variable selection을 하지 않는다!

The Lasso
---------

lasso는 ridge와 동일하게 glmnet()을 통해서 시행되며 alpha=1라고 설정해야 한다.

``` r
mod.lasso = glmnet(x[train,],y[train],alpha=1,lambda=grid)
```

이제 cross validation을 해보자.

``` r
set.seed(1)
cv.out = cv.glmnet(x[train,],y[train],alpha=1)
cv.out$lambda.min
```

    ## [1] 16.78016

lasso의 test mse 계산하기

``` r
lasso.pred = predict(mod.lasso,s=17,newx=x[test,])
mean((y[test]-lasso.pred)^2)
```

    ## [1] 100755.1

test mse는 ridge에서 람다가 212일때랑 거의 차이가 없지만! lasso의 장점은 ridge와는 다르게 몇몇 계수를 0으로 만든다는 점이다.

``` r
final.lasso.out = glmnet(x,y,lambda=17,alpha=1)
coef(final.lasso.out)
```

    ## 20 x 1 sparse Matrix of class "dgCMatrix"
    ##                       s0
    ## (Intercept)   20.2735038
    ## AtBat          .        
    ## Hits           1.8673382
    ## HmRun          .        
    ## Runs           .        
    ## RBI            .        
    ## Walks          2.2161744
    ## Years          .        
    ## CAtBat         .        
    ## CHits          .        
    ## CHmRun         .        
    ## CRuns          0.2074062
    ## CRBI           0.4121545
    ## CWalks         .        
    ## LeagueN        1.2663496
    ## DivisionW   -103.1080013
    ## PutOuts        0.2202108
    ## Assists        .        
    ## Errors         .        
    ## NewLeagueN     .

따라서 해석의 측면에서 ridge보다 더욱 강점이 있다.

Lab 3: PCR and PLS regression
=============================

Principal Components Regression
-------------------------------

PCR은 pls library의 pcr()을 통해서 할 수 있다. 기본적인 구조는 lm()과 유사하나, scale=TRUE을 통해서 표준화를 진행하여 변수의 스케일이 결과에 영향을 미치지 않도록, validation = 'CV'를 통해서 기본값으로 ten-fold-cv를 통해 M(\# of principal components)을 도출하게 한다.

``` r
library(pls)
```

    ## 
    ## Attaching package: 'pls'

    ## The following object is masked from 'package:stats':
    ## 
    ##     loadings

``` r
set.seed(2)
mod.pcr = pcr(Salary~. , data=hitters, scale=TRUE, validation = 'CV')
summary(mod.pcr)
```

    ## Data:    X dimension: 263 19 
    ##  Y dimension: 263 1
    ## Fit method: svdpc
    ## Number of components considered: 19
    ## 
    ## VALIDATION: RMSEP
    ## Cross-validated using 10 random segments.
    ##        (Intercept)  1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
    ## CV             452    348.9    352.2    353.5    352.8    350.1    349.1
    ## adjCV          452    348.7    351.8    352.9    352.1    349.3    348.0
    ##        7 comps  8 comps  9 comps  10 comps  11 comps  12 comps  13 comps
    ## CV       349.6    350.9    352.9     353.8     355.0     356.2     363.5
    ## adjCV    348.5    349.8    351.6     352.3     353.4     354.5     361.6
    ##        14 comps  15 comps  16 comps  17 comps  18 comps  19 comps
    ## CV        355.2     357.4     347.6     350.1     349.2     352.6
    ## adjCV     352.8     355.2     345.5     347.6     346.7     349.8
    ## 
    ## TRAINING: % variance explained
    ##         1 comps  2 comps  3 comps  4 comps  5 comps  6 comps  7 comps
    ## X         38.31    60.16    70.84    79.03    84.29    88.63    92.26
    ## Salary    40.63    41.58    42.17    43.22    44.90    46.48    46.69
    ##         8 comps  9 comps  10 comps  11 comps  12 comps  13 comps  14 comps
    ## X         94.96    96.28     97.26     97.98     98.65     99.15     99.47
    ## Salary    46.75    46.86     47.76     47.82     47.85     48.10     50.40
    ##         15 comps  16 comps  17 comps  18 comps  19 comps
    ## X          99.75     99.89     99.97     99.99    100.00
    ## Salary     50.55     53.01     53.85     54.61     54.61

M=0일때부터, M=19일때까지, 각각의 CV score을 볼 수 있는데, 이것이 RMSEP(root mean square error)이므로 실제 MSE를 계산하려면 제곱을 해줘야 한다.

또한 cv score를 validationplot()을 통해 그래프로 볼 수 있고, 이때 val.type='MSEP'을 통해 RMSE가 아니라 MSE로 나오게 설정할 수 있다.

``` r
validationplot(mod.pcr,val.type='MSEP')
```

![](6장_Lab_files/figure-markdown_github/unnamed-chunk-29-1.png)

그림을 보면 M=16일 때가 최소값을 가지는데 사실 M=1일 때랑 그렇게 큰 차이가 없다. 따라서 M=1이면 충분하다고 결론지을 수 있다.

이제 CV를 통해 PCR을 해보자.

``` r
set.seed(1)
mod.pcr = pcr(Salary~. , data=hitters, subset=train, scale=TRUE, validation='CV')
validationplot(mod.pcr, val.type="MSEP")
```

![](6장_Lab_files/figure-markdown_github/unnamed-chunk-30-1.png)

mod.pcr의 결과로 나온 MSEP는 어떻게 접근할까?? M=7일때 test mse가 가장 낮다.

``` r
pcr.pred = predict(mod.pcr,x[test,],ncomp=7)
mean((y[test] - pcr.pred)^2)
```

    ## [1] 96556.22

test mse가 ridge와 비슷하게 낮지만 pcr은 variable selection을 하는 것도 아니고, 계수에 대해서 estimate을 하는 것도 아니기 때문에 모델을 해석하기가 더 어려워졌다!

Partial Least Squares
---------------------

PLS는 plst library에 있는 plsr() func을 이용하면 된다. 문법은 pcr() func과 비슷하다.

``` r
set.seed(1)
mod.pls = plsr(Salary~. , data=hitters, subset=train, scale=TRUE, validation='CV')
summary(mod.pls)
```

    ## Data:    X dimension: 131 19 
    ##  Y dimension: 131 1
    ## Fit method: kernelpls
    ## Number of components considered: 19
    ## 
    ## VALIDATION: RMSEP
    ## Cross-validated using 10 random segments.
    ##        (Intercept)  1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
    ## CV           464.6    394.2    391.5    393.1    395.0    415.0    424.0
    ## adjCV        464.6    393.4    390.2    391.1    392.9    411.5    418.8
    ##        7 comps  8 comps  9 comps  10 comps  11 comps  12 comps  13 comps
    ## CV       424.5    415.8    404.6     407.1     412.0     414.4     410.3
    ## adjCV    418.9    411.4    400.7     402.2     407.2     409.3     405.6
    ##        14 comps  15 comps  16 comps  17 comps  18 comps  19 comps
    ## CV        406.2     408.6     410.5     408.8     407.8     410.2
    ## adjCV     401.8     403.9     405.6     404.1     403.2     405.5
    ## 
    ## TRAINING: % variance explained
    ##         1 comps  2 comps  3 comps  4 comps  5 comps  6 comps  7 comps
    ## X         38.12    53.46    66.05    74.49    79.33    84.56    87.09
    ## Salary    33.58    38.96    41.57    42.43    44.04    45.59    47.05
    ##         8 comps  9 comps  10 comps  11 comps  12 comps  13 comps  14 comps
    ## X         90.74    92.55     93.94     97.23     97.88     98.35     98.85
    ## Salary    47.53    48.42     49.68     50.04     50.54     50.78     50.92
    ##         15 comps  16 comps  17 comps  18 comps  19 comps
    ## X          99.11     99.43     99.78     99.99    100.00
    ## Salary     51.04     51.11     51.15     51.16     51.18

가장 작은 cv error는 M = 2일때 발생하였으며, 이때 test mse를 계산해보면,

``` r
pls.pred = predict(mod.pls, newdata=hitters[test,], ncomp=2)
mean((y[test]-pls.pred)^2)
```

    ## [1] 101417.5

PCR과 PLS의 결과를 살펴보면 PCR은 M = 7일때 percentage of variance가 46%였는데, PLS는 M = 2일때 거의 동일하게 percentage of variance가 46%이다. 이는 PCR이 오직 예측변수에서 variance의 양을 최소화하려고 한다면(unsupervised learning) PLS는 예측변수와 반응변수(supervised learning) 에서 variance을 설명하는 direction을 찾기 때문에 발생하는 것이다.
