Assignment 5
================
Chee Kay Cheong
2023-02-14

``` r
knitr::opts_chunk$set(warning = FALSE, message = FALSE, set.seed(123))

library(tidyverse) 
library(caret) 
library(glmnet)
```

# Load and clean dataset, data partitioning

``` r
# Load and clean dataset
alcohol = read_csv("./Data/alcohol_use.csv") %>%  
  janitor::clean_names() %>% 
  select(-x1) %>% 
  mutate(
    alc_consumption = as_factor(alc_consumption))

# Check the distribution of the outcome
alcohol %>% 
  select(alc_consumption) %>% 
  group_by(alc_consumption) %>% 
  count()
```

    ## # A tibble: 2 × 2
    ## # Groups:   alc_consumption [2]
    ##   alc_consumption     n
    ##   <fct>           <int>
    ## 1 CurrentUse       1004
    ## 2 NotCurrentUse     881

``` r
# Quite balance...

# Partition data into 70/30 split
set.seed(123)

train_index = 
  alcohol$alc_consumption %>% createDataPartition(p = 0.7, list = F)

train_data = alcohol[train_index, ]
test_data = alcohol[-train_index, ]
```

# 1. Create 3 different models

### Elastic Net Model

``` r
set.seed(123)

# Set validation method and options
control.settings = trainControl(method = "repeatedcv", number = 10, repeats = 10)

# Fit model
EN_model = train(alc_consumption ~ ., data = train_data, method = "glmnet", trControl = control.settings, preProc = c("center", "scale"), tuneLength = 10)

# Find best tuned parameters
EN_model$bestTune
```

    ##    alpha    lambda
    ## 63   0.7 0.2578427

``` r
EN_model$results
```

    ##    alpha       lambda  Accuracy     Kappa AccuracySD    KappaSD
    ## 1    0.1 0.0003178806 0.7933409 0.5847801 0.03317932 0.06678598
    ## 2    0.1 0.0007343454 0.7933409 0.5847801 0.03317932 0.06678598
    ## 3    0.1 0.0016964331 0.7932651 0.5846194 0.03322057 0.06685043
    ## 4    0.1 0.0039189805 0.7928886 0.5837952 0.03325263 0.06698090
    ## 5    0.1 0.0090533531 0.7939561 0.5857737 0.03393589 0.06839537
    ## 6    0.1 0.0209144200 0.7956096 0.5888728 0.03370261 0.06796881
    ## 7    0.1 0.0483150228 0.7935607 0.5845436 0.03467826 0.06996833
    ## 8    0.1 0.1116139691 0.7934810 0.5838226 0.03483757 0.07030299
    ## 9    0.1 0.2578427450 0.7952975 0.5866169 0.03402836 0.06882050
    ## 10   0.2 0.0003178806 0.7935693 0.5852465 0.03321510 0.06684062
    ## 11   0.2 0.0007343454 0.7935693 0.5852465 0.03321510 0.06684062
    ## 12   0.2 0.0016964331 0.7931894 0.5844873 0.03357265 0.06755221
    ## 13   0.2 0.0039189805 0.7929632 0.5839716 0.03317002 0.06677702
    ## 14   0.2 0.0090533531 0.7931188 0.5841597 0.03359402 0.06769573
    ## 15   0.2 0.0209144200 0.7949255 0.5875610 0.03280969 0.06613717
    ## 16   0.2 0.0483150228 0.7915869 0.5805977 0.03508728 0.07072850
    ## 17   0.2 0.1116139691 0.7937856 0.5846355 0.03324716 0.06690148
    ## 18   0.2 0.2578427450 0.8053775 0.6067204 0.03229849 0.06544403
    ## 19   0.3 0.0003178806 0.7937955 0.5857152 0.03306792 0.06653524
    ## 20   0.3 0.0007343454 0.7937955 0.5857152 0.03306792 0.06653524
    ## 21   0.3 0.0016964331 0.7934172 0.5849521 0.03359299 0.06757117
    ## 22   0.3 0.0039189805 0.7931148 0.5843201 0.03327101 0.06693757
    ## 23   0.3 0.0090533531 0.7925098 0.5829964 0.03345672 0.06738249
    ## 24   0.3 0.0209144200 0.7941719 0.5861044 0.03306833 0.06667506
    ## 25   0.3 0.0483150228 0.7916673 0.5808665 0.03368453 0.06782602
    ## 26   0.3 0.1116139691 0.7981086 0.5931311 0.03176868 0.06404126
    ## 27   0.3 0.2578427450 0.8040161 0.6039555 0.03084530 0.06253045
    ## 28   0.4 0.0003178806 0.7935687 0.5852659 0.03325321 0.06688642
    ## 29   0.4 0.0007343454 0.7935687 0.5852659 0.03325321 0.06688642
    ## 30   0.4 0.0016964331 0.7933420 0.5848135 0.03350615 0.06739888
    ## 31   0.4 0.0039189805 0.7927360 0.5835907 0.03352959 0.06743819
    ## 32   0.4 0.0090533531 0.7922079 0.5824637 0.03347377 0.06735427
    ## 33   0.4 0.0209144200 0.7936393 0.5852021 0.03331032 0.06709444
    ## 34   0.4 0.0483150228 0.7938608 0.5853154 0.03345842 0.06745139
    ## 35   0.4 0.1116139691 0.7993167 0.5956176 0.03095988 0.06238159
    ## 36   0.4 0.2578427450 0.8179507 0.6306406 0.03156424 0.06397365
    ## 37   0.5 0.0003178806 0.7933420 0.5848242 0.03346967 0.06730623
    ## 38   0.5 0.0007343454 0.7933420 0.5848242 0.03346967 0.06730623
    ## 39   0.5 0.0016964331 0.7934189 0.5849757 0.03376922 0.06789757
    ## 40   0.5 0.0039189805 0.7922825 0.5827202 0.03329199 0.06696075
    ## 41   0.5 0.0090533531 0.7923571 0.5828248 0.03310213 0.06656317
    ## 42   0.5 0.0209144200 0.7932559 0.5844591 0.03362135 0.06773340
    ## 43   0.5 0.0483150228 0.7943911 0.5863954 0.03202476 0.06454385
    ## 44   0.5 0.1116139691 0.7952223 0.5877041 0.03295047 0.06630639
    ## 45   0.5 0.2578427450 0.8240860 0.6423666 0.02959423 0.06054204
    ## 46   0.6 0.0003178806 0.7934172 0.5849788 0.03318928 0.06674876
    ## 47   0.6 0.0007343454 0.7934172 0.5849788 0.03318928 0.06674876
    ## 48   0.6 0.0016964331 0.7933432 0.5848372 0.03371880 0.06778658
    ## 49   0.6 0.0039189805 0.7923577 0.5828850 0.03320861 0.06678515
    ## 50   0.6 0.0090533531 0.7918245 0.5818300 0.03284383 0.06604365
    ## 51   0.6 0.0209144200 0.7911335 0.5803029 0.03297763 0.06638599
    ## 52   0.6 0.0483150228 0.7917396 0.5811694 0.03269271 0.06587430
    ## 53   0.6 0.1116139691 0.7947683 0.5868395 0.03139486 0.06316084
    ## 54   0.6 0.2578427450 0.8495481 0.6917951 0.02641471 0.05511481
    ## 55   0.7 0.0003178806 0.7935699 0.5852966 0.03336601 0.06708151
    ## 56   0.7 0.0007343454 0.7935699 0.5852966 0.03336601 0.06708151
    ## 57   0.7 0.0016964331 0.7929632 0.5840984 0.03387989 0.06807347
    ## 58   0.7 0.0039189805 0.7922808 0.5827566 0.03303991 0.06645257
    ## 59   0.7 0.0090533531 0.7919766 0.5821946 0.03329643 0.06692757
    ## 60   0.7 0.0209144200 0.7909103 0.5799245 0.03313214 0.06667146
    ## 61   0.7 0.0483150228 0.7897687 0.5774303 0.03246541 0.06537466
    ## 62   0.7 0.1116139691 0.7933289 0.5839843 0.03138717 0.06312107
    ## 63   0.7 0.2578427450 0.8515189 0.6956916 0.02628155 0.05490010
    ## 64   0.8 0.0003178806 0.7935699 0.5852966 0.03336601 0.06708151
    ## 65   0.8 0.0007343454 0.7935699 0.5852966 0.03336601 0.06708151
    ## 66   0.8 0.0016964331 0.7927360 0.5836525 0.03387105 0.06806717
    ## 67   0.8 0.0039189805 0.7921299 0.5824778 0.03307414 0.06651874
    ## 68   0.8 0.0090533531 0.7918228 0.5819531 0.03365134 0.06764407
    ## 69   0.8 0.0209144200 0.7911329 0.5804338 0.03363073 0.06759677
    ## 70   0.8 0.0483150228 0.7860583 0.5702906 0.03187057 0.06408897
    ## 71   0.8 0.1116139691 0.7795366 0.5577106 0.03384453 0.06772210
    ## 72   0.8 0.2578427450 0.8515189 0.6956916 0.02628155 0.05490010
    ## 73   0.9 0.0003178806 0.7932669 0.5847047 0.03353854 0.06741722
    ## 74   0.9 0.0007343454 0.7932669 0.5847047 0.03353854 0.06741722
    ## 75   0.9 0.0016964331 0.7926590 0.5835208 0.03365564 0.06761122
    ## 76   0.9 0.0039189805 0.7920547 0.5823719 0.03326326 0.06683631
    ## 77   0.9 0.0090533531 0.7906870 0.5797189 0.03349817 0.06737840
    ## 78   0.9 0.0209144200 0.7909808 0.5801760 0.03323008 0.06679413
    ## 79   0.9 0.0483150228 0.7799240 0.5585428 0.03267570 0.06566401
    ## 80   0.9 0.1116139691 0.7695438 0.5392133 0.03171629 0.06321802
    ## 81   0.9 0.2578427450 0.8515189 0.6956916 0.02628155 0.05490010
    ## 82   1.0 0.0003178806 0.7928863 0.5839710 0.03384191 0.06801265
    ## 83   1.0 0.0007343454 0.7928863 0.5839710 0.03384191 0.06801265
    ## 84   1.0 0.0016964331 0.7925069 0.5832206 0.03371756 0.06775799
    ## 85   1.0 0.0039189805 0.7918262 0.5819511 0.03337770 0.06705674
    ## 86   1.0 0.0090533531 0.7906852 0.5797420 0.03303710 0.06638673
    ## 87   1.0 0.0209144200 0.7908328 0.5799839 0.03264334 0.06558130
    ## 88   1.0 0.0483150228 0.7743195 0.5480499 0.03404505 0.06810511
    ## 89   1.0 0.1116139691 0.7696959 0.5396293 0.03203898 0.06382718
    ## 90   1.0 0.2578427450 0.7108125 0.3955179 0.05445666 0.11692927

``` r
# Highest accuracy = 0.8515189, Kappa = 0.6956916, alpha = 0.7, lambda = 0.2578427450
```

### Traditional Logistic Regression

``` r
set.seed(124)

# Set validation method and options
control.settings = trainControl(method = "repeatedcv", number = 10, repeats = 10)

# Fit model
logreg = train(alc_consumption ~ ., data = train_data, method = "glmnet", family = 'binomial', trControl = control.settings, preProc = c("center", "scale"), tuneLength = 10) 

# Find best tuned parameters
logreg$bestTune
```

    ##    alpha    lambda
    ## 63   0.7 0.2578427

``` r
logreg$results
```

    ##    alpha       lambda  Accuracy     Kappa AccuracySD    KappaSD
    ## 1    0.1 0.0003178806 0.7940725 0.5862230 0.03308505 0.06657178
    ## 2    0.1 0.0007343454 0.7940725 0.5862230 0.03308505 0.06657178
    ## 3    0.1 0.0016964331 0.7939962 0.5860670 0.03318001 0.06676034
    ## 4    0.1 0.0039189805 0.7933901 0.5847815 0.03399521 0.06837770
    ## 5    0.1 0.0090533531 0.7946051 0.5869966 0.03375538 0.06801189
    ## 6    0.1 0.0209144200 0.7958958 0.5894029 0.03521279 0.07098794
    ## 7    0.1 0.0483150228 0.7945299 0.5864659 0.03556557 0.07169215
    ## 8    0.1 0.1116139691 0.7941597 0.5852225 0.03432882 0.06918495
    ## 9    0.1 0.2578427450 0.7968067 0.5897094 0.03331321 0.06727704
    ## 10   0.2 0.0003178806 0.7940725 0.5862232 0.03299732 0.06639461
    ## 11   0.2 0.0007343454 0.7940725 0.5862232 0.03299732 0.06639461
    ## 12   0.2 0.0016964331 0.7937706 0.5856238 0.03306289 0.06653642
    ## 13   0.2 0.0039189805 0.7934681 0.5849399 0.03395997 0.06831670
    ## 14   0.2 0.0090533531 0.7940736 0.5860082 0.03371437 0.06784789
    ## 15   0.2 0.0209144200 0.7948352 0.5873570 0.03415823 0.06881536
    ## 16   0.2 0.0483150228 0.7926405 0.5827602 0.03500944 0.07054616
    ## 17   0.2 0.1116139691 0.7949896 0.5869830 0.03333344 0.06714783
    ## 18   0.2 0.2578427450 0.8057481 0.6074645 0.03319404 0.06714514
    ## 19   0.3 0.0003178806 0.7937695 0.5856354 0.03293041 0.06627346
    ## 20   0.3 0.0007343454 0.7937695 0.5856354 0.03293041 0.06627346
    ## 21   0.3 0.0016964331 0.7936949 0.5854822 0.03271584 0.06584419
    ## 22   0.3 0.0039189805 0.7933924 0.5848348 0.03330582 0.06697061
    ## 23   0.3 0.0090533531 0.7935444 0.5850415 0.03363961 0.06766702
    ## 24   0.3 0.0209144200 0.7944582 0.5866805 0.03384671 0.06821224
    ## 25   0.3 0.0483150228 0.7918060 0.5811783 0.03469614 0.06989957
    ## 26   0.3 0.1116139691 0.7976383 0.5921530 0.03310875 0.06678934
    ## 27   0.3 0.2578427450 0.8039924 0.6039291 0.03214687 0.06507649
    ## 28   0.4 0.0003178806 0.7938452 0.5857930 0.03283729 0.06607394
    ## 29   0.4 0.0007343454 0.7938452 0.5857930 0.03283729 0.06607394
    ## 30   0.4 0.0016964331 0.7936202 0.5853537 0.03292248 0.06623553
    ## 31   0.4 0.0039189805 0.7935445 0.5851780 0.03292821 0.06622962
    ## 32   0.4 0.0090533531 0.7933952 0.5848036 0.03305947 0.06644892
    ## 33   0.4 0.0209144200 0.7935450 0.5850401 0.03280755 0.06594079
    ## 34   0.4 0.0483150228 0.7943049 0.5862373 0.03430483 0.06909674
    ## 35   0.4 0.1116139691 0.7994537 0.5959060 0.02958670 0.05983125
    ## 36   0.4 0.2578427450 0.8187695 0.6322470 0.03309424 0.06704502
    ## 37   0.5 0.0003178806 0.7937683 0.5856586 0.03285114 0.06610870
    ## 38   0.5 0.0007343454 0.7937683 0.5856586 0.03285114 0.06610870
    ## 39   0.5 0.0016964331 0.7935428 0.5852084 0.03284969 0.06610330
    ## 40   0.5 0.0039189805 0.7932386 0.5846043 0.03297981 0.06633367
    ## 41   0.5 0.0090533531 0.7928638 0.5838089 0.03293096 0.06620080
    ## 42   0.5 0.0209144200 0.7924098 0.5828351 0.03270547 0.06571324
    ## 43   0.5 0.0483150228 0.7945333 0.5867404 0.03215805 0.06477884
    ## 44   0.5 0.1116139691 0.7958115 0.5888525 0.03029683 0.06118513
    ## 45   0.5 0.2578427450 0.8247528 0.6436459 0.03209681 0.06565406
    ## 46   0.6 0.0003178806 0.7936937 0.5855088 0.03275961 0.06594351
    ## 47   0.6 0.0007343454 0.7936937 0.5855088 0.03275961 0.06594351
    ## 48   0.6 0.0016964331 0.7933924 0.5849298 0.03258648 0.06557262
    ## 49   0.6 0.0039189805 0.7925613 0.5832959 0.03280126 0.06596031
    ## 50   0.6 0.0090533531 0.7927117 0.5835767 0.03271433 0.06576619
    ## 51   0.6 0.0209144200 0.7910473 0.5801457 0.03335448 0.06706096
    ## 52   0.6 0.0483150228 0.7924872 0.5826460 0.03070824 0.06192300
    ## 53   0.6 0.1116139691 0.7945201 0.5863086 0.03044524 0.06147573
    ## 54   0.6 0.2578427450 0.8495467 0.6917443 0.02840780 0.05921056
    ## 55   0.7 0.0003178806 0.7935428 0.5852397 0.03258699 0.06557700
    ## 56   0.7 0.0007343454 0.7935428 0.5852397 0.03258699 0.06557700
    ## 57   0.7 0.0016964331 0.7930130 0.5841961 0.03251820 0.06542690
    ## 58   0.7 0.0039189805 0.7926371 0.5834777 0.03297563 0.06629154
    ## 59   0.7 0.0090533531 0.7925607 0.5833349 0.03259680 0.06554121
    ## 60   0.7 0.0209144200 0.7904400 0.5789894 0.03355194 0.06741610
    ## 61   0.7 0.0483150228 0.7907413 0.5793074 0.03098727 0.06236643
    ## 62   0.7 0.1116139691 0.7916447 0.5806949 0.03065689 0.06170243
    ## 63   0.7 0.2578427450 0.8515188 0.6956381 0.02778528 0.05800645
    ## 64   0.8 0.0003178806 0.7935433 0.5852567 0.03242439 0.06525132
    ## 65   0.8 0.0007343454 0.7935433 0.5852567 0.03242439 0.06525132
    ## 66   0.8 0.0016964331 0.7932409 0.5846557 0.03257508 0.06554903
    ## 67   0.8 0.0039189805 0.7918783 0.5819958 0.03280210 0.06598320
    ## 68   0.8 0.0090533531 0.7927869 0.5838738 0.03229745 0.06489550
    ## 69   0.8 0.0209144200 0.7915764 0.5813103 0.03380318 0.06796688
    ## 70   0.8 0.0483150228 0.7864931 0.5711418 0.03104511 0.06236238
    ## 71   0.8 0.1116139691 0.7802710 0.5591077 0.03308260 0.06619852
    ## 72   0.8 0.2578427450 0.8515188 0.6956381 0.02778528 0.05800645
    ## 73   0.9 0.0003178806 0.7935433 0.5852641 0.03238861 0.06518188
    ## 74   0.9 0.0007343454 0.7935433 0.5852641 0.03238861 0.06518188
    ## 75   0.9 0.0016964331 0.7931651 0.5845164 0.03277520 0.06594985
    ## 76   0.9 0.0039189805 0.7918026 0.5818655 0.03257103 0.06550997
    ## 77   0.9 0.0090533531 0.7928609 0.5840739 0.03247413 0.06524629
    ## 78   0.9 0.0209144200 0.7908182 0.5798656 0.03395601 0.06826739
    ## 79   0.9 0.0483150228 0.7798176 0.5583635 0.03302176 0.06612919
    ## 80   0.9 0.1116139691 0.7700521 0.5401867 0.03458083 0.06897154
    ## 81   0.9 0.2578427450 0.8515188 0.6956381 0.02778528 0.05800645
    ## 82   1.0 0.0003178806 0.7933161 0.5848186 0.03236530 0.06513575
    ## 83   1.0 0.0007343454 0.7933161 0.5848186 0.03236530 0.06513575
    ## 84   1.0 0.0016964331 0.7929378 0.5840719 0.03271565 0.06583740
    ## 85   1.0 0.0039189805 0.7919535 0.5821833 0.03253379 0.06544084
    ## 86   1.0 0.0090533531 0.7919524 0.5823060 0.03219297 0.06466349
    ## 87   1.0 0.0209144200 0.7904354 0.5791754 0.03327427 0.06685971
    ## 88   1.0 0.0483150228 0.7742131 0.5477575 0.03387569 0.06771165
    ## 89   1.0 0.1116139691 0.7696733 0.5395763 0.03411591 0.06801996
    ## 90   1.0 0.2578427450 0.7115226 0.3971621 0.05326149 0.11421142

``` r
# Highest accuracy = 0.8515188, Kappa = 0.6956381, alpha = 0.7, lambda = 0.2578427450
```

### LASSO Model

``` r
set.seed(125)

lambda = 10^seq(-3, 1, length = 100)
lambda.grid = expand.grid(alpha = 1, lambda = lambda)

lasso = train(alc_consumption ~ ., data = train_data, method = "glmnet", preProc = c("center", "scale"), trControl = control.settings, tuneGrid = lambda.grid)

# Find best tuned parameters
lasso$bestTune
```

    ##    alpha    lambda
    ## 60     1 0.2420128

``` r
lasso$results
```

    ##     alpha       lambda  Accuracy        Kappa  AccuracySD     KappaSD
    ## 1       1  0.001000000 0.7951579 0.5884782793 0.034509697 0.069380020
    ## 2       1  0.001097499 0.7951567 0.5884754432 0.034362998 0.069089531
    ## 3       1  0.001204504 0.7950816 0.5883371875 0.034485798 0.069331704
    ## 4       1  0.001321941 0.7950821 0.5883352004 0.034364011 0.069108642
    ## 5       1  0.001450829 0.7951585 0.5884851903 0.034438255 0.069252977
    ## 6       1  0.001592283 0.7951585 0.5884914682 0.034438255 0.069262475
    ## 7       1  0.001747528 0.7950827 0.5883537873 0.034310389 0.068985676
    ## 8       1  0.001917910 0.7950822 0.5883658454 0.033855714 0.068029816
    ## 9       1  0.002104904 0.7949312 0.5880699672 0.033834528 0.067988654
    ## 10      1  0.002310130 0.7950075 0.5882287734 0.034009964 0.068332686
    ## 11      1  0.002535364 0.7949295 0.5880783929 0.034318262 0.068939177
    ## 12      1  0.002782559 0.7948537 0.5879401533 0.034374789 0.069050436
    ## 13      1  0.003053856 0.7947016 0.5876590196 0.034122274 0.068535434
    ## 14      1  0.003351603 0.7945512 0.5873659426 0.034195846 0.068676417
    ## 15      1  0.003678380 0.7944738 0.5872125746 0.034010983 0.068322079
    ## 16      1  0.004037017 0.7947762 0.5878210394 0.034279896 0.068860063
    ## 17      1  0.004430621 0.7948520 0.5879813645 0.034491664 0.069307140
    ## 18      1  0.004862602 0.7946259 0.5875508492 0.034333528 0.068972664
    ## 19      1  0.005336699 0.7940187 0.5863505405 0.034726314 0.069740340
    ## 20      1  0.005857021 0.7936410 0.5856333636 0.034708447 0.069722325
    ## 21      1  0.006428073 0.7932611 0.5849039285 0.034429482 0.069129350
    ## 22      1  0.007054802 0.7930332 0.5844664406 0.034575527 0.069430936
    ## 23      1  0.007742637 0.7928048 0.5840290422 0.034591358 0.069451215
    ## 24      1  0.008497534 0.7925046 0.5834067043 0.034561817 0.069366479
    ## 25      1  0.009326033 0.7918205 0.5820294235 0.033973613 0.068173634
    ## 26      1  0.010235310 0.7918951 0.5821706642 0.034122276 0.068550311
    ## 27      1  0.011233240 0.7909854 0.5803295484 0.034459587 0.069206344
    ## 28      1  0.012328467 0.7912115 0.5807996342 0.034523838 0.069324459
    ## 29      1  0.013530478 0.7911352 0.5806399493 0.033700120 0.067697795
    ## 30      1  0.014849683 0.7914400 0.5812091904 0.033480931 0.067246374
    ## 31      1  0.016297508 0.7904511 0.5792287235 0.033517962 0.067302669
    ## 32      1  0.017886495 0.7901480 0.5786595230 0.034171655 0.068617827
    ## 33      1  0.019630407 0.7903730 0.5791052273 0.034041469 0.068365541
    ## 34      1  0.021544347 0.7903719 0.5790886168 0.033255182 0.066803082
    ## 35      1  0.023644894 0.7902209 0.5788059357 0.033151689 0.066621222
    ## 36      1  0.025950242 0.7896160 0.5776530109 0.032805113 0.065902645
    ## 37      1  0.028480359 0.7890088 0.5764930922 0.032589529 0.065452723
    ## 38      1  0.031257158 0.7877209 0.5740315879 0.032755496 0.065694793
    ## 39      1  0.034304693 0.7851451 0.5690128512 0.033087362 0.066282098
    ## 40      1  0.037649358 0.7828717 0.5645664899 0.033437604 0.066942039
    ## 41      1  0.041320124 0.7802213 0.5594505108 0.032502016 0.065205613
    ## 42      1  0.045348785 0.7774946 0.5541999484 0.032603417 0.065402601
    ## 43      1  0.049770236 0.7744636 0.5483412376 0.032722964 0.065588163
    ## 44      1  0.054622772 0.7710544 0.5417796574 0.031855673 0.063772840
    ## 45      1  0.059948425 0.7704478 0.5407149274 0.032044317 0.064109633
    ## 46      1  0.065793322 0.7698412 0.5396505265 0.031853795 0.063745212
    ## 47      1  0.072208090 0.7694584 0.5390421605 0.032070836 0.064157474
    ## 48      1  0.079248290 0.7693080 0.5387922083 0.032098708 0.064237838
    ## 49      1  0.086974900 0.7694589 0.5391497228 0.031917972 0.063884373
    ## 50      1  0.095454846 0.7696856 0.5396169551 0.032060643 0.064165629
    ## 51      1  0.104761575 0.7696856 0.5396169551 0.032060643 0.064165629
    ## 52      1  0.114975700 0.7696856 0.5396169551 0.032060643 0.064165629
    ## 53      1  0.126185688 0.7696856 0.5396169551 0.032060643 0.064165629
    ## 54      1  0.138488637 0.7696856 0.5396169551 0.032060643 0.064165629
    ## 55      1  0.151991108 0.7710470 0.5420861555 0.034701694 0.068927246
    ## 56      1  0.166810054 0.8463109 0.6857601114 0.032122070 0.064699858
    ## 57      1  0.183073828 0.8515234 0.6957193637 0.026233844 0.054692937
    ## 58      1  0.200923300 0.8515234 0.6957193637 0.026233844 0.054692937
    ## 59      1  0.220513074 0.8515234 0.6957193637 0.026233844 0.054692937
    ## 60      1  0.242012826 0.8515234 0.6957193637 0.026233844 0.054692937
    ## 61      1  0.265608778 0.6878664 0.3462182635 0.025381950 0.055008641
    ## 62      1  0.291505306 0.5327280 0.0003417969 0.002671142 0.002404631
    ## 63      1  0.319926714 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 64      1  0.351119173 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 65      1  0.385352859 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 66      1  0.422924287 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 67      1  0.464158883 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 68      1  0.509413801 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 69      1  0.559081018 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 70      1  0.613590727 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 71      1  0.673415066 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 72      1  0.739072203 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 73      1  0.811130831 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 74      1  0.890215085 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 75      1  0.977009957 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 76      1  1.072267222 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 77      1  1.176811952 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 78      1  1.291549665 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 79      1  1.417474163 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 80      1  1.555676144 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 81      1  1.707352647 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 82      1  1.873817423 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 83      1  2.056512308 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 84      1  2.257019720 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 85      1  2.477076356 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 86      1  2.718588243 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 87      1  2.983647240 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 88      1  3.274549163 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 89      1  3.593813664 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 90      1  3.944206059 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 91      1  4.328761281 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 92      1  4.750810162 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 93      1  5.214008288 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 94      1  5.722367659 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 95      1  6.280291442 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 96      1  6.892612104 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 97      1  7.564633276 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 98      1  8.302175681 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 99      1  9.111627561 0.5325765 0.0000000000 0.002587401 0.000000000
    ## 100     1 10.000000000 0.5325765 0.0000000000 0.002587401 0.000000000

``` r
# Highest accuracy = 0.8515234, Kappa = 0.6957193637, alpha = 1, lambda = 0.242012826
```

# 2. Decide final model

Based on the performance tests of all three models, I decided to use the
**LASSO** model because it has the highest accuracy (0.8515234), as well
as the highest Kappa value (0.6957193637), among all three models. The
best tune of the LASSO model is lambda equals 0.242012826.

- The **Kappa** value is a metric that compares an *Observed Accuracy*
  with an *Expected Accuracy* (random chance). The kappa statistic is
  used not only to evaluate a single classifier, but also to evaluate
  classifiers amongst themselves. It is a measurement of the accuracy of
  a model while taking into account chance. The closer the value is to 1
  the better.
  - Cited: rbx (<https://stats.stackexchange.com/users/37276/rbx>),
    Cohen’s kappa in plain English, URL (version: 2017-10-29):
    <https://stats.stackexchange.com/q/82187>

# 3. Apply final model in the test set

I decided to use Confusion Matrix as my final evaluation metric.

``` r
# Test model
test_outcome = predict(lasso, test_data)

# Evaluation metric:
confusionMatrix(test_outcome, test_data$alc_consumption, positive = "CurrentUse")
```

    ## Confusion Matrix and Statistics
    ## 
    ##                Reference
    ## Prediction      CurrentUse NotCurrentUse
    ##   CurrentUse           301            82
    ##   NotCurrentUse          0           182
    ##                                           
    ##                Accuracy : 0.8549          
    ##                  95% CI : (0.8231, 0.8829)
    ##     No Information Rate : 0.5327          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.7028          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ##                                           
    ##             Sensitivity : 1.0000          
    ##             Specificity : 0.6894          
    ##          Pos Pred Value : 0.7859          
    ##          Neg Pred Value : 1.0000          
    ##              Prevalence : 0.5327          
    ##          Detection Rate : 0.5327          
    ##    Detection Prevalence : 0.6779          
    ##       Balanced Accuracy : 0.8447          
    ##                                           
    ##        'Positive' Class : CurrentUse      
    ## 

# Report Analysis and Results

Before conducting any analysis, I cleaned the `alcohol_use` dataset and
checked the distribution of the outcome (`alc_consumption`). The
proportion of `*CurrentUse*` is 0.53, while the proportion of
`*NotCurrentUse*` is 0.47. I think they are pretty balance so I decided
not to down sample the outcome during `trainControl`. I partitioned the
dataset into a 70/30 split.

To better predict the current alcohol consumption, I have created and
compared three different models (Elastic Net, Logistic Regression, and
LASSO). Among the three models, LASSO gives the highest accuracy
(0.8515234) and Kappa value (0.6957193637). The best tune lambda of this
model is 0.242012826.

I then fit the LASSO model to the testing dataset to test its
performance. It turned out that this model has an accuracy of 85.5%,
with a sensitivity of 1, a specificity of 0.69, a positive predictive
value of 0.7859, and a negative predictive value of 1.

# Problem 5

Assuming we have a another 2000 participants who had taken the
behavioral test but did not answer whether they currently consume
alcohol or not. We can use this analysis to directly predict the current
alcohol consumption for this group of people using their test scores. On
another instance, if we were interested in seeking the association
between alcohol consumption and liver cancer among this population, this
analysis can indirectly provide information about the prediction of
alcohol consumption of each individual in this population.