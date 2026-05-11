# Try SMOTE oversampling
python manage.py train_balanced_model --symbols-file symbols.txt --years 7 --sampling-method smote --output-model smote_model

# Try ADASYN
python manage.py train_balanced_model --symbols-file symbols.txt --years 7 --sampling-method adasyn --output-model adasyn_model

# Try SMOTE + TOMEK
python manage.py train_balanced_model --symbols-file symbols.txt --years 7 --sampling-method smote_tomek --output-model smote_tomek_model

# Try class weights
python manage.py train_balanced_model --symbols-file symbols.txt --years 7 --sampling-method class_weight --output-model weighted_model

# Try with lower RR ratio (more balanced labels)
python manage.py train_balanced_model --symbols-file symbols.txt --years 7 --rr-ratios "1.5,2.0" --sampling-method smote

-----
Best Model: smote_model_RR1.5_smote
  RR Ratio: 3.0:1
  Accuracy: 89.43%
  Precision: 100.00%
  Recall: 89.43%
  F1 Score: 94.42%
  Expected Value per trade: 3.000-

Best Model: adasyn_model_RR1.5_adasyn
  RR Ratio: 3.0:1
  Accuracy: 86.65%
  Precision: 100.00%
  Recall: 86.65%
  F1 Score: 92.85%
  Expected Value per trade: 3.000

Best Model: smote_tomek_model_RR1.5_smote_tomek
  RR Ratio: 3.0:1
  Accuracy: 89.56%
  Precision: 100.00%
  Recall: 89.56%
  F1 Score: 94.49%
  Expected Value per trade: 3.000

Best Model: weighted_model_RR2.0_class_weight
  RR Ratio: 2.0:1
  Accuracy: 61.42%
  Precision: 47.95%
  Recall: 35.00%
  F1 Score: 40.46%
  Expected Value per trade: 0.438

Best Model: balanced_trend_model_RR1.5_smote
  RR Ratio: 2.0:1
  Accuracy: 71.26%
  Precision: 100.00%
  Recall: 71.26%
  F1 Score: 83.22%
  Expected Value per trade: 2.000


-----------------------

python manage.py train_with_validation_split --symbols-file tickers.txt --train-end-date 2019-12-31 --test-years 7 --rr-ratio 3.0 --sampling-method smote --output-model sp500_model

============================================================                                                                                            
COMBINED TRAINING DATA                                                                                                                                  
============================================================                                                                                            
Total training samples: 81016                                                                                                                           
From 478 symbols                                                                                                                                        
Class distribution: {0: 59143, 1: 21873}                                                                                                                
Train samples: 64812, Validation samples: 16204                                                                                                         
Applying SMOTE oversampling...                                                                                                                          
After SMOTE: Class 0=47402, Class 1=47402                                                                                                               
Training XGBoost model...                                                                                                                               
Validation Accuracy: 47.19%                                                                                                                             
Validation Precision: 100.00%                                                                                                                           
Validation Recall: 47.19%                                                                                                                               
Expected Value per Trade (RR 3.0:1): 3.000R                                                                                                             
  Created dataset: sp500_model_dataset_20260328_161125                                                                                                  
✓ Model saved: sp500_model_RR3.0_20260328_161125                                                                                                        
  Model ID: 13                                                                                                                                          
                                                                                                                                                        
============================================================                                                                                            
VALIDATING ON OUT-OF-SAMPLE TEST DATA                                                                                                                   
============================================================                                                                                            
Test samples: 22940                                                                                                                                     
Test symbols: 478                                                                                                                                       
                                                                                                                                                        
Threshold    Precision    Recall       Signals    Expected Value                                                                                        
----------------------------------------------------------------------                                                                                  
0.50      39.53%      0.27%          43      +0.581R      ✓                                                                                             
0.55      9.09%      0.02%          11      -0.636R      ✗                                                                                              

============================================================
OUT-OF-SAMPLE TEST SUMMARY
============================================================
Best Threshold: 0.50
Expected Value: 0.581R per trade
Test Period: 7 years after 2019-12-31

Break-even Precision Needed: 25.0%

✓ Model is PROFITABLE on out-of-sample data!
  For every $100 risked, expect $58 profit

Recommended trading parameters:
  - Probability threshold: 0.50
  - Risk:Reward ratio: 3.0:1
  - Expected win rate: 39.5%

Test results saved to: test_results_sp500_model_RR3.0_20260328_161125_20260328_161125.json

============================================================
FEATURE IMPORTANCE ANALYSIS
============================================================
Top 10 Most Important Features:
  1. stoch_oversold: 0.4688
  2. stoch_cross_below: 0.0612
  3. regression_cross_down: 0.0409
  4. above_vwap: 0.0286
  5. stoch_overbought: 0.0270
  6. below_vwap: 0.0257
  7. vwap_deviation: 0.0248
  8. return_5d: 0.0240
  9. regression_cross_up: 0.0237
  10. volume_ratio: 0.0233

--------------------------------------------------------------------------------------------------------

python manage.py train_with_validation_split --symbols-file tickers.txt --train-end-date 2019-12-31 --test-years 7 --rr-ratio 2.0 --sampling-method smote --output-model sp500_model


============================================================                                                                                            
COMBINED TRAINING DATA                                                                                                                                  
============================================================                                                                                            
Total training samples: 85401                                                                                                                           
From 478 symbols                                                                                                                                        
Class distribution: {0: 54195, 1: 31206}                                                                                                                
Train samples: 68320, Validation samples: 17081                                                                                                         
Applying SMOTE oversampling...                                                                                                                          
After SMOTE: Class 0=43363, Class 1=43363                                                                                                               
Training XGBoost model...                                                                                                                               
Validation Accuracy: 38.39%                                                                                                                             
Validation Precision: 100.00%                                                                                                                           
Validation Recall: 38.39%                                                                                                                               
Expected Value per Trade (RR 2.0:1): 2.000R                                                                                                             
  Created dataset: sp500_model_rr2_dataset_20260328_163010                                                                                              
✓ Model saved: sp500_model_rr2_RR2.0_20260328_163010                                                                                                    
  Model ID: 14                                                                                                                                          
                                                                                                                                                        
============================================================                                                                                            
VALIDATING ON OUT-OF-SAMPLE TEST DATA                                                                                                                   
============================================================                                                                                            
Test samples: 23641                                                                                                                                     
Test symbols: 478                                                                                                                                       
                                                                                                                                                        
Threshold    Precision    Recall       Signals    Expected Value                                                                                        
----------------------------------------------------------------------                                                                                  
0.50      48.07%      2.02%         362      +0.442R      ✓                                                                                             
0.55      55.70%      0.51%          79      +0.671R      ✓                                                                                             
0.60      70.00%      0.08%          10      +1.100R      ✓                                                                                             

============================================================
OUT-OF-SAMPLE TEST SUMMARY
============================================================
Best Threshold: 0.60
Expected Value: 1.100R per trade
Test Period: 7 years after 2019-12-31

Break-even Precision Needed: 33.3%

✓ Model is PROFITABLE on out-of-sample data!
  For every $100 risked, expect $110 profit

Recommended trading parameters:
  - Probability threshold: 0.60
  - Risk:Reward ratio: 2.0:1
  - Expected win rate: 70.0%

Test results saved to: test_results_sp500_model_rr2_RR2.0_20260328_163010_20260328_163010.json

============================================================
FEATURE IMPORTANCE ANALYSIS
============================================================
Top 10 Most Important Features:
  1. stoch_oversold: 0.1695
  2. stoch_overbought: 0.0479
  3. return_5d: 0.0458
  4. atr_pct: 0.0445
  5. return_1d: 0.0407
  6. stoch_cross_below: 0.0405
  7. above_vwap: 0.0403
  8. regression_signal: 0.0403
  9. stoch_cross_above: 0.0396
  10. regression_cross_up: 0.0395

