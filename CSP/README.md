# CSP
 - proc_variance, proc_sliding_variance를 모두 bbci_public/processing/ 내부에 넣음
   
## Segment
- vector_embedding 코드에서 proc_variance(fv_tr_EEG, n_sess);
- proc_multicsp는 그대로 이용
  

## Sliding Window
- vector_embedding 코드에서 proc_sliding_variance(fv_tr_EEG, window_len, n_sess, 0);
- proc_multicsp는 그대로 이용
