load("C:\Users\user\OneDrive\바탕 화면\matlab_code\voice recon data\sub001\epo_imagined.mat");
epo_imagined = struct('x', permute(x, [1, 2, 3]), 'y', y, 'y_dec', y_dec, 'clab', clap);
%epo_imagined.x = epo_imagined.x*-100000;
load("C:\Users\user\OneDrive\바탕 화면\matlab_code\voice recon data\sub001\epo_spoken.mat");
epo_spoken = struct('x', permute(x, [1, 2, 3]), 'y', y, 'y_dec', y_dec, 'clab', clap);
%epo_spoken.x = epo_spoken.x*-100000;


% setting parameters
numcsp = 4; % the number of CSP features
n_sess = 172; % the number of time segments
num_class = 13; % the number of labels
n_fold = 4; % n-fold cross validation
seed = 0; % seed number
window_len = 125;

rng(seed)
r_ts_val = randperm(36, 2*36/n_fold);

r_val = r_ts_val(1:36/n_fold);
r_ts = r_ts_val(36/n_fold+1:end);
r_tr = setdiff(1:36, r_ts_val);

% random n-fold of each class
y_dec=[];e_tr=[];e_ts=[];e_val=[];
for dd = 1:num_class
    y_dec{dd} = find(epo_imagined.y_dec==dd);
    e_tr = [e_tr, y_dec{dd}(r_tr)];
    e_ts = [e_ts, y_dec{dd}(r_ts)];
    e_val = [e_val, y_dec{dd}(r_val)];
end

epo_tr_spoken = proc_selectEpochs(epo_spoken, e_tr);
epo_ts_spoken = proc_selectEpochs(epo_spoken, e_ts);
epo_val_spoken = proc_selectEpochs(epo_spoken, e_val);

epo_tr_imagined = proc_selectEpochs(epo_imagined, e_tr);
epo_ts_imagined = proc_selectEpochs(epo_imagined, e_ts);
epo_val_imagined = proc_selectEpochs(epo_imagined, e_val);

% y lable to one hot encoding label
[~,temp] = max(epo_tr_spoken.y); % argmax
epo_tr_spoken.y_dec = temp';
[~,temp] = max(epo_ts_spoken.y); % argmax
epo_ts_spoken.y_dec = temp';
[~,temp] = max(epo_val_spoken.y); % argmax
epo_val_spoken.y_dec = temp';

% y lable to one hot encoding label
[~,temp] = max(epo_tr_imagined.y); % argmax
epo_tr_imagined.y_dec = temp';
[~,temp] = max(epo_ts_imagined.y); % argmax
epo_ts_imagined.y_dec = temp';
[~,temp] = max(epo_val_imagined.y); % argmax
epo_val_imagined.y_dec = temp'; 

%% Training CSP

%%%%%%%%%%%%% Imagined  %%%%%%%%%%%%
% training CSP
[fv_tr_imagined, csp_w_tr]= proc_multicsp(epo_tr_imagined,numcsp); % training
fv_tr_imagined = proc_variance(fv_tr_imagined, n_sess); %proc_variance(fv_tr_imagined, window_len, n_sess, 0);
fv_tr_imagined = proc_logarithm(fv_tr_imagined);

% test CSP
fv_te_imagined = proc_linearDerivation(epo_ts_imagined, csp_w_tr); % inference
fv_te_imagined = proc_variance(fv_te_imagined, n_sess);%proc_variance(fv_te_imagined, window_len, n_sess, 0);
fv_te_imagined = proc_logarithm(fv_te_imagined);

% val CSP
fv_val_imagined = proc_linearDerivation(epo_val_imagined, csp_w_tr); % inference
fv_val_imagined = proc_variance(fv_val_imagined, n_sess);%proc_variance(fv_val_imagined, window_len,n_sess, 0);
fv_val_imagined = proc_logarithm(fv_val_imagined);

%%%%%%%%%%%%% Imagined  %%%%%%%%%%%%
% train CSP
fv_tr_spoken = proc_linearDerivation(epo_tr_spoken, csp_w_tr); % inference
fv_tr_spoken= proc_variance(fv_tr_spoken, n_sess);%proc_variance(fv_tr_spoken, window_len,n_sess, 0);
fv_tr_spoken= proc_logarithm(fv_tr_spoken);

% test CSP
fv_te_spoken = proc_linearDerivation(epo_ts_spoken, csp_w_tr); % inference
fv_te_spoken = proc_variance(fv_te_spoken,n_sess);%proc_variance(fv_te_spoken,window_len,n_sess, 0);
fv_te_spoken = proc_logarithm(fv_te_spoken);

% val CSP
fv_val_spoken = proc_linearDerivation(epo_val_spoken, csp_w_tr); % inference
fv_val_spoken = proc_variance(fv_val_spoken,n_sess);%proc_variance(fv_val_spoken,window_len,n_sess, 0);
fv_val_spoken = proc_logarithm(fv_val_spoken);

