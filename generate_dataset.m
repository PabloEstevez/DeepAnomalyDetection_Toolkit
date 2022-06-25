close all

%% MEDA Toolbox

MEDA_PATH = '~/github/MEDA-Toolbox';
% MEDA_PATH = 'C:\Users\pablo\OneDrive\Escritorio\MEDA-Toolbox';

addpath(MEDA_PATH);
addpath(strcat(MEDA_PATH, '/BigData'));
addpath(strcat(MEDA_PATH, '/GUI'));


%% Dataset generation

% Dataset aspect ratio constant (n_obs*n_vars)
% k = 10000*1000;
k = 500*500;

test_size = 0.2;
% n_vars_arr = [10 50 100 200 500 1000 5000 10000 50000 100000];
% n_vars_arr = [10 100 500 2500 25000];
n_vars_arr = [25000/5];

for i=1:length(n_vars_arr)
   n_vars = n_vars_arr(i);
   n_obs = 100; %k/n_vars;
   
   gen_dataset_old(n_obs, n_vars, test_size);
end

%% Dataset generation function

function [] = gen_dataset_old(n_obs, n_vars, test_size)
    train_size = 1-test_size;

    % Generate training dataset0
    n_obs_train = round(n_obs*train_size);
    level_correlation = 6;
    tic
    train_dataset = simuleMV(n_obs_train,n_vars,level_correlation);
    toc
    % Generate test dataset
    n_obs_test = round(n_obs*test_size);
    anomalies_percentage = 0.5;
    n_anomalies = floor(n_obs_test*anomalies_percentage);
    n_normal = n_obs_test-n_anomalies;
    corM = corr(train_dataset)*(n_obs_test-1)/(n_obs_train-1); % covariance for simulation

    test_dataset = simuleMV(n_obs_test,n_vars,level_correlation,corM);

    % Add anomalies to test dataset
    test_dataset(n_normal+1:end,:) = 3*test_dataset(n_normal+1:end,:);

    % PCA
    n_PCs = 2;
    pcs = 1:n_PCs;
    classes = [ones(n_obs_train,1);2*ones(n_normal,1);3*ones(n_anomalies,1)];

%     mspc_pca(train_dataset,pcs,test_dataset,2,'100',[],classes);

    % Export dataset

    Var = [train_dataset; test_dataset];
    Var = [Var ~(classes==3)];
    exported_dataset = array2table(Var);
    exported_dataset.Properties.VariableNames(width(exported_dataset)) = {'Class'};

    writetable(exported_dataset,strcat('part_simuleMV_',num2str(n_obs),'x',num2str(n_vars),'_',num2str(n_anomalies),'anom.csv'));

end

function [] = gen_dataset(n_obs, n_vars, test_size)
    train_size = 1-test_size;

    % Generate  dataset
    n_obs_train = round(n_obs*train_size);
    level_correlation = 6;
    tic
    dataset = simuleMV(n_obs,n_vars,level_correlation);
    toc
    
    n_obs_test = round(n_obs*test_size);
    anomalies_percentage = 0.5;
    n_anomalies = floor(n_obs_test*anomalies_percentage);
    n_normal = n_obs_test-n_anomalies;
    
    train_dataset = dataset(1:n_obs_train,:);
    test_dataset = dataset(n_obs_train+1:end,:);

    % Add anomalies to test dataset
    test_dataset(n_normal+1:end,:) = 3*test_dataset(n_normal+1:end,:);

    % PCA
    n_PCs = 2;
    pcs = 1:n_PCs;
    classes = [ones(n_obs_train,1);2*ones(n_normal,1);3*ones(n_anomalies,1)];

%     mspc_pca(train_dataset,pcs,test_dataset,2,'100',[],classes);

    % Export dataset

    Var = [train_dataset; test_dataset];
    Var = [Var ~(classes==3)];
    exported_dataset = array2table(Var);
    exported_dataset.Properties.VariableNames(width(exported_dataset)) = {'Class'};

    writetable(exported_dataset,strcat('simuleMV_',num2str(n_obs),'x',num2str(n_vars),'_',num2str(n_anomalies),'anom.csv'));

end
