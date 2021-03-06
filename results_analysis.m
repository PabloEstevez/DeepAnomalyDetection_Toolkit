close all

%% MEDA Toolbox (https://github.com/josecamachop/MEDA-Toolbox)

% UNIX
MEDA_PATH = '~/github/MEDA-Toolbox';

% Windows
% MEDA_PATH = 'C:\Users\pablo\OneDrive\Escritorio\MEDA-Toolbox';

addpath(MEDA_PATH);
addpath(strcat(MEDA_PATH, '/BigData'));
addpath(strcat(MEDA_PATH, '/GUI'));

%% Read hyperparameters configuration

fname = 'hyperparams_config.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid);
full_configuration = jsondecode(str);
config = full_configuration.DatasetList(4);

%% Config

global anomaly_class;
global normal_class;

if isfield(config, "normal_class")
    normal_class = config.normal_class;
    anomaly_class = ~config.normal_class;
else
    normal_class = 1;
    anomaly_class = 0;
end

% kk = strsplit(config.dataset_file,"_");
% kk = strsplit(char(kk(2)), "x");
% n_vars = str2double(char(kk(2)));
% n_obs = str2double(char(kk(1)));
% n_obs_test = config.train_test_split_method.test_size*n_obs;

n_vars = 140;
n_obs = 1000;
n_obs_test = 200;

beta = 1;

%% ROC

kk = strsplit(config.dataset_file, "/");
autoencoders_results_files = dir(strcat("logs/mat/",strrep(kk(end),'.csv','*.mat')));
results_table = cell2table(cell(0,9), 'VariableNames', {'test_labels', 'pred_loss', 'threshold', 'pred_labels', 'ROC_AUC', 'precision', 'recall', 'accuracy', 'F1'});

figure(1)
legend_index = 1;
anom_percentages = [0 0.1 0.2 0.5];
line_styles = {'-' '.-' '--' '-.'};

for i=1:length(autoencoders_results_files)
    kk = strsplit(autoencoders_results_files(i).name,"_");
    network_type = strsplit(char(kk(end)),".");
    network_type = char(network_type(1));
%     network_type = strrep(network_type,"AS-",""); % Remove preprocessing string
    kk = strsplit(char(kk(end-1)),"gen");
    n_anom_gen = kk(1);
    
    S = load(strcat("logs/mat/",autoencoders_results_files(i).name));
    
    [network_fpr,network_tpr,network_thresholds,network_AUC] = perfcurve(S.test_labels,S.pred_loss,anomaly_class);
    [~,network_closest_threshold] = min(abs(network_thresholds-S.threshold));
    
    pred_labels = ~(S.pred_loss > S.threshold);
    [conf_matrix, ~] = confMatrix(S.test_labels, pred_labels);
    precision = conf_matrix(1,1)/(conf_matrix(1,1)+conf_matrix(2,1));
    recall = conf_matrix(1,1)/(conf_matrix(1,1)+conf_matrix(1,2));
    accuracy = (conf_matrix(1,1)+conf_matrix(2,2))/sum(sum(conf_matrix));
    F1 = (1+beta^2)*precision*recall/(precision*beta^2 + recall);
    
    results_table(i,:) = [struct2table(S, 'AsArray', 1) table(pred_labels, network_AUC, precision, recall, accuracy, F1)];
    results_row_names(i) = strcat(network_type, "_", n_anom_gen);
    
%     plot(network_fpr,network_tpr,char(line_styles(n_vars*anom_percentages==str2double(char(n_anom_gen)))));
    if contains(network_type, 'PCA')
        line_style = '-.';
    elseif contains(network_type, '-S-Autoencoder')
        line_style = '-';
    else
        line_style = '.-';
    end
    plot(network_fpr,network_tpr,line_style);
    legend_arr(legend_index) = {network_type};%{strcat(network_type, " (\Psi=", n_anom_gen, ")")};%{strcat(network_type, " ", n_anom_gen, " anom. vars")};
    legend_index = legend_index + 1;
    hold on;
    %plot(autoencoder_fpr(autoencoder_closest_threshold),autoencoder_tpr(autoencoder_closest_threshold),'.','MarkerSize',20)
end

plot([0 1],[0 1],'--r')
legend_arr(legend_index) = {"Random classifier"};
legend_index = legend_index + 1;
hold off
xlabel('False positive rate') 
ylabel('True positive rate')
title(strcat('ROC curve (1000x', num2str(n_vars), ' dataset)'));
legend(legend_arr,'Location','southeast');

results_table.Properties.RowNames = cellstr(results_row_names);
disp(results_table(:,5:end))
