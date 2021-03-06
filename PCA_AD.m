close all

%% MEDA Toolbox

MEDA_PATH = '~/github/MEDA-Toolbox';
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
config = full_configuration.DatasetList(3);

%% Import dataset
dataset_table = readtable(config.dataset_file);
% dataset_table = readtable('../Datasets/ecg.csv');
% [n_measures, n_variables] = size(dataset_table);
% dataset_table = sortrows(dataset_table, [n_variables], 'descend');
% if ~strcmpi(dataset_table.Properties.VariableNames(end), {'Class'})
%     disp('Using last feature as Class')
%     dataset_table.Properties.VariableNames(end) = {'Class'};
% end
% dataset_table(12,:) = []; % Remove Senegal

%%%% CONFIG %%%%
global anomaly_class;
global normal_class;

if isfield(config, "normal_class")
    normal_class = config.normal_class;
    anomaly_class = ~config.normal_class;
else
    normal_class = 1;
    anomaly_class = 0;
end

if isfield(config.train_test_split_method, "method_name")
    split_method = config.train_test_split_method.method_name;
else
    split_method = "random";
end

if isfield(config.train_test_split_method, "test_size")
    test_size = config.train_test_split_method.test_size;
    train_size = 1-test_size;
    train_test_cut_point = floor(train_size*height(dataset_table));
else
    train_test_cut_point = config.train_test_split_method.train_test_cut_point;
end
%%%%%%%%%%%%%%%%

dataset_matrix = table2array(dataset_table);

if split_method == "random"
    % Shuffle dataset
    dataset_matrix = dataset_matrix(randperm(size(dataset_matrix,1)),:);
end

labels = dataset_matrix(:,end);
data = dataset_matrix(:,1:end-1);

% Train dataset
train_data = data(1:train_test_cut_point, :);
train_labels = labels(1:train_test_cut_point);
normal_train_data = train_data(train_labels==normal_class,:);
% anomalous_train_data = train_data(train_labels==anomaly_class,:);

% Test dataset
test_data = data(train_test_cut_point+1:end, :);
% test_labels = labels(train_test_cut_point+1:end);
% normal_test_data = test_data(test_labels==normal_class,:);
% anomalous_test_data = test_data(test_labels==anomaly_class,:);

%% Anomaly generation
[normal_train_data_height,n_vars] = size(normal_train_data);
[test_data_height, ~] = size(test_data);
if config.n_anom_vars > 0
    % test_data = [test_data(1:test_data_height/2,:); 3*test_data(test_data_height/2+1:end,:)];
    n_anom_vars = config.n_anom_vars;
%     anom_gain = 5*sqrt(n_vars)*log10(n_vars/2)/n_anom_vars;
    anom2 = test_data(test_data_height/2+1:end,:);
    anom_add = 1.5 * sum(anom2'.^2)'/n_anom_vars;
    anom2(:,1:n_anom_vars) = anom2(:,1:n_anom_vars) + anom_add;
    test_data = [test_data(1:test_data_height/2,:); anom2];
    test_labels = [normal_class*ones(test_data_height/2, 1); anomaly_class*ones(test_data_height/2, 1)];
else
    test_labels = labels(train_test_cut_point+1:end);
    n_anom_vars = 0;
end

%% Pre-processing comparison
% figure(1);
% 
% % No pre-process
% subplot(1,3,1);
% scatter3(train_dataset(:,1),train_dataset(:,2),train_dataset(:,3));
% title("Raw dataset");
% 
% % Mean centering
% subplot(1,3,2);
% train_dataset_prep = preprocess2D(train_dataset,1);
% scatter3(train_dataset_prep(:,1),train_dataset_prep(:,2),train_dataset_prep(:,3))
% title("Mean centering");
% 
% % Auto-scaling
% subplot(1,3,3);
% train_dataset_prep = preprocess2D(train_dataset,2);
% scatter3(train_dataset_prep(:,1),train_dataset_prep(:,2),train_dataset_prep(:,3))
% title("Autoscaling");

%% Pre-process

% Vaccinations normalization
% dataset_table.vaccination_index = dataset_table.people_fully_vaccinated./dataset_table.population;

% Save countries on array
% countries = dataset_table.country;
% dataset_table.Properties.RowNames = countries;

% Delete unwanted features
% dataset_table = removevars(dataset_table,{'country','population','people_fully_vaccinated', 'population_density','aged_65_older','aged_70_older','cardiovasc_death_rate','diabetes_prevalence','vaccination_index'});

prep_method_names = ["NoPrep","MC","AS"];

% Save feature labels
features = dataset_table.Properties.VariableNames;

prep_method = 2;%config.prep_method;   % Autoscaling
prep_method_name = prep_method_names(prep_method+1);

[train_dataset_prep, prep_mean, prep_scale] = preprocess2D(normal_train_data,prep_method);
test_dataset_prep = preprocess2Dapp(test_data,prep_mean,prep_scale);

%% PCA

pcs = 3;config.PCs;    % Principal Components

% figure(2);
% scatter3(train_dataset_prep(:,1),train_dataset_prep(:,2),train_dataset_prep(:,3))
% hold on;
% [P,T] = pca_pp(train_dataset_prep,1:pcs);
% 
% % Notas PCA
% % (B=dataset, C=covarianzas, V=autovectores, D=autovalores_o_loadings, T=principal_components)
% % C = B' * B
% % T = B * V
% 
% % Eigenvectors
% eigenvalues = [3 1.3 0.8]; % eigenvalues = V^-1 * C * V = D 
% eigencolors = ['g','r','y'];
% for i=1:pcs
%     plot3(eigenvalues(i)*[-P(1,i) P(1,i)],eigenvalues(i)*[-P(2,i) P(2,i)],eigenvalues(i)*[-P(3,i) P(3,i)],eigencolors(i),'linewidth',5);
% end
% % Model
% fmesh(@(s,t)P(1,1)*s+P(1,2)*t,@(s,t)P(2,1)*s+P(2,2)*t,@(s,t)P(3,1)*s+P(3,2)*t,'EdgeColor','cyan')
% hold off
% legend("Data","Eigenvector 1","Eigenvector 2","Eigenvector 3");

% threshold_percentile = config.threshold_percentile;
threshold_percentile = 98;

p_valueD = 1 - threshold_percentile/100;
p_valueQ = 1 - threshold_percentile/100;

[Dst,Qst,Dstt,Qstt,D_threshold,Q_threshold] = mspc_pca(train_dataset_prep,1:pcs,test_dataset_prep,0,'000',[],[normal_class*ones(normal_train_data_height,1); test_labels],p_valueD,p_valueQ,1);

var_pca(train_dataset_prep,1:40,0,'10');
% var_pca(dataset,1:length(countries),prep);
% mspc_pca(dataset,1:pcs,[],prep);
% omeda_pca(dataset,1:2,dataset(12,:),1);

PCA_ROC_scores = (pcs/n_vars)*Dstt/D_threshold + ((n_vars-pcs)/n_vars)*Qstt/Q_threshold;
QD_threshold = (pcs/n_vars) + ((n_vars-pcs)/n_vars);

% PCA_ROC_scores = Qstt;
% QD_threshold = Q_threshold;

test_labels = transpose(test_labels);
pred_loss = transpose(PCA_ROC_scores);
threshold = QD_threshold;
kk = strsplit(config.dataset_file,"/");
kk = strrep(kk(end), ".csv", strcat("_",num2str(n_anom_vars),"gen_",prep_method_name,"-PCA"));
if config.enable_logging
    %save(strcat("logs/mat/",kk),'test_labels','pred_loss','threshold')
end
%% Validation

% pred_labels = (Qstt > Q_threshold) | (Dstt > D_threshold);
pred_labels = pred_loss > threshold;
if anomaly_class == 0
    pred_labels = ~pred_labels;
end

[conf_matrix, conf_table] = confMatrix(test_labels, pred_labels);
precision = conf_matrix(1,1)/(conf_matrix(1,1)+conf_matrix(2,1));
recall = conf_matrix(1,1)/(conf_matrix(1,1)+conf_matrix(1,2));
accuracy = (conf_matrix(1,1)+conf_matrix(2,2))/sum(sum(conf_matrix));
beta = 1;
F1 = (1+beta^2)*precision*recall/(precision*beta^2 + recall);

% Confusion matrix
% conf_matrix = zeros(2,2);
% for sample=1:length(pred_labels)
%     if test_labels(sample) == pred_labels(sample)
%         if test_labels(sample) == anomaly_class
%             i = 1; j = i;
%         else
%             i = 2; j = i;
%         end
%     else
%         if test_labels(sample) == anomaly_class
%             i = 1; j = 2;
%         else
%             i = 2; j = 1;
%         end
%     end
%     
%     conf_matrix(i,j) = conf_matrix(i,j) + 1;
% end

% conf_table = array2table(conf_matrix,'RowNames',{'Positive';'Negative'},'VariableNames',{'Positive';'Negative'});

disp("Confusion matrix:")
disp(conf_table)

%% D-st & Q-st scatter

figure(3)
clf

decimation_step = 1;

test_normal_Dstt = Dstt(test_labels==normal_class);
test_anomalous_Dstt = Dstt(test_labels==anomaly_class);
test_normal_Qstt = Qstt(test_labels==normal_class);
test_anomalous_Qstt = Qstt(test_labels==anomaly_class);

scatter(Dst(1:decimation_step:end),Qst(1:decimation_step:end), 'filled')
hold on
scatter(test_normal_Dstt(1:decimation_step:end),test_normal_Qstt(1:decimation_step:end), 'filled')
scatter(test_anomalous_Dstt(1:decimation_step:end),test_anomalous_Qstt(1:decimation_step:end), 'filled')%, 'MarkerFaceAlpha', 0.2)
% xline(D_threshold,'--r');
% yline(Q_threshold,'--r');
line([D_threshold D_threshold],[0 max([Qst; Qstt])],'Color','red','LineStyle','--')
line([0 max([Dst; Dstt])],[Q_threshold Q_threshold],'Color','red','LineStyle','--')
hold off
xlim([0 max([Dst; Dstt])])
ylim([0 max([Qst; Qstt])])
xlabel('D-st')
ylabel('Q-st')
legend('Train','Test (normal)','Test (anomalous)', 'D-threshold', 'Q-threshold')

figure(5)
normal_test_loss = Qstt(1:length(Qstt)/2);
anomalous_test_loss = Qstt(1+length(Qstt)/2:end);
histogram(Qst);
hold on
histogram(normal_test_loss);
histogram(anomalous_test_loss);
h.FaceColor = 'r';
hold off
line([Q_threshold Q_threshold], [0 length(Qst)/2],'Color','red','LineStyle','--')
xlabel("Loss")
ylabel("No of examples")
legend("Train", "Test (normal)", "Test (anomalous)")

%% ROC

% Load autoencoder results
kk = strsplit(config.dataset_file, "/");
autoencoders_results_files = dir(strcat("logs/mat/",strrep(kk(end),'.csv','*.mat')));
results_table = cell2table(cell(0,4), 'VariableNames', {'test_labels', 'pred_loss', 'threshold', 'ROC_AUC'});

figure(4)
legend_index = 1;
anom_percentages = [0 0.1 0.2 0.5];
line_styles = {'-' '.-' '--' '-.'};
    
[network_fpr,network_tpr,network_thresholds,network_AUC] = perfcurve(test_labels,pred_loss,anomaly_class);
[~,network_closest_threshold] = min(abs(network_thresholds-threshold));

plot(network_fpr,network_tpr,char(line_styles(n_vars*anom_percentages==n_anom_vars)));
hold on;
plot(network_fpr(network_closest_threshold),network_tpr(network_closest_threshold),'.','MarkerSize',20)

% PCA_ROC_scores = (pcs/n_vars)*Dstt/D_threshold + ((n_vars-pcs)/n_vars)*Qstt/Q_threshold;
% QD_threshold = (pcs/n_vars) + ((n_vars-pcs)/n_vars);
% [PCA_fpr,PCA_tpr,PCA_thresholds,PCA_AUC] = perfcurve(test_labels,PCA_ROC_scores,anomaly_class);
% [~,PCA_closest_threshold] = min(abs(PCA_thresholds-QD_threshold));
% 
% [PCA_Q_fpr,PCA_Q_tpr,PCA_Q_thresholds,PCA_Q_AUC] = perfcurve(test_labels,Qstt,anomaly_class);
% [~,PCA_Q_closest_threshold] = min(abs(PCA_Q_thresholds-Q_threshold));
% 
% plot(PCA_fpr,PCA_tpr)
% legend_arr(legend_index) = {strcat("PCA ", num2str(n_anom_vars), " anom. vars")};
% legend_index = legend_index + 1;
% plot(PCA_Q_fpr,PCA_Q_tpr)
% legend_arr(legend_index) = {strcat("PCA_Q ", num2str(n_anom_vars), " anom. vars")};
% legend_index = legend_index + 1;

%plot(PCA_fpr(PCA_closest_threshold),PCA_tpr(PCA_closest_threshold),'.','MarkerSize',20)
%plot(PCA_Q_fpr(PCA_Q_closest_threshold),PCA_Q_tpr(PCA_Q_closest_threshold),'.','MarkerSize',20)
plot([0 1],[0 1],'--r')
hold off
xlabel('False positive rate') 
ylabel('True positive rate')
title(strcat('ROC curve (1000x', num2str(n_vars), ' dataset)'));
%legend(strcat('PCA (AUC=',num2str(PCA_AUC),')'), strcat('PCA_Q (AUC=',num2str(PCA_Q_AUC),')'), strcat('Autoencoder (AUC=',num2str(autoencoder_AUC),')'), 'Random (AUC=0.5)','PCA selected threshold', 'PCA_Q selected threshold','Autoencoder selected threshold','Location','southeast')
legend(strcat('ROC (AUC=',num2str(network_AUC),')'),"Threshold","Random clasifier",'Location','southeast');

disp(table(network_AUC, precision, recall, accuracy, F1));

% figure(5)
% plot(PCA_fpr,PCA_tpr)
% hold on
% plot(PCA_Q_fpr,PCA_Q_tpr)
% plot(autoencoder_fpr,autoencoder_tpr)
% plot([0 1],[0 1],'--r')
% plot(PCA_fpr(PCA_closest_threshold),PCA_tpr(PCA_closest_threshold),'.','MarkerSize',20)
% plot(PCA_Q_fpr(PCA_Q_closest_threshold),PCA_Q_tpr(PCA_Q_closest_threshold),'.','MarkerSize',20)
% plot(autoencoder_fpr(autoencoder_closest_threshold),autoencoder_tpr(autoencoder_closest_threshold),'.','MarkerSize',20)
% hold off
% xlim([0 0.2]);
% ylim([0.8 1]);
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC curve (enlarged)')
% legend(strcat('PCA (AUC=',num2str(PCA_AUC),')'), strcat('PCA_Q (AUC=',num2str(PCA_Q_AUC),')'), strcat('Autoencoder (AUC=',num2str(autoencoder_AUC),')'), 'Random (AUC=0.5)','PCA selected threshold', 'PCA_Q selected threshold','Autoencoder selected threshold','Location','southeast')

%% Test

% figure(6)

% [network_fpr,network_tpr,network_thresholds,network_AUC] = perfcurve(test_labels,S.pred_loss,anomaly_class);
% [~,network_closest_threshold] = min(abs(network_thresholds-S.threshold));
