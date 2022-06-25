close all

%% MEDA Toolbox

% MEDA_PATH = '~/github/MEDA-Toolbox';
MEDA_PATH = 'C:\Users\pablo\OneDrive\Escritorio\MEDA-Toolbox';

addpath(MEDA_PATH);
addpath(strcat(MEDA_PATH, '/BigData'));
addpath(strcat(MEDA_PATH, '/GUI'));

%% Import dataset
dataset_table = readtable('reduced_dataset.csv');
[n_measures, n_variables] = size(dataset_table);
dataset_table = sortrows(dataset_table, [n_variables], 'descend');
% dataset_table(12,:) = []; % Remove Senegal

normal_class = 1;
anomaly_class = 0;

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
dataset_table.vaccination_index = dataset_table.people_fully_vaccinated./dataset_table.population;

% Save countries on array
countries = dataset_table.country;
dataset_table.Properties.RowNames = countries;

% Delete unwanted features
dataset_table = removevars(dataset_table,{'country','population','people_fully_vaccinated', 'population_density','aged_65_older','aged_70_older','cardiovasc_death_rate','diabetes_prevalence','vaccination_index'});

% Save feature labels
features = dataset_table.Properties.VariableNames;

prep = 2;   % Autoscaling

% Train dataset
train_data = table2array(dataset_table(1:end-2,:));

% Test dataset
test_data = table2array(dataset_table(end-1:end,:));
test_labels = [1; 0];

% Pre-process
[train_dataset_prep, prep_mean, prep_scale] = preprocess2D(train_data,prep);
test_dataset_prep = preprocess2Dapp(test_data,prep_mean,prep_scale);

%% PCA

pcs = 3;    % Principal Components

figure(2);
scatter3(train_dataset_prep(:,1),train_dataset_prep(:,2),train_dataset_prep(:,3))
hold on;
[P,T] = pca_pp(train_dataset_prep,1:pcs);

% Notas PCA
% (B=dataset, C=covarianzas, V=autovectores, D=autovalores_o_loadings, T=principal_components)
% C = B' * B
% T = B * V

% Eigenvectors
eigenvalues = [3 1.3 0.8]; % eigenvalues = V^-1 * C * V = D 
eigencolors = ['g','r','y'];
for i=1:pcs
    plot3(eigenvalues(i)*[-P(1,i) P(1,i)],eigenvalues(i)*[-P(2,i) P(2,i)],eigenvalues(i)*[-P(3,i) P(3,i)],eigencolors(i),'linewidth',5);
end

% Eigenvalues
eigenvalues_matrix = inv(P)*cov(train_dataset_prep)*P;
eigenvalues = zeros(length(P),1);
for i=1:length(P)
    eigenvalues(i) = eigenvalues_matrix(i,i);
end
eigenvalues_percent = eigenvalues./sum(eigenvalues);

% Model
fmesh(@(s,t)P(1,1)*s+P(1,2)*t,@(s,t)P(2,1)*s+P(2,2)*t,@(s,t)P(3,1)*s+P(3,2)*t,'EdgeColor','cyan')
hold off
legend("Data","Eigenvector 1","Eigenvector 2","Eigenvector 3","Model (subespace)");

figure(3)
bar(eigenvalues_percent);
xlabel('PC')
ylabel('Variance percentage captured')

[train_data_height,n_vars] = size(train_data);

p_valueD = 1 - 95/100;
p_valueQ = 1 - 95/100;

[Dst,Qst,Dstt,Qstt,D_threshold,Q_threshold] = mspc_pca(train_dataset_prep,pcs,test_dataset_prep,0,'110',[],[normal_class*ones(train_data_height,1); test_labels],p_valueD,p_valueQ);

[loadings,scores] = pca_pp(train_dataset_prep,1:pcs);
var_pca(train_dataset_prep,1:train_data_height,prep,'11');
% mspc_pca(dataset,1:pcs,[],prep);
% omeda_pca(dataset,1:2,dataset(12,:),1);

%% Validation

pred_labels = (Qstt > Q_threshold) | (Dstt > D_threshold);
if anomaly_class == 0
    pred_labels = ~pred_labels;
end

% Confusion matrix
conf_matrix = zeros(2,2);
for sample=1:length(pred_labels)
    if test_labels(sample) == pred_labels(sample)
        if test_labels(sample) == anomaly_class
            i = 1; j = i;
        else
            i = 2; j = i;
        end
    else
        if test_labels(sample) == anomaly_class
            i = 1; j = 2;
        else
            i = 2; j = 1;
        end
    end
    
    conf_matrix(i,j) = conf_matrix(i,j) + 1;
end

conf_table = array2table(conf_matrix,'RowNames',{'Positive';'Negative'},'VariableNames',{'Positive';'Negative'});

disp("Confusion matrix:")
disp(conf_table)

%% D-st & Q-st scatter

figure(6)
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
% line([D_threshold D_threshold],[0 max([Qst; Qstt])],'Color','red','LineStyle','--')
% line([0 max([Dst; Dstt])],[Q_threshold Q_threshold],'Color','red','LineStyle','--')

line([D_threshold D_threshold],[0 max([Qst; Qstt])],'Color','red','LineStyle','--')
line([0 max([Dst; Dstt])],[Q_threshold Q_threshold],'Color','red','LineStyle','--')

hold off
xlim([0 max([Dst; Dstt])])
ylim([0 max([Qst; Qstt])])
xlabel('D-st')
ylabel('Q-st')
legend('Train','Test (normal)','Test (anomalous)', 'D-threshold', 'Q-threshold')
