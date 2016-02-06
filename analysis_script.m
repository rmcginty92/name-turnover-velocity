%% Baby-Name Analysis 

%% Importing Data

clear all; close all; clc;
import_name_data;

NC = name_count;

%% The Simple Stuff
% First, it'd be nice to see some simple characteristics of the data,
% namely standard Deviation, summations, highest ranked as a ratio etc.
% This kind of info will help disclose and overall trend of How many names
% are being used, how concentrated the top names are being chosen.
close all;

% Normalized Standard Deviation

figure;
NC = name_count;
norm_NC = NC./repmat(NC(1,:,:),size(NC,1),1,1);
std_NC = squeeze(std(norm_NC,0,1));
plot(yrs,std_NC','-*');
title('Normalized Standard Deviation of Name Counts');
xlabel('Year'); ylabel('Standard Deviation');
legend('Male','Female');

% Showing the Max values and summation through the years
figure;
for i = 1:2
    Gend_max = squeeze(NC(1,i,:));
    Gend_sum = squeeze(sum(NC(:,i,:),1));
    Gend_max = Gend_max / max(Gend_max);
    Gend_sum = Gend_sum / max(Gend_sum);
    subplot(2,1,i);
    plot(yrs,Gend_max,'-*');hold on;
    plot(yrs,Gend_sum,'r-*');
end




% Max Count / Sum of Counts
N = [5,10,25,50,100,250];
sum_NC = squeeze(sum(NC,1));
figure;
for i = 1:length(N)
    subplot(3,2,i);
    maxi_NC = squeeze(sum(NC(1:N(i),:,:),1));
    ratioi_NC = maxi_NC ./ sum_NC;
    plot(yrs,ratioi_NC');
    str = ['Ratio of ' num2str(N(i)) ' Highest Names Used to All Names'];
    title(str);
    xlabel('Year'); ylabel('Percentage');
    legend('Male','Female');
end

max10_NC = squeeze(sum(NC(1:10,:,:),1));
sum_NC = squeeze(sum(NC,1));
ratio_NC = max10_NC ./ sum_NC;
figure;
plot(yrs,ratio_NC','-*');
title('Ratio of 10 Highest Names Used to All Names');
xlabel('Year'); ylabel('Percentage');
legend('Male','Female');

% fprintf('Take a Break! Press any key to continue...\n'); pause

%% Analysis of Name-Turnover

% First create Unique IDs for all names for easier computation.
if ~exist('NameList','var') || ~exist('NameID','var') 
    [NameList,NameID] = create_name_ID(name_rank); 
    save('NameDataFull.mat','name_count','name_rank','yrs','NameList','NameID');
end

NameIDi = calculate_yearly_diff(NameID);

figure;
sumyrlyDelta = squeeze(sum(abs(NameIDi),1));
plot(yrs,medfilt1(squeeze(sumyrlyDelta)',10));



% Comparing the name differential in top 10, 25, 50, 100, 250 and 1000
% names
N = [10,25,50,100,250,1000];
figure;
for i = 1:length(N)
    subplot(3,2,i);
    sumyrlyDelta = squeeze(sum(abs(NameIDi(1:N(i),:,:)),1));
    plot(yrs,medfilt1(squeeze(sumyrlyDelta)',10));
    str = ['Top ' num2str(N(i)) ' Name Differentials'];
    title(str);
    xlabel('Year'); ylabel('Summation of Yearly differentials');
    legend('Male','Female');
end


% Varying Median filter
N = [3,5,7,10,15,20];
sumyrlyDelta = squeeze(sum(abs(NameIDi(1:50,:,:)),1));

figure;
for i = 1:length(N)
    subplot(3,2,i);
    plot(yrs,medfilt1(squeeze(sumyrlyDelta)',N(i)));
    str = ['Top 50 Name Differentials with ' num2str(N(i)) ' windowed Median Filter'];
    title(str);
    xlabel('Year'); ylabel('Summation of Yearly differentials');
    legend('Male','Female');

end

% Varying an averaging filter
N = [3,5,7,10,15,20];
figure;
for i = 1:length(N)
    subplot(3,2,i);
    b = 1/N(i)*ones(N(i),1);
    plot(yrs,filter(b,1,squeeze(sumyrlyDelta)',N(i)));
    str = ['Top 50 Name Differentials with ' num2str(N(i)) ' windowed Median Filter'];
    title(str);
    xlabel('Year'); ylabel('Summation of Yearly differentials');
    legend('Male','Female');
end



% fprintf('Take a Break! Press any key to continue...\n'); pause
