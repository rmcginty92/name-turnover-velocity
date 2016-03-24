%% Baby-Name Analysis 

%% Importing Data

clear all; close all; clc;
import_name_data;

ss_c = ss_count_data;

%% The Simple Stuff
% First, it'd be nice to see some simple characteristics of the data,
% namely standard Deviation, summations, highest ranked as a ratio etc.
% This kind of info will help disclose and overall trend of How many names
% are being used, how concentrated the top names are being chosen.
close all;

% Normalized Standard Deviation

figure;
ss_c = ss_count_data;
norm_ss_c = ss_c./repmat(ss_c(1,:,:),size(ss_c,1),1,1);
std_ss_c = zeros(size(ss_c,2),size(ss_c,3));
for i = 1:size(ss_c,3)
    ind_m = find(ss_c(:,1,i)==0,1)-1;
    ind_m(end+1) = size(ss_c,1);
    ind_f = find(ss_c(:,2,i)==0,1)-1;
    ind_f(end+1) = size(ss_c,1);
    std_ss_c(1,i) = std(norm_ss_c(1:ind_m(1),1,i),0,1);
    std_ss_c(2,i) = std(norm_ss_c(1:ind_f(1),2,i),0,1);
end 
plot(yrs,std_ss_c','-*');
title('Normalized Standard Deviation of Name Counts');
xlabel('Year'); ylabel('Standard Deviation');
legend('Male','Female');

% Showing the Max values and summation through the years
figure;
for i = 1:2
    Gend_max = squeeze(ss_c(1,i,:));
    Gend_sum = squeeze(sum(ss_c(:,i,:),1));
    Gend_max = Gend_max / max(Gend_max);
    Gend_sum = Gend_sum / max(Gend_sum);
    subplot(2,1,i);
    plot(yrs,Gend_max,'-*');hold on;
    plot(yrs,Gend_sum,'r-*');
end

% Max Count / Sum of Counts
N = [5,10,25,50,100,250];
sum_ss_c = squeeze(sum(ss_c,1));
figure;
for i = 1:length(N)
    subplot(3,2,i);
    maxi_ss_c = squeeze(sum(ss_c(1:N(i),:,:),1));
    ratioi_ss_c = maxi_ss_c ./ sum_ss_c;
    plot(yrs,ratioi_ss_c');
    str = ['Ratio of ' num2str(N(i)) ' Highest Names Used to All Names'];
    title(str);
    xlabel('Year'); ylabel('Percentage');
    legend('Male','Female');
end

max10_ss_c = squeeze(sum(ss_c(1:10,:,:),1));
sum_ss_c = squeeze(sum(ss_c,1));
ratio_ss_c = max10_ss_c ./ sum_ss_c;
figure;
plot(yrs,ratio_ss_c','-*');
title('Ratio of 10 Highest Names Used to All Names');
xlabel('Year'); ylabel('Percentage');
legend('Male','Female');

% fprintf('Take a Break! Press any key to continue...\n'); pause

%% Analysis of Name-Turnover
% This section analyzes the differences of rankings between years for every
% name listed. This gives insight as to how frequently nemes might change
% year to year. 

% First create Unique IDs for all names for easier computation.
if ~exist('name_list','var') || ~exist('ssid_rank_data','var') 
    [name_list,ssid_rank_data] = create_name_ID2(ss_rank_data); 
    save('NameDataFull.mat','ss_count_data','ss_rank_data','yrs','name_list','ssid_rank_data');
end

vo = calc_velocity(ssid_rank_data);
vo2 = square_calc(vo)
% Comparing the name differential in top 10, 25, 50, 100, 250 and 1000
% names
N = [10,25,50,100,250,1000];
figure;
for i = 1:length(N)
    subplot(3,2,i);
    diff_total = squeeze(sum(abs(vo(1:N(i),:,2:end)),1));
    plot(yrs(2:end),filter(ones(5,1)/5,1,squeeze(diff_total)'));
    str = ['Top ' num2str(N(i)) ' Name Differentials'];
    title(str);
    xlabel('Year'); ylabel('Summation of Yearly differentials');
    legend('Male','Female');
end


% Varying Median filter
N = [3,5,7,10,15,20];
diff_total = squeeze(sum(abs(vo(1:50,:,:)),1));

figure;
for i = 1:length(N)
    subplot(3,2,i);
    plot(yrs,medfilt1(squeeze(diff_total)',N(i)));
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
    plot(yrs,filter(b,1,squeeze(diff_total)',N(i)));
    str = ['Top 50 Name Differentials with ' num2str(N(i)) ' windowed Median Filter'];
    title(str);
    xlabel('Year'); ylabel('Summation of Yearly differentials');
    legend('Male','Female');
end



% fprintf('Take a Break! Press any key to continue...\n'); pause
%%
% More Analysis

diff_output = calc_velocity(ssid_rank_data);
vo = diff_output;
%%
plot(yrs,sum(squeeze(abs(vo(1:15,1,:).^2)),1))




