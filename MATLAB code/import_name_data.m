if ~exist('NameDataFull.mat','file')
    file_dir = fullfile(pwd,'data');
    fnames = dir(file_dir);fnames(1) = [];fnames(1) = [];
    N = length(fnames); 
    if ~exist('lim','var') lim = 20000; end
    ss_rank_data = cell(lim,2,N);
    ss_count_data = zeros(lim,2,N);
    yrs = zeros(1,N);
    for i = 1:N
        [ss_rank_data(:,:,i),ss_count_data(:,:,i),yrs(i)] = extract_name_rankings(fnames(i).name,lim);
        display_progress(['Status: ',num2str(i/N*100),'%']  );
    end
    [name_list,ssid_rank_data] = create_name_ID2(ss_rank_data);
    ssid_rank_data(ssid_rank_data ==0) = -1;
    save('NameDataFull.mat','ss_rank_data','ss_count_data','ssid_rank_data','name_list','yrs');
elseif ~exist('ss_rank_data','var') || ~exist('ss_count_data','var') ||...
       ~exist('yrs','var') || ~exist('ssid_rank_data','var') ||...
       ~exist('name_list','var')
    load('NameDataFull.mat');
end

clear file_dir fnames 