if ~exist('NameDataFull.mat','file')
    file_dir = fullfile(pwd,'data');
    fnames = dir(file_dir);fnames(1) = [];fnames(1) = [];
    N = length(fnames); 
    if ~exist('lim','var') lim = 1000; end
    name_rank = cell(lim,2,N);
    name_count = zeros(lim,2,N);
    yrs = zeros(1,N);
    for i = 1:N
        i
        [name_rank(:,:,i),name_count(:,:,i),yrs(i)] = extract_name_rankings(fnames(i).name,lim);
        
    end
    save('NameDataFull.mat','name_rank','name_count','yrs');
else 
    load('NameDataFull.mat');
end