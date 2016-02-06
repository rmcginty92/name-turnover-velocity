function [names_rank, num_named, yr] = extract_name_rankings(filename,varargin)

if nargin < 1
    file_dir = fullfile(pwd,'data');
    filename = 'yob1880.txt';
    lim = 500;
elseif nargin < 2
    file_dir = fullfile(pwd,'data');
    lim = 500;
elseif length(varargin) < 2
    % discern what is passed to function 
    if isa(varargin{1},'double')
        lim = varargin{1};
        file_dir = fullfile(pwd,'data');
    else lim = 500; file_dir = varargin{1}; 
    end
else 
    if isa(varargin{1},'double')
        lim = varargin{1};
        file_dir = varargin{2};
    else
        lim = varargin{2};
        file_dir = varargin{1}; 
    end
end

% Sets path of data directory
file_path = fullfile(file_dir,filename);
    
try
    file = fopen(file_path); 
    try
        yr = str2num(filename(end-7:end-4));
    catch
        try 
            yr = input('Input Year: ');
        catch 
        end
    end
    line = fgets(file);
    names_rank = cell(lim,2);
    num_named = zeros(lim,2);
    iM = 1;
    iF = 1;
    while ischar(line)
        split_line = strsplit(line,',');
        if (split_line{2} == 'F') i = iF; j = 2;
        else i = iM; j = 1; end
        if i <= lim
            names_rank{i,j} = split_line{1};
            num_named(i,j) = str2num(split_line{3});
        end
        iF = iF + (split_line{2} == 'F');
        iM = iM + (split_line{2} ~= 'F');
        line = fgets(file);
    end
    % names_rank = 0; num_named = 0;

catch
    fprintf('Error: File cannot be opened\n');
    names_rank = -1; num_named = -1; yr = -1;

end 