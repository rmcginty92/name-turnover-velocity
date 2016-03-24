function [name_list,ssid_rd] = create_name_ID2(ss_rd)
name_list = cell(size(ss_rd,1)*10,2,2);
nl_N = size(name_list,1);
ssid_rd = zeros(size(ss_rd));
for k = 1:size(ss_rd,3)
    display_progress(['Status: ',num2str(k/134*100),'%']); 
    tic;
    for i = 1:size(ss_rd,1)
        M = ss_rd{i,1,k}; F = ss_rd{i,2,k};
        
        % Male Check
        if ~isempty(M)
            indM = mod(string2hash(M),nl_N);
            nxt = 1;
            has_id = strcmp(M,name_list{indM,1,1});
            while ~isempty(name_list{indM,1,1}) && ~has_id
                indM = mod(indM.^nxt + M(1).^nxt + M(end).^nxt,nl_N) + 1;
                nxt=mod(nxt,20)+1;
                has_id = strcmp(M,name_list{indM,1,1});
            end
            if ~has_id
                name_list{indM,1,1} = M;
                name_list{indM,1,2} = k;
            end
            ssid_rd(i,1,k) = indM;
        end
        
        % Female Check
        if ~isempty(F)
            indF = mod(string2hash(F),nl_N);
            nxt = 1;
            has_id = strcmp(F,name_list{indF,2,1});
            while ~isempty(name_list{indF,2,1}) && ~has_id
                indF = mod(indF.^nxt + F(1).^nxt + F(end).^nxt,nl_N) + 1;
                nxt=mod(nxt,20)+1;
                has_id = strcmp(F,name_list{indF,2,1});
            end
            if ~has_id
                name_list{indF,2,1} = F;
                name_list{indF,2,2} = k;
            end
            ssid_rd(i,2,k) = indF;
        end
    end
    toc;
end