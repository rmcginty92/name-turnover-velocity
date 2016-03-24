function [NameList,NameID] = create_name_ID(NR)

NameList = cell(1,2,2);
NameID = zeros(size(NR));
Fcount = 1; 
Mcount = 1;
NameList(1,:,1) = NR(1,1:2,1);
NameList(1,:,2) = {[1],[1]};
for k = 1:size(NR,3)
    display_progress(k); tic;
    for i = 1:size(NR,1)
        M = NR{i,1,k}; F = NR{i,2,k};
        ptr = 1; isF = 0; isM = 0;M_ID = 0; F_ID = 0;
        while ptr <= size(NameList,1) && ( ~isM || ~isF )
            % 
            if ~isM && strcmp(M,NameList{ptr,1})
                M_ID = ptr;
                isM = 1;
            end
            if ~isF && strcmp(F,NameList{ptr,2})
                F_ID = ptr;
                isF = 1;
            end          
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ptr = ptr + 1;
        end
        if (~isM && ~isempty(M)) 
            NameList{Mcount+1,1,1} = M;
            NameList{Mcount+1,1,2} = k;
            Mcount = Mcount + 1;
            NameID(i,1,k) = Mcount;
        else
            NameID(i,1,k) = M_ID;
        end
        if (~isF && ~isempty(F)) 
            NameList{Fcount+1,2,1} = F; 
            NameList{Fcount+1,2,2} = k; 
            Fcount = Fcount + 1; 
            NameID(i,2,k) = Fcount;
        else
            NameID(i,2,k) = F_ID;
        end 
    end
    toc;
end