function [NameList,NameID] = create_name_ID(NR)

NameList = cell(1,2);
NameID = zeros(size(NR));
Fcount = 1; 
Mcount = 1;
NameList(1,:) = NR(1,1:2,1);
for k = 1:size(NR,3)
    disp(k); tic;
    for i = 1:size(NR,1)
        M = NR{i,1,k}; F = NR{i,2,k};
        ptr = 1; isF = 0; isM = 0;M_ID = 0; F_ID = 0;
        while ptr <= size(NameList,1) && ( ~isM || ~isF )
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Method 1 ( Runtime evaluation ==> slower)
%             isM = max(isM,strcmp(M,NameList{ptr,1}));
%             isF = max(isF,strcmp(F,NameList{ptr,2}));
                        
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Method 2
            
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
        if (~isM) 
            NameList{Mcount+1,1} = M; Mcount = Mcount + 1;
            NameID(i,1,k) = Mcount;
        else NameID(i,1,k) = M_ID; end
        if (~isF) 
            NameList{Fcount+1,2} = F; Fcount = Fcount + 1; 
            NameID(i,2,k) = Fcount;
        else NameID(i,2,k) = F_ID; end 
    end
    toc;
end