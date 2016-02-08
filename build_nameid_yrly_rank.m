function [nid_rank] = build_nameid_yrly_rank(nid,nl)


nid_rank = -1*ones(size(nl,1),size(nid,2),size(nid,3));
for k = 2:size(nid,3)
    for j = 1:size(nid,2)
        for i = 1:size(nid,1)
            i_prev = find(nid(:,j,k) == nid(i,j,k-1));
            if isempty(i_prev); i_prev = size(nid,1); end
            nid_rank(nid(i,j,k),j,k) = i;
        end
        display_progress(['Completed k = ',num2str(k)]);

    end
end



end