function [vort] = computeCurl(uv)
% compute the vorticity from the velocity field
% Edited by Shengze Cai
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


u = uv(:,:,1);
v = uv(:,:,2);

%% simple kernel
u1 = [u(1,:);u;u(end,:)];       % add boundary 
u = [u1(:,1),u1,u1(:,end)];
v1 = [v(1,:);v;v(end,:)];       % add boundary 
v = [v1(:,1),v1,v1(:,end)];
kernel = [-1 0 1;
        -2 0 2;
        -1 0 1]./8;

uy = conv2(u,kernel','same');
vx = conv2(v,kernel,'same');
uy = uy(2:end-1,2:end-1);
vx = vx(2:end-1,2:end-1);
vort = vx-uy;

